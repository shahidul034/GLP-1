#!/usr/bin/env python3
"""Label-blind LLM re-judging using google/gemma-4-31B-it via vLLM endpoint.

Sets CUDA_VISIBLE_DEVICES=4 for GPU 4 (H200).
Uses the OpenAI-compatible API at http://localhost:8091/v1.

Prompt includes only:
  - task description
  - 7 label definitions
  - clinical note text
  - JSON output schema

Saves predictions to data/llm_blind_predictions.jsonl
"""
import os
import json
import re
import sys
import time
from pathlib import Path
from openai import OpenAI

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BLINDED_FILE = DATA_DIR / "blinded_test_notes.jsonl"
MAPPING_FILE = DATA_DIR / "blind_id_to_note_id.json"
OUTPUT_FILE = DATA_DIR / "llm_blind_predictions.jsonl"

API_BASE = "http://localhost:8091/v1"
API_KEY = "not-needed"
MODEL = "google/gemma-4-31B-it"

CATEGORIES = [
    "ADVERSE_EFFECTS_TOLERABILITY",
    "ACCESS_COST_SYSTEM_BARRIERS",
    "EFFICACY_EXPECTATION_MISMATCH",
    "PATIENT_PREFERENCE_BELIEFS",
    "PLANNED_GOAL_BASED_DISCONTINUATION",
    "CLINICAL_DECISION_MEDICAL_APPROPRIATENESS",
    "OTHER_UNKNOWN",
]

CATEGORY_DEFINITIONS_TEXT = """
- ADVERSE_EFFECTS_TOLERABILITY: GI side effects (nausea, vomiting, diarrhea, constipation), systemic symptoms, serious adverse events, injection site reactions, or other medication-related adverse effects.
- ACCESS_COST_SYSTEM_BARRIERS: Insurance coverage issues, high out-of-pocket costs, prior authorization problems, medication shortages, pharmacy access barriers, or system-level hurdles.
- EFFICACY_EXPECTATION_MISMATCH: Inadequate weight loss, insufficient glycemic control, weight plateau after initial loss, or perceived loss of effectiveness.
- PATIENT_PREFERENCE_BELIEFS: Patient concerns about long-term safety, dislike of injections, stigma, preference for natural remedies, misinformation, or personal choice.
- PLANNED_GOAL_BASED_DISCONTINUATION: Reached target weight, glycemic goal achieved, planned treatment duration completed, or deliberate elective tapering.
- CLINICAL_DECISION_MEDICAL_APPROPRIATENESS: Contraindications, drug interactions, pregnancy, declining renal function, frailty/sarcopenia, surgery preparation, or switching therapy.
- OTHER_UNKNOWN: Primary reason unclear, not documented, lost to follow-up, or does not fit other categories.
"""

SYSTEM_PROMPT = """You are a clinical NLP specialist. Your task is to analyze clinical notes and classify the primary reason for GLP-1 receptor agonist discontinuation. Read the note carefully and select the single most appropriate category. Respond only with the requested JSON format."""

USER_PROMPT_TEMPLATE = """Analyze the following clinical note and classify the primary discontinuation reason.

Categories:
{category_definitions}

Respond with a JSON object in exactly this format — no other text before or after:
{{
  "primary_reason": "<one of the seven labels exactly as listed above>",
  "confidence": <float between 0.0 and 1.0>,
  "supporting_evidence": "<short direct quote or paraphrase from the note>",
  "secondary_factors": ["<optional secondary factor, if any>"],
  "ambiguity_note": "<why this was difficult, or null if straightforward>"
}}

Clinical Note:
{note_text}"""


def load_blinded_notes():
    records = []
    with open(BLINDED_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_mapping():
    with open(MAPPING_FILE) as f:
        return json.load(f)


def load_existing_predictions():
    preds = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    preds[d["blind_id"]] = d
    return preds


def parse_llm_response(text):
    """Extract JSON from LLM response, handling markdown fences and extra text."""
    # Remove markdown code fences
    cleaned = re.sub(r'```(?:json)?\s*', '', text)
    # Find JSON object
    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            return result
        except json.JSONDecodeError:
            pass
    return None


def validate_prediction(pred):
    """Check prediction has required fields and valid category."""
    if not pred:
        return False
    if "primary_reason" not in pred:
        return False
    if pred["primary_reason"] not in CATEGORIES:
        return False
    return True


def main():
    client = OpenAI(base_url=API_BASE, api_key=API_KEY)
    records = load_blinded_notes()
    mapping = load_mapping()
    existing = load_existing_predictions()
    mapping_by_bid = mapping

    print(f"Loaded {len(records)} blinded notes")
    print(f"Existing predictions: {len(existing)}")
    print(f"Model: {MODEL}")
    print(f"API: {API_BASE}")

    # Filter to notes not yet predicted
    to_process = [r for r in records if r["blind_id"] not in existing]
    print(f"To process: {len(to_process)}")

    if not to_process:
        print("All notes already processed.")
        return

    stats = {"success": 0, "failed": 0, "parse_errors": 0, "total_tokens": 0}

    for idx, rec in enumerate(to_process):
        blind_id = rec["blind_id"]
        note_text = rec["note_text"]

        user_prompt = USER_PROMPT_TEMPLATE.format(
            category_definitions=CATEGORY_DEFINITIONS_TEXT.strip(),
            note_text=note_text,
        )

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=512,
            )

            raw_text = response.choices[0].message.content
            pred = parse_llm_response(raw_text)

            if pred and validate_prediction(pred):
                stats["success"] += 1
            else:
                stats["parse_errors"] += 1
                pred = {
                    "primary_reason": "PARSE_ERROR",
                    "confidence": 0.0,
                    "supporting_evidence": "",
                    "secondary_factors": [],
                    "ambiguity_note": f"Failed to parse: {raw_text[:200]}",
                }

            # Add metadata
            original_info = mapping_by_bid.get(blind_id, {})
            pred["blind_id"] = blind_id
            pred["original_note_id"] = original_info.get("original_note_id", "unknown")
            pred["intended_label"] = original_info.get("intended_label", "unknown")
            pred["raw_response"] = raw_text

            if hasattr(response, "usage") and response.usage:
                stats["total_tokens"] += response.usage.total_tokens

            # Append to output file incrementally
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(pred) + "\n")

        except Exception as e:
            stats["failed"] += 1
            err_pred = {
                "blind_id": blind_id,
                "primary_reason": "API_ERROR",
                "confidence": 0.0,
                "supporting_evidence": "",
                "secondary_factors": [],
                "ambiguity_note": f"API error: {str(e)}",
                "intended_label": mapping_by_bid.get(blind_id, {}).get("intended_label", "unknown"),
            }
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(err_pred) + "\n")
            print(f"  Error on {blind_id}: {e}")

        if (idx + 1) % 20 == 0:
            print(f"  Progress: {idx + 1}/{len(to_process)} "
                  f"(success={stats['success']}, parse_err={stats['parse_errors']}, "
                  f"fail={stats['failed']})")

        # Small delay to avoid hammering the endpoint
        time.sleep(0.1)

    print(f"\n{'='*60}")
    print(f"LLM Blind Judging Complete")
    print(f"{'='*60}")
    print(f"Total processed: {len(to_process)}")
    print(f"Successful: {stats['success']}")
    print(f"Parse errors: {stats['parse_errors']}")
    print(f"API errors: {stats['failed']}")
    print(f"Total tokens used: {stats['total_tokens']}")
    print(f"Output saved to: {OUTPUT_FILE}")

    # Reload and verify
    final_preds = load_existing_predictions()
    print(f"\nTotal predictions in output: {len(final_preds)}")


if __name__ == "__main__":
    main()
