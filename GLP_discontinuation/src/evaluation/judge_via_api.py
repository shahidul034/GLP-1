import json
import requests
import re
from pathlib import Path

VALID_CATEGORIES = [
    "ADVERSE_EFFECTS_OR_TOLERABILITY",
    "ACCESS_COST_SYSTEM_BARRIERS",
    "PATIENT_PREFERENCE_BELIEFS",
    "PLANNED_GOAL_BASED_DISCONTINUATION",
]

JUDGE_PROMPT_TEMPLATE = """\
You are an expert clinical informaticist reviewing synthetic VA-style clinical notes for quality. \
Your task is to evaluate the following synthetic clinical note on 5 dimensions using a 1-5 Likert scale.

The note was generated with the intended primary discontinuation reason category: **{PRIMARY_REASON}**

--- BEGIN CLINICAL NOTE ---
{CLINICAL_NOTE}
--- END CLINICAL NOTE ---

Evaluate the note on each dimension below. For each, provide a score (1-5) and a brief justification (1-2 sentences).

**Scoring rubric (1-5):**
- 1 = Very poor / completely fails
- 2 = Poor / major issues
- 3 = Acceptable / some issues
- 4 = Good / minor issues
- 5 = Excellent / no issues

**Dimensions:**

1. **clinical_realism**: Does the note read like authentic VA-style clinical documentation? Consider abbreviations, tone, section structure, plausible clinical details, and realistic imperfections.

2. **category_alignment**: Does the note's content clearly reflect the labeled primary reason category ({PRIMARY_REASON})? The reason may be implicit or require inference, but the dominant factor must match. Penalize if the note reflects a different primary category or mixes categories without a clear dominant one.

3. **clinical_coherence**: Is the note internally consistent? Are vitals, labs, medications, history, and plan logically compatible? Penalize contradictions, implausible values, or disjointed reasoning.

4. **note_variability**: Does the note avoid being templatic or formulaic? Consider variation in length, section ordering, tone, level of detail, and whether the GLP-1 issue is foregrounded vs. embedded. A note that feels like a unique patient encounter scores higher.

5. **glp1_relevance**: Is a GLP-1 medication (semaglutide, liraglutide, tirzepatide, dulaglutide, exenatide, etc.) appropriately included? Is the medication scenario (continued, held, tapered, discontinued) clinically reasonable?

Output EXACTLY one valid JSON object with this structure (no markdown fences, no extra text):
{{
  "clinical_realism": {{"score": <int 1-5>, "justification": "<string>"}},
  "category_alignment": {{"score": <int 1-5>, "justification": "<string>"}},
  "clinical_coherence": {{"score": <int 1-5>, "justification": "<string>"}},
  "note_variability": {{"score": <int 1-5>, "justification": "<string>"}},
  "glp1_relevance": {{"score": <int 1-5>, "justification": "<string>"}},
  "overall_accept": <boolean true if all scores >= 3 else false>
}}"""

DIMENSIONS = [
    "clinical_realism",
    "category_alignment",
    "clinical_coherence",
    "note_variability",
    "glp1_relevance",
]

API_URL = "http://localhost:8091/v1/chat/completions"


def load_synthetic_data(input_path: str):
    records = []
    with open(input_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_index"] = i
            records.append(record)
    return records


def format_gemma_chat(user_content: str):
    return (
        "<start_of_turn>user\n"
        f"{user_content}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def build_judge_prompt(record):
    return JUDGE_PROMPT_TEMPLATE.format(
        PRIMARY_REASON=record["primary_reason"],
        CLINICAL_NOTE=record["clinical_note"],
    )


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    brace_start = text.find("{")
    if brace_start == -1:
        return None

    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[brace_start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def validate_judgment(parsed):
    for dim in DIMENSIONS:
        if dim not in parsed:
            return False
        entry = parsed[dim]
        if not isinstance(entry, dict):
            return False
        if "score" not in entry or "justification" not in entry:
            return False
        if not isinstance(entry["score"], int) or entry["score"] not in range(1, 6):
            return False
    return True


def judge_record(record, api_url=API_URL):
    prompt = build_judge_prompt(record)
    formatted = format_gemma_chat(prompt)

    payload = {
        "model": "google/gemma-4-31B-it",
        "messages": [
            {"role": "user", "content": formatted}
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
    }

    response = requests.post(api_url, json=payload, timeout=120)
    response.raise_for_status()

    generated_text = response.json()["choices"][0]["message"]["content"]
    parsed = extract_json(generated_text)

    if parsed and validate_judgment(parsed):
        judgment = {
            "_index": record["_index"],
            "primary_reason": record["primary_reason"],
            "scores": {dim: parsed[dim]["score"] for dim in DIMENSIONS},
            "justifications": {dim: parsed[dim]["justification"] for dim in DIMENSIONS},
            "overall_accept": parsed.get("overall_accept", all(
                parsed[dim]["score"] >= 3 for dim in DIMENSIONS
            )),
        }
    else:
        judgment = {
            "_index": record["_index"],
            "primary_reason": record["primary_reason"],
            "scores": {dim: -1 for dim in DIMENSIONS},
            "justifications": {dim: "PARSE_FAILURE" for dim in DIMENSIONS},
            "overall_accept": False,
            "_parse_error": True,
            "_raw_output": generated_text[:500],
        }
        print(f"  Warning: failed to parse judgment for index {record['_index']}")

    return judgment


def main():
    input_path = "/home/user/GLP_project/GLP_project_ad/data/synthetic_data/gpt5.4_100.jsonl"
    output_path = "/home/user/GLP_project/GLP_project_ad/judge_results/gpt5.4/judgments.jsonl"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    records = load_synthetic_data(input_path)
    print(f"Loaded {len(records)} records")

    all_judgments = []
    for i, record in enumerate(records):
        print(f"Judging {i+1}/{len(records)}...")
        judgment = judge_record(record)
        all_judgments.append(judgment)

    with open(output_path, "w") as f:
        for j in all_judgments:
            f.write(json.dumps(j) + "\n")

    print(f"\nSaved judgments to {output_path}")

    valid = [j for j in all_judgments if not j.get("_parse_error")]
    accept_count = sum(1 for j in valid if j["overall_accept"])
    print(f"Valid judgments: {len(valid)}/{len(all_judgments)}")
    print(f"Accept rate: {accept_count}/{len(valid)} ({accept_count/len(valid)*100:.1f}%)")


if __name__ == "__main__":
    main()