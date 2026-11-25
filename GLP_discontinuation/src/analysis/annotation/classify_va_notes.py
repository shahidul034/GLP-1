"""
Classify VA clinical notes into GLP-1 discontinuation categories using google/gemma-4-31B-it.

Uses the same 7-category taxonomy from the synthetic note generation pipeline
(generate_synthetic_all_categories_v2_gpt54.py) to label real clinical notes.
"""

import json
import re
import sys
from pathlib import Path
from vllm import LLM, SamplingParams

CATEGORY_DEFINITIONS = {
    "ADVERSE_EFFECTS_TOLERABILITY": {
        "definition": "Discontinuation driven by intolerance or harm from the GLP-1 (nausea, vomiting, diarrhea, dizziness, excessive appetite suppression, injection-site reaction).",
        "subcategories": [
            "GI side effects (nausea, vomiting, diarrhea)",
            "systemic symptoms (fatigue, dizziness, headache)",
            "serious adverse events (pancreatitis, gallbladder issues)",
            "injection site reactions",
            "gastrointestinal discomfort and bloating",
            "loss of appetite beyond therapeutic intent",
        ],
    },
    "ACCESS_COST_SYSTEM_BARRIERS": {
        "definition": "Discontinuation driven by inability to obtain or pay for therapy (insurance denial, prior authorization failure, shortage, backorder, unaffordable copay, pharmacy access problem).",
        "subcategories": [
            "insurance loss or denial of coverage",
            "coverage differences between anti-obesity medication (AOM) and anti-diabetic medication (ADM) indications",
            "high out-of-pocket costs",
            "formulary restrictions or prior authorization requirements",
            "medication shortage or supply issues",
            "pharmacy access or availability problems",
            "medication unavailability or sold out",
        ],
    },
    "EFFICACY_EXPECTATION_MISMATCH": {
        "definition": "Discontinuation driven mainly by insufficient perceived or measured benefit (inadequate weight loss, glycemic under-response, plateau, waning effect, lack of expected progress).",
        "subcategories": [
            "weight plateau after initial success",
            "insufficient weight loss compared to expectations",
            "inadequate glycemic control despite medication",
            "'not working anymore' perception",
            "slower than expected results",
            "weight regain while still on medication",
        ],
    },
    "PATIENT_PREFERENCE_BELIEFS": {
        "definition": "Discontinuation driven primarily by the patient's values, beliefs, or unwillingness to continue (fear of side effects, injection fatigue, stigma, misinformation, preference for lifestyle-only management).",
        "subcategories": [
            "fear of long-term safety concerns (thyroid cancer, cardiovascular)",
            "misinformation from social media or online sources",
            "social stigma around weight loss medications",
            "desire to stop after reaching goal weight",
            "psychological discomfort with medication dependency",
            "preference for 'natural' or lifestyle-only approaches",
            "family or peer influence against medication use",
        ],
    },
    "PLANNED_GOAL_BASED_DISCONTINUATION": {
        "definition": "Discontinuation framed as an expected endpoint after goal achievement or planned completion (goal weight reached, short-term bridge therapy completed, planned taper, maintenance off medication, pregnancy planning as a planned stop).",
        "subcategories": [
            "used as a short-term tool to jumpstart weight loss",
            "reached target weight goal",
            "intentional tapering as planned with provider",
            "underlying condition improved (diabetes remission, A1c normalized)",
            "transition to maintenance phase without medication",
            "planned pregnancy or fertility considerations",
        ],
    },
    "CLINICAL_DECISION_MEDICAL_APPROPRIATENESS": {
        "definition": "Discontinuation driven by provider judgment about safety, appropriateness, or treatment strategy (contraindication, comorbidity, drug-drug interaction, switch to alternative therapy, guideline-based change, provider-advised stop).",
        "subcategories": [
            "contraindications discovered (CKD progression, pancreatitis history)",
            "drug-drug interactions requiring discontinuation",
            "new comorbidity making GLP-1 inappropriate",
            "switching to alternative therapy (bariatric surgery, different medication class)",
            "guideline-driven medication changes",
            "symptom or disease resolution making treatment unnecessary",
            "provider recommendation due to lack of benefit",
            "pregnancy",
        ],
    },
    "OTHER_UNKNOWN": {
        "definition": "No single dominant reason can be assigned from the note (reason not documented, vague discontinuation mention, conflicting explanations, lost to follow-up, weakly documented non-adherence without a stated cause).",
        "subcategories": [
            "reason not documented or unclear from note",
            "patient lost to follow-up",
            "medication simply not continued without explanation",
            "ambiguous or vague documentation",
            "multiple possible reasons without clear primary cause",
            "patient non-adherence without stated reason",
            "discontinuation noted in medication list only",
        ],
    },
}

CATEGORIES = list(CATEGORY_DEFINITIONS.keys())

SYSTEM_PROMPT = (
    "You are an expert clinical NLP system specialized in classifying the primary reason "
    "for GLP-1 receptor agonist (semaglutide/Ozempic/Wegovy, liraglutide/Victoza/Saxenda, "
    "dulaglutide/Trulicity, tirzepatide/Mounjaro/Zepbound, exenatide/Byetta/Bydureon) "
    "discontinuation from clinical notes. Your task is to read a clinical note and identify "
    "the SINGLE most dominant reason the patient stopped, held, or discontinued the GLP-1 medication. "
    "Respond ONLY with a valid JSON object containing 'primary_reason' and 'rationale' fields. "
    "No explanation outside the JSON."
)


def build_classification_prompt(clinical_note: str) -> str:
    categories_text = ""
    for cat, info in CATEGORY_DEFINITIONS.items():
        subcat_list = "\n".join(f"      - {sc}" for sc in info["subcategories"])
        categories_text += (
            f"  - {cat}\n"
            f"      Definition: {info['definition']}\n"
            f"      Common scenarios:\n{subcat_list}\n"
        )

    return (
        f"### Task\n"
        f"Classify the following clinical note into exactly ONE of the seven GLP-1 "
        f"discontinuation reason categories below. Choose the PRIMARY (most dominant) "
        f"reason for stopping/holding/discontinuing the GLP-1 therapy.\n\n"
        f"### Categories\n"
        f"{categories_text}\n"
        f"### Instructions\n"
        f"1. Read the clinical note carefully.\n"
        f"2. Identify whether the patient stopped, held, or discontinued a GLP-1 medication.\n"
        f"3. Determine the PRIMARY reason for that discontinuation.\n"
        f"4. If multiple reasons co-occur, choose the one most explicitly linked to the stop/hold/discontinuation decision.\n"
        f"5. Use OTHER_UNKNOWN only when no dominant reason is identifiable.\n\n"
        f"### Clinical Note\n"
        f"```text\n{clinical_note}\n```\n\n"
        f"### Output Format\n"
        f'Respond with a JSON object:\n'
        f'{{"primary_reason": "CATEGORY_NAME", "rationale": "Brief explanation of why this category was chosen"}}\n'
    )


def parse_response(text: str) -> dict | None:
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
                    parsed = json.loads(text[brace_start : i + 1])
                    return parsed
                except json.JSONDecodeError:
                    return None
    return None


def main():
    input_path = Path("/home/user/GLP_project/GLP_project_ad/data/va_data/clinical_notes (2).jsonl")
    output_path = Path("/home/user/GLP_project/GLP_project_ad/data/va_data/classified_va_notes.jsonl")
    model_id = "google/gemma-4-31B-it"

    print(f"Model:      {model_id}")
    print(f"Input:      {input_path}")
    print(f"Output:     {output_path}")
    print(f"Categories: {CATEGORIES}")
    print()

    notes = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                notes.append(json.loads(line))

    print(f"Loaded {len(notes)} notes")
    print()

    print("Loading model...")
    llm = LLM(
        model=model_id,
        max_model_len=8192,
        gpu_memory_utilization=0.92,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.95,
        max_tokens=256,
        seed=42,
    )

    print("Model loaded. Starting classification...")
    print()

    prompts = []
    for note in notes:
        prompts.append(
            llm.get_tokenizer().apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_classification_prompt(note["clinical_note"])},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    outputs = llm.generate(prompts, sampling_params)

    results = []
    category_counts = {cat: 0 for cat in CATEGORIES}
    parse_failures = 0

    for note, output in zip(notes, outputs):
        raw_text = output.outputs[0].text
        parsed = parse_response(raw_text)
        predicted = parsed.get("primary_reason", "PARSE_FAILED") if parsed else "PARSE_FAILED"
        rationale = parsed.get("rationale", "") if parsed else ""

        if predicted not in CATEGORIES:
            predicted = "PARSE_FAILED"
            parse_failures += 1
        else:
            category_counts[predicted] += 1

        results.append({
            "note_id": note.get("note_id", ""),
            "clinical_note": note["clinical_note"],
            "primary_reason": note.get("primary_reason", ""),
            "gemma_predicted_reason": predicted,
            "gemma_rationale": rationale,
        })

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print("=" * 60)
    print("Classification Complete")
    print("=" * 60)
    print(f"Total notes:          {len(notes)}")
    print(f"Parse failures:       {parse_failures}")
    print()
    print("Predicted category distribution:")
    for cat in CATEGORIES:
        print(f"  {cat}: {category_counts[cat]}")
    print()
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
