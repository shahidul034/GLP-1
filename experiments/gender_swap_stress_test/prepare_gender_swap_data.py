#!/usr/bin/env python3
"""
prepare_gender_swap_data.py

Convert simple schema JSONL files (clinical_note + primary_reason)
to ShareGPT format for training with finetune_gender_swap.py.

Uses the same instruction as the original paper model (UPPER_SNAKE_CASE labels).
"""

import json, os

BASE_DIR = "/home/mshahidul/GLP_project/GLP_project_ad/experiments/gender_swap_stress_test"
DATA_DIR = f"{BASE_DIR}/data"

INSTRUCTION = """You are a clinical NLP specialist. Analyze the following clinical note and classify the primary reason for GLP-1 receptor agonist discontinuation.

Categories:
- ADVERSE_EFFECTS_TOLERABILITY: GI side effects, systemic symptoms, serious adverse events, injection site reactions
- ACCESS_COST_SYSTEM_BARRIERS: Insurance issues, high costs, prior authorization, supply shortages
- EFFICACY_EXPECTATION_MISMATCH: Weight plateau, insufficient results, medication not working as expected
- PATIENT_PREFERENCE_BELIEFS: Safety fears, misinformation, stigma, preference for natural approaches
- PLANNED_GOAL_BASED_DISCONTINUATION: Reached goal weight, planned tapering, underlying condition improved
- CLINICAL_DECISION_MEDICAL_APPROPRIATENESS: Contraindications, drug interactions, switching therapy, pregnancy
- OTHER_UNKNOWN: Reason unclear, lost to follow-up, ambiguous documentation

Respond with a JSON object containing:
1. "primary_reason": The main discontinuation category
2. "confidence": Your confidence level (high/medium/low)
3. "supporting_evidence": Key phrases from the note supporting your classification
4. "secondary_factors": Any additional contributing factors (if applicable)"""


def to_sharegpt(clinical_note, label):
    return {
        "conversations": [
            {"from": "human", "value": INSTRUCTION + "\n\nClinical Note:\n" + clinical_note},
            {"from": "gpt", "value": json.dumps({
                "primary_reason": label,
                "confidence": "high",
                "supporting_evidence": "",
                "secondary_factors": []
            })}
        ]
    }


def convert_file(input_path, output_path):
    data = [json.loads(line) for line in open(input_path) if line.strip()]
    with open(output_path, 'w') as f:
        for item in data:
            sharegpt = to_sharegpt(item["clinical_note"], item["primary_reason"])
            f.write(json.dumps(sharegpt) + '\n')
    print(f"  Converted {len(data)} notes: {input_path} -> {output_path}")
    return len(data)


def main():
    print("Converting datasets to ShareGPT format...", flush=True)

    inputs = [
        ("train_gender_swapped.jsonl", "train_gender_swapped_sharegpt.jsonl"),
        ("test_gender_swapped.jsonl", "test_gender_swapped_sharegpt.jsonl"),
        ("train_gender_balanced_augmented.jsonl", "train_gender_balanced_augmented_sharegpt.jsonl"),
    ]

    for in_name, out_name in inputs:
        in_path = f"{DATA_DIR}/{in_name}"
        out_path = f"{DATA_DIR}/{out_name}"
        if os.path.exists(in_path):
            convert_file(in_path, out_path)
        else:
            print(f"  SKIP: {in_path} not found", flush=True)

    # Also convert original datasets for reference
    orig_train = "/home/mshahidul/GLP_project/GLP_project_ad/experiments/dataset/train.jsonl"
    orig_test = "/home/mshahidul/GLP_project/GLP_project_ad/experiments/dataset/test.jsonl"
    convert_file(orig_train, f"{DATA_DIR}/train_original_sharegpt.jsonl")
    convert_file(orig_test, f"{DATA_DIR}/test_original_sharegpt.jsonl")

    print("Done!", flush=True)


if __name__ == "__main__":
    main()
