#!/usr/bin/env python3
"""
Sample 100 test notes from gpt5.4.jsonl for ablation study evaluation.
Stratified sampling to maintain category distribution.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

INPUT_PATH = "/home/user/GLP_project/GLP_project_ad/data/gpt5.4/synthetic_data_gpt54/gpt5.4.jsonl"
OUTPUT_DIR = Path("/home/user/GLP_project/GLP_project_ad/code/ablation_studies/datasets")
SAMPLE_SIZE = 200
SEED = 42

CATEGORY_MAP = {
    "ADVERSE_EFFECTS_OR_TOLERABILITY": "ADVERSE_EFFECTS_TOLERABILITY",
    "ACCESS_COST_SYSTEM_BARRIERS": "ACCESS_COST_SYSTEM_BARRIERS",
    "EFFICACY_EXPECTATION_MISMATCH": "EFFICACY_EXPECTATION_MISMATCH",
    "PATIENT_PREFERENCE_BELIEFS": "PATIENT_PREFERENCE_BELIEFS",
    "PLANNED_GOAL_BASED_DISCONTINUATION": "PLANNED_GOAL_BASED_DISCONTINUATION",
    "CLINICAL_DECISION_MEDICAL_APPROPRIATENESS": "CLINICAL_DECISION_MEDICAL_APPROPRIATENESS",
    "OTHER_UNKNOWN": "OTHER_UNKNOWN",
}


def load_records():
    records = []
    with open(INPUT_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def stratified_sample(records, n_samples, seed):
    by_category = defaultdict(list)
    for r in records:
        cat = r["primary_reason"]
        by_category[cat].append(r)

    sampled = []
    rng = np.random.default_rng(seed)

    target_per_cat = {}
    total = sum(len(v) for v in by_category.values())
    for cat, recs in sorted(by_category.items()):
        proportion = len(recs) / total
        n = max(1, int(round(proportion * n_samples)))
        target_per_cat[cat] = n

    shortfall = n_samples - sum(target_per_cat.values())
    if shortfall != 0:
        cats_with_headroom = [c for c, n in target_per_cat.items() if n < len(by_category[c])]
        if cats_with_headroom:
            rng.shuffle(cats_with_headroom)
            for cat in cats_with_headroom[:abs(shortfall)]:
                target_per_cat[cat] += 1 if shortfall > 0 else -1

    for cat, recs in sorted(by_category.items()):
        n = target_per_cat.get(cat, 1)
        indices = list(range(len(recs)))
        chosen_idx = rng.choice(indices, size=min(n, len(recs)), replace=False)
        for idx in chosen_idx:
            r = dict(recs[idx])
            r["primary_reason"] = CATEGORY_MAP.get(cat, cat)
            r["note_id"] = f"gpt54_{r.get('note_id', idx)}"
            sampled.append(r)

    rng.shuffle(sampled)
    return sampled[:n_samples]


def main():
    print("=" * 60)
    print("Sampling 200 test notes from gpt5.4.jsonl")
    print("=" * 60)

    records = load_records()
    print(f"Loaded {len(records)} records")

    by_cat = defaultdict(int)
    for r in records:
        by_cat[r["primary_reason"]] += 1
    print("\nOriginal distribution:")
    for cat, cnt in sorted(by_cat.items()):
        print(f"  {cat}: {cnt}")

    sampled = stratified_sample(records, SAMPLE_SIZE, SEED)
    print(f"\nSampled {len(sampled)} records")

    sampled_by_cat = defaultdict(int)
    for r in sampled:
        sampled_by_cat[r["primary_reason"]] += 1
    print("\nSampled distribution:")
    for cat, cnt in sorted(sampled_by_cat.items()):
        print(f"  {cat}: {cnt}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "test_gpt54_200.jsonl"

    with open(output_path, "w") as f:
        for r in sampled:
            clean = {
                "note_id": r["note_id"],
                "primary_reason": r["primary_reason"],
                "clinical_note": r["clinical_note"],
            }
            f.write(json.dumps(clean) + "\n")

    print(f"\nSaved to: {output_path}")

    print("\nSample records:")
    for r in sampled[:3]:
        print(f"  note_id={r['note_id']}, primary_reason={r['primary_reason']}")


if __name__ == "__main__":
    main()