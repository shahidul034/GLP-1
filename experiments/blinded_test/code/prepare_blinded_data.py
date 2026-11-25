#!/usr/bin/env python3
"""Prepare blinded test notes for label-blind annotation experiment.

Reads the 320-note test set, strips all label/metadata information,
assigns anonymous blind IDs, shuffles order, and writes:
  - data/blinded_test_notes.jsonl    (blind_id + note_text + category definitions)
  - data/blind_id_to_note_id.json    (mapping for later analysis)
"""
import json
import random
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
TEST_SET = Path("/home/mshahidul/GLP_project/GLP_project_ad/experiments/dataset/test.jsonl")
OUTPUT_NOTES = BASE / "data" / "blinded_test_notes.jsonl"
OUTPUT_MAPPING = BASE / "data" / "blind_id_to_note_id.json"
RANDOM_SEED = 42

CATEGORY_DEFINITIONS = {
    "ADVERSE_EFFECTS_TOLERABILITY": (
        "Discontinuation primarily due to gastrointestinal side effects "
        "(nausea, vomiting, diarrhea, constipation), systemic symptoms, "
        "serious adverse events, injection site reactions, or other medication-related adverse effects."
    ),
    "ACCESS_COST_SYSTEM_BARRIERS": (
        "Discontinuation primarily due to insurance coverage issues, "
        "high out-of-pocket costs, prior authorization problems, "
        "medication shortages, pharmacy access barriers, or system-level hurdles."
    ),
    "EFFICACY_EXPECTATION_MISMATCH": (
        "Discontinuation primarily because the medication is not working "
        "as expected — inadequate weight loss, insufficient glycemic control, "
        "weight plateau after initial loss, or perceived loss of effectiveness."
    ),
    "PATIENT_PREFERENCE_BELIEFS": (
        "Discontinuation primarily due to patient concerns or beliefs: "
        "fear of long-term side effects, dislike of injections, stigma, "
        "preference for natural remedies, misinformation, or personal choice."
    ),
    "PLANNED_GOAL_BASED_DISCONTINUATION": (
        "Discontinuation because the patient reached a predefined target: "
        "goal weight achieved, glycemic target met, planned treatment "
        "duration completed, or deliberate elective tapering."
    ),
    "CLINICAL_DECISION_MEDICAL_APPROPRIATENESS": (
        "Discontinuation due to a clinician's medical judgment: "
        "contraindications, drug interactions, pregnancy, declining renal "
        "function, frailty/sarcopenia, surgery preparation, or switching "
        "to an alternative therapy."
    ),
    "OTHER_UNKNOWN": (
        "The primary discontinuation reason is unclear, not documented, "
        "lost to follow-up, or does not fit any of the above categories."
    ),
}

CATEGORIES_SORTED = sorted(CATEGORY_DEFINITIONS.keys())

ANNOTATOR_INSTRUCTION = (
    "Read the following clinical note carefully and determine the PRIMARY reason "
    "for GLP-1 receptor agonist discontinuation. "
    "Select exactly one category from the definitions below. "
    "If the note describes multiple contributing factors, choose the dominant one. "
    "Provide brief supporting evidence from the text."
)


def main():
    random.seed(RANDOM_SEED)

    # Load original test set
    records = []
    with open(TEST_SET) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} test records")

    # Shuffle to remove any ordering hints
    random.shuffle(records)

    # Build blinded records and mapping
    blinded_records = []
    id_mapping = {}

    for idx, rec in enumerate(records, start=1):
        blind_id = f"B-{idx:03d}"
        original_note_id = rec.get("note_id", f"unknown_{idx}")

        id_mapping[blind_id] = {
            "original_note_id": original_note_id,
            "intended_label": rec["primary_reason"],
        }

        blinded_records.append({
            "blind_id": blind_id,
            "instruction": ANNOTATOR_INSTRUCTION,
            "categories": CATEGORIES_SORTED,
            "category_definitions": CATEGORY_DEFINITIONS,
            "note_text": rec["clinical_note"],
        })

    # Write blinded notes
    with open(OUTPUT_NOTES, "w") as f:
        for rec in blinded_records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(blinded_records)} blinded notes to {OUTPUT_NOTES}")

    # Write mapping
    with open(OUTPUT_MAPPING, "w") as f:
        json.dump(id_mapping, f, indent=2)
    print(f"Wrote mapping to {OUTPUT_MAPPING}")

    # Verify no label leakage
    label_leak_count = 0
    for rec in blinded_records:
        text_upper = rec["note_text"].upper()
        for cat in CATEGORIES_SORTED:
            label_words = cat.replace("_", " ")
            if label_words in text_upper:
                label_leak_count += 1
                break
    print(f"Notes with exact category label string in text: {label_leak_count}/{len(blinded_records)}")

    # Distribution check
    from collections import Counter
    dist = Counter(v["intended_label"] for v in id_mapping.values())
    print(f"Intended label distribution in blinded set:")
    for k, v in sorted(dist.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
