#!/usr/bin/env python3
"""Re-analyze existing annotation data as a label-conditioned baseline.

Loads Anu.json, Parth.json, adiba.json from the existing annotator_evaluation/
directory and computes inter-annotator agreement metrics.

NOTE: This existing annotation is NOT fully blind — annotators saw the
assigned_label as a reference. Reported here as a baseline comparison.
"""
import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
from nltk.metrics.agreement import AnnotationTask

BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXISTING_DIR = Path("/home/mshahidul/GLP_project/GLP_project_ad/code/annotator_evaluation")

NORMALIZE_MAP = {"ADVERSE_EFFECTS_OR_TOLERABILITY": "ADVERSE_EFFECTS_TOLERABILITY"}

ANNOTATOR_FILES = {
    "Anu": EXISTING_DIR / "Anu.json",
    "Parth": EXISTING_DIR / "Parth.json",
    "Adiba": EXISTING_DIR / "adiba.json",
}


def normalize(label):
    return NORMALIZE_MAP.get(label, label)


def load_data(path):
    with open(path) as f:
        data = json.load(f)
    out = {}
    for ev in data["evaluations"].values():
        nid = ev["note_id"]
        out[nid] = {
            "assigned_label": normalize(ev["assigned_label"]),
            "human_corrected_label": normalize(ev["human_corrected_label"]),
            "human_agrees": ev["human_agrees"],
        }
    return out


def interpret_kappa(k):
    if k < 0: return "Poor (worse than chance)"
    if k < 0.20: return "Slight"
    if k < 0.40: return "Fair"
    if k < 0.60: return "Moderate"
    if k < 0.80: return "Substantial"
    return "Almost Perfect"


def main():
    annotators = {}
    for name, path in ANNOTATOR_FILES.items():
        annotators[name] = load_data(path)
        print(f"Loaded {name}: {len(annotators[name])} annotations")

    # Intersection of note IDs
    all_ids = sorted(set.intersection(*[set(a.keys()) for a in annotators.values()]))
    print(f"Common note IDs: {len(all_ids)}")

    # Extract human corrected labels
    labels = {}
    for name in annotators:
        labels[name] = [annotators[name][nid]["human_corrected_label"] for nid in all_ids]

    all_label_set = sorted(set(
        l for name in labels for l in labels[name]
    ))

    # === IAA ===
    iaa_data = []
    for name in annotators:
        for nid in all_ids:
            iaa_data.append((name, nid, annotators[name][nid]["human_corrected_label"]))

    task = AnnotationTask(data=iaa_data)
    fleiss_kappa = task.multi_kappa()

    # Pairwise Cohen's kappa
    names = list(annotators.keys())
    pairwise_kappa = {}
    pairwise_agreement = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            k = cohen_kappa_score(labels[n1], labels[n2], labels=all_label_set)
            pairwise_kappa[f"{n1}-{n2}"] = float(k)
            agree = sum(1 for a, b in zip(labels[n1], labels[n2]) if a == b)
            pairwise_agreement[f"{n1}-{n2}"] = {"count": agree, "total": len(all_ids), "rate": agree / len(all_ids)}

    # Unanimous / disagreement
    unanimous_count = 0
    disagreement_items = []
    for idx, nid in enumerate(all_ids):
        vals = [annotators[name][nid]["human_corrected_label"] for name in names]
        if len(set(vals)) == 1:
            unanimous_count += 1
        else:
            disagreement_items.append({
                "note_id": nid,
                "labels": {name: annotators[name][nid]["human_corrected_label"] for name in names},
                "assigned_label": annotators[names[0]][nid]["assigned_label"],
            })

    # Majority consensus
    consensus_labels = []
    for idx, nid in enumerate(all_ids):
        votes = [annotators[name][nid]["human_corrected_label"] for name in names]
        c = Counter(votes)
        max_count = max(c.values())
        winners = [k for k, v in c.items() if v == max_count]
        if len(winners) == 1:
            consensus_labels.append(winners[0])
        else:
            consensus_labels.append(labels[names[-1]][idx])

    # Compare consensus to assigned label (model/intended)
    assigned_labels = [annotators[names[0]][nid]["assigned_label"] for nid in all_ids]
    consensus_vs_assigned_acc = accuracy_score(assigned_labels, consensus_labels)
    consensus_vs_assigned_kappa = float(cohen_kappa_score(
        assigned_labels, consensus_labels, labels=all_label_set
    ))

    # Build results
    three_way = [d for d in disagreement_items if len(set(d["labels"].values())) == 3]
    two_one = [d for d in disagreement_items if len(set(d["labels"].values())) == 2]

    results = {
        "experiment": "existing_annotation_baseline",
        "note": "Annotators saw assigned_label before judging — NOT fully blind. Baseline only.",
        "num_notes": len(all_ids),
        "num_annotators": len(names),
        "annotators": names,
        "fleiss_kappa": float(fleiss_kappa),
        "fleiss_kappa_interpretation": interpret_kappa(fleiss_kappa),
        "pairwise_cohens_kappa": pairwise_kappa,
        "pairwise_agreement": pairwise_agreement,
        "unanimous_agreement": {
            "count": unanimous_count,
            "total": len(all_ids),
            "percentage": unanimous_count / len(all_ids) * 100,
        },
        "disagreement_summary": {
            "any_disagreement": len(disagreement_items),
            "percentage": len(disagreement_items) / len(all_ids) * 100,
            "two_vs_one": len(two_one),
            "three_way": len(three_way),
        },
        "consensus_vs_assigned_label": {
            "accuracy": float(consensus_vs_assigned_acc),
            "cohens_kappa": float(consensus_vs_assigned_kappa),
            "interpretation": interpret_kappa(consensus_vs_assigned_kappa),
        },
        "methodology_note": (
            "Annotators were shown the assigned_label (model prediction or intended label) "
            "and asked whether they agreed. The human_corrected_label field records their "
            "final judgment. This is a label-conditioned annotation, not a blind annotation."
        ),
    }

    out_path = RESULTS_DIR / "existing_annotation_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"\n{'='*60}")
    print(f"Existing Annotation Baseline Results")
    print(f"{'='*60}")
    print(f"Fleiss' kappa: {fleiss_kappa:.4f} ({interpret_kappa(fleiss_kappa)})")
    print(f"Unanimous agreement: {unanimous_count}/{len(all_ids)} ({unanimous_count/len(all_ids)*100:.1f}%)")
    print(f"Any disagreement: {len(disagreement_items)}/{len(all_ids)}")
    print(f"  2-vs-1: {len(two_one)}, 3-way: {len(three_way)}")
    print(f"Consensus vs assigned label: acc={consensus_vs_assigned_acc:.4f}, kappa={consensus_vs_assigned_kappa:.4f}")
    for pair, k in pairwise_kappa.items():
        ar = pairwise_agreement[pair]["rate"]
        print(f"  {pair}: Cohen's kappa={k:.4f}, agreement rate={ar:.4f}")


if __name__ == "__main__":
    main()
