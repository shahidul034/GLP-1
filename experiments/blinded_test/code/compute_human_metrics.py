#!/usr/bin/env python3
"""Compute human annotation metrics for the label-blind annotation experiment.

Loads the 3 blind annotator files + mapping to intended labels, then computes:
  - Pairwise agreement / Fleiss' kappa
  - Unanimous agreement count
  - Majority consensus labels
  - Consensus vs intended label comparison
  - Per-class precision/recall/F1
  - Confusion matrix
  - Adjudicated cases
  - Common disagreement patterns
"""
import json
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix,
    precision_recall_fscore_support, classification_report,
)
import pandas as pd
from nltk.metrics.agreement import AnnotationTask

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ANNOTATORS = ["Anu", "Parth", "Adiba"]
MAPPING_FILE = DATA_DIR / "blind_id_to_note_id.json"
BLINDED_NOTES = DATA_DIR / "blinded_test_notes.jsonl"

CATEGORIES = [
    "ADVERSE_EFFECTS_TOLERABILITY",
    "ACCESS_COST_SYSTEM_BARRIERS",
    "EFFICACY_EXPECTATION_MISMATCH",
    "PATIENT_PREFERENCE_BELIEFS",
    "PLANNED_GOAL_BASED_DISCONTINUATION",
    "CLINICAL_DECISION_MEDICAL_APPROPRIATENESS",
    "OTHER_UNKNOWN",
]


def interpret_kappa(k):
    if k < 0: return "Poor (worse than chance)"
    if k < 0.20: return "Slight"
    if k < 0.40: return "Fair"
    if k < 0.60: return "Moderate"
    if k < 0.80: return "Substantial"
    return "Almost Perfect"


def load_mapping():
    with open(MAPPING_FILE) as f:
        return json.load(f)


def load_blinded_notes():
    recs = {}
    with open(BLINDED_NOTES) as f:
        for line in f:
            d = json.loads(line)
            recs[d["blind_id"]] = d
    return recs


def load_annotator(name):
    path = DATA_DIR / f"annotator_{name}_blind.jsonl"
    anns = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    e = json.loads(line)
                    anns[e["blind_id"]] = e
    return anns


def main():
    mapping = load_mapping()
    blinded = load_blinded_notes()

    # Load all annotator data
    ann_data = {}
    for name in ANNOTATORS:
        ann_data[name] = load_annotator(name)
        print(f"Loaded {name}: {len(ann_data[name])} annotations")

    # Common blind IDs across all annotators
    common_ids = sorted(set.intersection(
        *[set(ann_data[name].keys()) for name in ANNOTATORS]
    ))
    # Only consider IDs present in the mapping
    common_ids = [bid for bid in common_ids if bid in mapping]

    print(f"Common annotations across all annotators: {len(common_ids)}")

    if len(common_ids) == 0:
        print("ERROR: No common annotations found. Run the annotation UI first.")
        return

    # Extract labels per annotator
    labels = {}
    for name in ANNOTATORS:
        labels[name] = [ann_data[name][bid]["selected_label"] for bid in common_ids]

    intended_labels = [mapping[bid]["intended_label"] for bid in common_ids]

    all_label_set = sorted(set(
        l for name in labels for l in labels[name]
    ) | set(intended_labels))

    # ========== Inter-Annotator Agreement ==========
    iaa_data = []
    for name in ANNOTATORS:
        for bid in common_ids:
            iaa_data.append((name, bid, ann_data[name][bid]["selected_label"]))

    task = AnnotationTask(data=iaa_data)
    fleiss_kappa = task.multi_kappa()

    pairwise_kappa = {}
    pairwise_agreement = {}
    for i in range(len(ANNOTATORS)):
        for j in range(i + 1, len(ANNOTATORS)):
            n1, n2 = ANNOTATORS[i], ANNOTATORS[j]
            k = cohen_kappa_score(labels[n1], labels[n2], labels=all_label_set)
            pairwise_kappa[f"{n1}-{n2}"] = float(k)
            agree = sum(1 for a, b in zip(labels[n1], labels[n2]) if a == b)
            pairwise_agreement[f"{n1}-{n2}"] = {
                "count": agree, "total": len(common_ids), "rate": agree / len(common_ids),
            }

    # Unanimous / disagreement
    unanimous_ids = []
    disagreement_items = []
    for idx, bid in enumerate(common_ids):
        vals = [ann_data[name][bid]["selected_label"] for name in ANNOTATORS]
        if len(set(vals)) == 1:
            unanimous_ids.append(bid)
        else:
            disagreement_items.append({
                "blind_id": bid,
                "intended_label": mapping[bid]["intended_label"],
                "annotator_labels": {name: ann_data[name][bid]["selected_label"] for name in ANNOTATORS},
            })

    # Majority consensus
    consensus_labels = []
    adjudicated_ids = []
    for idx, bid in enumerate(common_ids):
        votes = [ann_data[name][bid]["selected_label"] for name in ANNOTATORS]
        c = Counter(votes)
        max_count = max(c.values())
        winners = [k for k, v in c.items() if v == max_count]
        if len(winners) == 1:
            consensus_labels.append(winners[0])
        else:
            # 3-way tie — adjudicated
            consensus_labels.append(winners[0])
            adjudicated_ids.append(bid)

    # ========== Consensus vs Intended Labels ==========
    valid_idx = [i for i in range(len(consensus_labels)) if consensus_labels[i] in all_label_set]
    consensus_valid = [consensus_labels[i] for i in valid_idx]
    intended_valid = [intended_labels[i] for i in valid_idx]

    acc_consensus_vs_intended = accuracy_score(intended_valid, consensus_valid)
    kappa_consensus_vs_intended = float(cohen_kappa_score(
        intended_valid, consensus_valid, labels=all_label_set
    ))

    # Per-class metrics
    p, r, f1, support = precision_recall_fscore_support(
        intended_valid, consensus_valid, labels=CATEGORIES, zero_division=0,
    )
    per_class = {}
    for i, cat in enumerate(CATEGORIES):
        per_class[cat] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    macro_f1 = sum(f1[i] for i in range(len(CATEGORIES)) if support[i] > 0) / sum(1 for s in support if s > 0)
    weighted_f1 = sum(f1[i] * support[i] for i in range(len(CATEGORIES))) / sum(support)

    # Confusion matrix
    cm = confusion_matrix(intended_valid, consensus_valid, labels=CATEGORIES)
    cm_df = pd.DataFrame(cm, index=[f"Intended_{c}" for c in CATEGORIES],
                         columns=[f"Blind_{c}" for c in CATEGORIES])
    cm_csv = RESULTS_DIR / "human_label_independence_confusion_matrix.csv"
    cm_df.to_csv(cm_csv)

    # Full classification report
    cls_report = classification_report(intended_valid, consensus_valid, labels=CATEGORIES, zero_division=0)

    # ========== Confidence Analysis ==========
    confidence_by_annotator = {}
    for name in ANNOTATORS:
        confs = [ann_data[name][bid].get("confidence", 0.5) for bid in common_ids]
        confidence_by_annotator[name] = {
            "mean": float(np.mean(confs)),
            "median": float(np.median(confs)),
            "std": float(np.std(confs)),
        }

    # ========== Error Pattern Analysis ==========
    confusion_pairs = Counter()
    for i, bid in enumerate(common_ids):
        intended = mapping[bid]["intended_label"]
        if consensus_labels[i] != intended:
            confusion_pairs[(intended, consensus_labels[i])] += 1

    top_confusions = [
        {"intended": t, "predicted": p, "count": c}
        for (t, p), c in confusion_pairs.most_common(15)
    ]

    # Annotator vs annotator disagreements
    pair_confusions = {}
    for i in range(len(ANNOTATORS)):
        for j in range(i + 1, len(ANNOTATORS)):
            n1, n2 = ANNOTATORS[i], ANNOTATORS[j]
            pair_conf = Counter()
            for idx, bid in enumerate(common_ids):
                l1 = labels[n1][idx]
                l2 = labels[n2][idx]
                if l1 != l2:
                    pair_conf[(l1, l2)] += 1
            pair_confusions[f"{n1}_vs_{n2}"] = [
                {"annotator1_label": t, "annotator2_label": p, "count": c}
                for (t, p), c in pair_conf.most_common(10)
            ]

    # ========== Intended Label Distribution ==========
    intended_dist = Counter(intended_labels)
    consensus_dist = Counter(consensus_labels)

    # ========== Build Results ==========
    results = {
        "experiment": "blind_human_annotation",
        "num_notes": len(common_ids),
        "num_total_notes_in_set": len(mapping),
        "annotators": ANNOTATORS,
        "inter_annotator_agreement": {
            "fleiss_kappa": float(fleiss_kappa),
            "fleiss_kappa_interpretation": interpret_kappa(fleiss_kappa),
            "pairwise_cohens_kappa": pairwise_kappa,
            "pairwise_agreement": pairwise_agreement,
            "unanimous_agreement": {
                "count": len(unanimous_ids),
                "total": len(common_ids),
                "percentage": len(unanimous_ids) / len(common_ids) * 100,
            },
            "disagreement_summary": {
                "any_disagreement_count": len(disagreement_items),
                "percentage": len(disagreement_items) / len(common_ids) * 100,
                "two_vs_one": len(disagreement_items) - len(adjudicated_ids),
                "adjudicated_three_way": len(adjudicated_ids),
            },
        },
        "consensus_vs_intended": {
            "accuracy": float(acc_consensus_vs_intended),
            "cohens_kappa": float(kappa_consensus_vs_intended),
            "kappa_interpretation": interpret_kappa(kappa_consensus_vs_intended),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "per_class": per_class,
            "classification_report": cls_report,
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_csv": str(cm_csv),
        "top_confusions_vs_intended": top_confusions,
        "pairwise_annotator_confusions": pair_confusions,
        "confidence_by_annotator": confidence_by_annotator,
        "label_distributions": {
            "intended": dict(intended_dist),
            "blind_consensus": dict(consensus_dist),
        },
        "methodology_note": (
            "Annotators saw ONLY the clinical note text and category definitions. "
            "No intended label, model prediction, or category metadata was visible. "
            "This is a fully label-blind annotation."
        ),
    }

    out_path = RESULTS_DIR / "human_label_independence_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Blind Human Annotation Results")
    print(f"{'='*60}")
    print(f"Notes annotated: {len(common_ids)}/{len(mapping)}")
    print(f"Fleiss' kappa: {fleiss_kappa:.4f} ({interpret_kappa(fleiss_kappa)})")
    print(f"Unanimous agreement: {len(unanimous_ids)}/{len(common_ids)} "
          f"({len(unanimous_ids)/len(common_ids)*100:.1f}%)")
    print(f"Any disagreement: {len(disagreement_items)}/{len(common_ids)} "
          f"({len(disagreement_items)/len(common_ids)*100:.1f}%)")
    print(f"  Adjudicated (3-way): {len(adjudicated_ids)}")
    print(f"Consensus vs Intended: acc={acc_consensus_vs_intended:.4f}, "
          f"kappa={kappa_consensus_vs_intended:.4f}")
    print(f"Macro F1: {macro_f1:.4f}, Weighted F1: {weighted_f1:.4f}")
    print(f"\nPer-class F1:")
    for cat in CATEGORIES:
        if cat in per_class:
            pc = per_class[cat]
            print(f"  {cat}: P={pc['precision']:.4f} R={pc['recall']:.4f} "
                  f"F1={pc['f1']:.4f} (n={pc['support']})")
    print(f"\nTop confusions (consensus → intended):")
    for tc in top_confusions[:10]:
        print(f"  Intended={tc['intended']} → Consensus={tc['predicted']}: {tc['count']}")

    # Save consensus labels
    consensus_out = DATA_DIR / "human_consensus_labels.jsonl"
    with open(consensus_out, "w") as f:
        for idx, bid in enumerate(common_ids):
            f.write(json.dumps({
                "blind_id": bid,
                "original_note_id": mapping[bid]["original_note_id"],
                "intended_label": intended_labels[idx],
                "consensus_label": consensus_labels[idx],
                "annotator_labels": {name: ann_data[name][bid]["selected_label"] for name in ANNOTATORS},
                "unanimous": bid in unanimous_ids,
                "adjudicated": bid in adjudicated_ids,
            }) + "\n")
    print(f"\nConsensus labels saved to {consensus_out}")

    # Save combined raw annotations
    raw_out = DATA_DIR / "human_annotations_blind.jsonl"
    with open(raw_out, "w") as f:
        for bid in common_ids:
            for name in ANNOTATORS:
                rec = ann_data[name][bid].copy()
                rec["original_note_id"] = mapping[bid]["original_note_id"]
                rec["intended_label"] = mapping[bid]["intended_label"]
                f.write(json.dumps(rec) + "\n")
    print(f"Raw annotations saved to {raw_out}")


if __name__ == "__main__":
    import numpy as np
    main()
