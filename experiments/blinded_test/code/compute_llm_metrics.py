#!/usr/bin/env python3
"""Compute LLM blind judging metrics.

Compares Gemma blind predictions against:
  1. Intended labels from test.jsonl
  2. Human majority consensus (if available)

Reports agreement, per-class metrics, confusion matrices, low-confidence cases.
"""
import json
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix,
    precision_recall_fscore_support, classification_report,
)
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_FILE = DATA_DIR / "llm_blind_predictions.jsonl"
CONSENSUS_FILE = DATA_DIR / "human_consensus_labels.jsonl"
MAPPING_FILE = DATA_DIR / "blind_id_to_note_id.json"

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


def load_predictions():
    preds = {}
    if not PREDICTIONS_FILE.exists():
        print(f"ERROR: {PREDICTIONS_FILE} not found. Run run_llm_blind_judging.py first.")
        return preds
    with open(PREDICTIONS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                preds[d["blind_id"]] = d
    return preds


def load_consensus():
    consensus = {}
    if not CONSENSUS_FILE.exists():
        return consensus
    with open(CONSENSUS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                consensus[d["blind_id"]] = d
    return consensus


def load_mapping():
    with open(MAPPING_FILE) as f:
        return json.load(f)


def compute_metrics(y_true, y_pred, name):
    """Compute full metrics comparing predictions to reference."""
    valid = [(t, p) for t, p in zip(y_true, y_pred)
             if t in CATEGORIES and p in CATEGORIES]

    if not valid:
        print(f"  No valid comparisons for {name}")
        return None

    yt, yp = zip(*valid)
    acc = accuracy_score(yt, yp)
    kappa = float(cohen_kappa_score(yt, yp, labels=CATEGORIES))

    p, r, f1, support = precision_recall_fscore_support(
        yt, yp, labels=CATEGORIES, zero_division=0,
    )
    per_class = {}
    for i, cat in enumerate(CATEGORIES):
        per_class[cat] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    macro_f1 = sum(f1[i] for i in range(len(CATEGORIES)) if support[i] > 0) / \
               max(sum(1 for s in support if s > 0), 1)
    weighted_f1 = sum(f1[i] * support[i] for i in range(len(CATEGORIES))) / \
                  max(sum(support), 1)

    cm = confusion_matrix(yt, yp, labels=CATEGORIES)

    cls_report = classification_report(yt, yp, labels=CATEGORIES, zero_division=0)

    # Confusion pairs
    confusion_pairs = Counter()
    for t, p in zip(yt, yp):
        if t != p:
            confusion_pairs[(t, p)] += 1
    top_confusions = [
        {"true": t, "predicted": p, "count": c}
        for (t, p), c in confusion_pairs.most_common(15)
    ]

    return {
        "comparison": name,
        "valid_samples": len(valid),
        "total_samples": len(y_true),
        "accuracy": float(acc),
        "cohens_kappa": kappa,
        "kappa_interpretation": interpret_kappa(kappa),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "top_confusions": top_confusions,
        "classification_report": cls_report,
    }


def main():
    predictions = load_predictions()
    if not predictions:
        return
    print(f"Loaded {len(predictions)} LLM predictions")

    mapping = load_mapping()
    consensus = load_consensus()
    print(f"Loaded {len(consensus)} human consensus labels")
    print(f"Loaded {len(mapping)} mapping entries")

    # Common blind IDs
    common_with_intended = [bid for bid in predictions if bid in mapping]
    common_with_consensus = [bid for bid in predictions if bid in consensus]

    print(f"Predictions with intended labels: {len(common_with_intended)}")
    print(f"Predictions with human consensus: {len(common_with_consensus)}")

    results = {
        "experiment": "llm_blind_judging",
        "model": "google/gemma-4-31B-it",
        "api_endpoint": "http://localhost:8091/v1",
        "methodology_note": (
            "LLM received only: task description, category definitions, and clinical note text. "
            "No intended label, model prediction, or category metadata was provided. "
            "This is a fully label-blind re-judging experiment."
        ),
    }

    # 1. LLM vs Intended Labels
    if common_with_intended:
        yt = [mapping[bid]["intended_label"] for bid in common_with_intended]
        yp = [predictions[bid]["primary_reason"] for bid in common_with_intended]
        metrics_intended = compute_metrics(yt, yp, "llm_vs_intended")
        if metrics_intended:
            results["llm_vs_intended_labels"] = metrics_intended

            # Save confusion matrix
            cm_intended = pd.DataFrame(
                metrics_intended["confusion_matrix"],
                index=[f"Intended_{c}" for c in CATEGORIES],
                columns=[f"LLM_{c}" for c in CATEGORIES],
            )
            cm_intended.to_csv(RESULTS_DIR / "llm_vs_intended_confusion_matrix.csv")

    # 2. LLM vs Human Consensus
    if common_with_consensus:
        yt = [consensus[bid]["consensus_label"] for bid in common_with_consensus]
        yp = [predictions[bid]["primary_reason"] for bid in common_with_consensus]
        metrics_consensus = compute_metrics(yt, yp, "llm_vs_human_consensus")
        if metrics_consensus:
            results["llm_vs_human_consensus"] = metrics_consensus

            # Save confusion matrix
            cm_consensus = pd.DataFrame(
                metrics_consensus["confusion_matrix"],
                index=[f"Consensus_{c}" for c in CATEGORIES],
                columns=[f"LLM_{c}" for c in CATEGORIES],
            )
            cm_consensus.to_csv(RESULTS_DIR / "llm_vs_consensus_confusion_matrix.csv")

    # 3. LLM + Humans Agree Against Intended Label
    if common_with_intended and common_with_consensus:
        joint_common = [bid for bid in predictions
                        if bid in mapping and bid in consensus]
        llm_human_agree_against_intended = []
        for bid in joint_common:
            intended = mapping[bid]["intended_label"]
            human = consensus[bid]["consensus_label"]
            llm = predictions[bid]["primary_reason"]
            if human == llm and human != intended:
                llm_human_agree_against_intended.append({
                    "blind_id": bid,
                    "original_note_id": mapping[bid].get("original_note_id", "unknown"),
                    "intended_label": intended,
                    "human_consensus": human,
                    "llm_prediction": llm,
                    "llm_confidence": predictions[bid].get("confidence", 0.0),
                    "llm_ambiguity_note": predictions[bid].get("ambiguity_note", ""),
                })
        results["llm_human_agree_against_intended"] = {
            "count": len(llm_human_agree_against_intended),
            "total_joint": len(joint_common),
            "percentage": len(llm_human_agree_against_intended) / max(len(joint_common), 1) * 100,
            "cases": llm_human_agree_against_intended,
        }

    # 4. Low-confidence predictions
    low_conf = []
    for bid, pred in predictions.items():
        conf = pred.get("confidence", 0.5)
        if isinstance(conf, (int, float)) and conf < 0.6:
            low_conf.append({
                "blind_id": bid,
                "confidence": float(conf),
                "primary_reason": pred.get("primary_reason", "unknown"),
                "intended_label": mapping.get(bid, {}).get("intended_label", "unknown"),
                "ambiguity_note": pred.get("ambiguity_note", ""),
            })
    low_conf.sort(key=lambda x: x["confidence"])
    results["low_confidence_predictions"] = {
        "count": len(low_conf),
        "total": len(predictions),
        "percentage": len(low_conf) / max(len(predictions), 1) * 100,
        "threshold": 0.6,
        "cases": low_conf[:30],  # Top 30 lowest
    }

    # 5. Parse / API errors
    parse_errors = [
        {"blind_id": bid, "error": pred.get("ambiguity_note", "")}
        for bid, pred in predictions.items()
        if pred.get("primary_reason") in ("PARSE_ERROR", "API_ERROR")
    ]
    results["parse_and_api_errors"] = {
        "count": len(parse_errors),
        "total": len(predictions),
        "cases": parse_errors[:20],
    }

    # 6. Average confidence by category
    conf_by_cat = defaultdict(list)
    for bid, pred in predictions.items():
        pr = pred.get("primary_reason", "unknown")
        conf = pred.get("confidence", 0.5)
        if isinstance(conf, (int, float)) and pr in CATEGORIES:
            conf_by_cat[pr].append(conf)
    results["average_confidence_by_category"] = {
        cat: {
            "mean": float(sum(confs) / max(len(confs), 1)),
            "count": len(confs),
        }
        for cat, confs in sorted(conf_by_cat.items())
    }

    # Save results
    out_path = RESULTS_DIR / "llm_blind_judge_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"LLM Blind Judging Results")
    print(f"{'='*60}")
    if "llm_vs_intended_labels" in results:
        m = results["llm_vs_intended_labels"]
        print(f"LLM vs Intended Labels:")
        print(f"  Accuracy: {m['accuracy']:.4f}")
        print(f"  Cohen's kappa: {m['cohens_kappa']:.4f} ({m['kappa_interpretation']})")
        print(f"  Macro F1: {m['macro_f1']:.4f}, Weighted F1: {m['weighted_f1']:.4f}")

    if "llm_vs_human_consensus" in results:
        m = results["llm_vs_human_consensus"]
        print(f"\nLLM vs Human Consensus:")
        print(f"  Accuracy: {m['accuracy']:.4f}")
        print(f"  Cohen's kappa: {m['cohens_kappa']:.4f} ({m['kappa_interpretation']})")
        print(f"  Macro F1: {m['macro_f1']:.4f}, Weighted F1: {m['weighted_f1']:.4f}")

    if "llm_human_agree_against_intended" in results:
        m = results["llm_human_agree_against_intended"]
        print(f"\nLLM + Humans agree against intended:")
        print(f"  {m['count']}/{m['total_joint']} ({m['percentage']:.1f}%)")

    if "low_confidence_predictions" in results:
        m = results["low_confidence_predictions"]
        print(f"\nLow-confidence predictions (<0.6): {m['count']}/{m['total']} ({m['percentage']:.1f}%)")

    if "parse_and_api_errors" in results:
        m = results["parse_and_api_errors"]
        print(f"Parse/API errors: {m['count']}/{m['total']}")


if __name__ == "__main__":
    main()
