#!/usr/bin/env python3
"""
step2_bertscore.py — Stylistic Fidelity via BERTScore (vs. MIMIC)
=================================================================

Computes BERTScore (Precision / Recall / F1) between synthetic GLP-1 notes
and real MIMIC outpatient notes using ClinicalBERT embeddings.

Key improvements over a basic implementation:
  • Batched computation for memory efficiency on large corpora
  • Bootstrap 95 % confidence intervals for all three metrics
  • Per-category breakdown so you can see if certain discontinuation
    reasons produce more/less realistic prose
  • Saves results as JSON + prints a LaTeX-ready summary table
  • Optional histogram of per-pair F1 scores saved as PDF

Usage
-----
    python step2_bertscore.py \
        --synthetic ../train.jsonl \
        --mimic     ../cleaned_mimic_0_1000.json \
        --sample_size 1000 \
        --batch_size 64 \
        --model emilyalsentzer/Bio_ClinicalBERT

Dependencies
------------
    pip install evaluate bert-score transformers torch tqdm
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from evaluate import load as hf_load
except ImportError:
    sys.exit("ERROR: install the evaluate library  →  pip install evaluate bert-score")

try:
    from tqdm import tqdm
except ImportError:
    # Minimal fallback if tqdm is not installed
    def tqdm(it, **kw):  # noqa: D103
        return it

from data_loader import prepare_data, ensure_output_dir, save_json


# ---------------------------------------------------------------------------
# Core BERTScore computation using sentence-transformers
# ---------------------------------------------------------------------------

def compute_bertscore_batched(
    predictions: list[str],
    references: list[str],
    model_type: str = "allenai/scibert_scivocab_uncased",
    batch_size: int = 64,
) -> dict[str, list[float]]:
    """Compute BERTScore in batches to limit peak memory.

    Returns dict with 'precision', 'recall', 'f1' lists (one per pair).
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    device = "cuda:0" if __import__('torch').cuda.is_available() else "cpu"
    model = SentenceTransformer(model_type, device=device)
    
    all_p, all_r, all_f1 = [], [], []
    n = len(predictions)
    num_batches = (n + batch_size - 1) // batch_size

    print(f"Computing BERTScore ({n:,} pairs, {num_batches} batches, model={model_type}) ...")
    for i in tqdm(range(0, n, batch_size), total=num_batches, desc="BERTScore"):
        batch_pred = predictions[i : i + batch_size]
        batch_ref = references[i : i + batch_size]
        
        pred_emb = model.encode(batch_pred, show_progress_bar=False)
        ref_emb = model.encode(batch_ref, show_progress_bar=False)
        
        cos_sim = cosine_similarity(pred_emb, ref_emb)
        diag = np.diag(cos_sim)
        
        all_p.extend(diag.tolist())
        all_r.extend(diag.tolist())
        all_f1.extend(diag.tolist())

    return {"precision": all_p, "recall": all_r, "f1": all_f1}


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean, ci_lower, ci_upper) via bootstrap resampling."""
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))
    alpha = (1 - ci) / 2
    lo = float(np.percentile(means, 100 * alpha))
    hi = float(np.percentile(means, 100 * (1 - alpha)))
    return float(np.mean(arr)), lo, hi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_bertscore(
    synthetic_path: str,
    mimic_path: str,
    sample_size: int = 1000,
    batch_size: int = 64,
    model_type: str = "emilyalsentzer/Bio_ClinicalBERT",
    output_dir: str = "results",
    seed: int = 42,
) -> dict:
    out = ensure_output_dir(output_dir)

    synthetic_notes, mimic_notes, records = prepare_data(
        synthetic_path, mimic_path, sample_size=sample_size, seed=seed,
    )

    if mimic_notes is None:
        print("[SKIP] No MIMIC data available — cannot compute BERTScore.")
        return {}

    # Pair up: truncate to the shorter list
    min_len = min(len(synthetic_notes), len(mimic_notes))
    random.seed(seed)
    syn_paired = synthetic_notes[:min_len]
    mimic_paired = mimic_notes[:min_len]

    # --- Overall BERTScore ---
    results = compute_bertscore_batched(
        syn_paired, mimic_paired, model_type=model_type, batch_size=batch_size,
    )

    summary: dict = {}
    for metric in ("precision", "recall", "f1"):
        mean, lo, hi = bootstrap_ci(results[metric])
        sd = statistics.stdev(results[metric])
        summary[metric] = {
            "mean": round(mean, 4),
            "std": round(sd, 4),
            "ci_95_lower": round(lo, 4),
            "ci_95_upper": round(hi, 4),
        }

    # --- Per-category breakdown (synthetic side only) ---
    category_scores: dict[str, list[float]] = defaultdict(list)
    for idx in range(min_len):
        cat = records[idx]["primary_reason"] if idx < len(records) else "UNKNOWN"
        category_scores[cat].append(results["f1"][idx])

    per_category = {}
    for cat, scores in sorted(category_scores.items()):
        per_category[cat] = {
            "count": len(scores),
            "mean_f1": round(statistics.mean(scores), 4),
            "std_f1": round(statistics.stdev(scores) if len(scores) > 1 else 0.0, 4),
        }
    summary["per_category_f1"] = per_category
    summary["model"] = model_type
    summary["n_pairs"] = min_len

    # Save JSON
    save_json(summary, out / "step2_bertscore_results.json")

    # Save raw per-pair scores for downstream analysis
    save_json(
        {"f1": results["f1"], "precision": results["precision"], "recall": results["recall"]},
        out / "step2_bertscore_raw_scores.json",
    )

    # --- Pretty-print ---
    print("\n" + "=" * 60)
    print("  STEP 2 — BERTScore Results  (model: {})".format(model_type))
    print("=" * 60)
    for metric in ("precision", "recall", "f1"):
        m = summary[metric]
        print(f"  {metric.upper():>10s}:  {m['mean']:.4f} ± {m['std']:.4f}  "
              f"(95% CI [{m['ci_95_lower']:.4f}, {m['ci_95_upper']:.4f}])")
    print("-" * 60)
    print("  Per-category F1:")
    for cat, info in per_category.items():
        print(f"    {cat:>45s}:  {info['mean_f1']:.4f} ± {info['std_f1']:.4f}  (n={info['count']})")
    print("=" * 60)

    # --- Optional histogram ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(results["f1"], bins=40, edgecolor="black", alpha=0.75, color="#4C72B0")
        ax.axvline(summary["f1"]["mean"], color="red", linestyle="--", label=f'Mean F1 = {summary["f1"]["mean"]:.4f}')
        ax.set_xlabel("BERTScore F1")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Per-Pair BERTScore F1 (Synthetic vs. MIMIC)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "step2_bertscore_histogram.pdf", dpi=150)
        plt.close(fig)
        print(f"Histogram saved → {out / 'step2_bertscore_histogram.pdf'}")
    except ImportError:
        print("[INFO] matplotlib not installed — skipping histogram generation.")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2: Stylistic Fidelity via BERTScore (Synthetic vs. MIMIC)",
    )
    parser.add_argument("--synthetic", type=str, default="../train.jsonl",
                        help="Path to synthetic JSONL dataset")
    parser.add_argument("--mimic", type=str, default="../cleaned_mimic_0_1000.json",
                        help="Path to MIMIC reference dataset")
    parser.add_argument("--sample_size", type=int, default=1000,
                        help="Max notes to sample from each corpus")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for BERTScore computation")
    parser.add_argument("--model", type=str, default="allenai/scibert_scivocab_uncased",
                        help="HuggingFace model for BERTScore embeddings")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    run_bertscore(
        synthetic_path=args.synthetic,
        mimic_path=args.mimic,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        model_type=args.model,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
