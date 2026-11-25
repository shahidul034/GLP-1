#!/usr/bin/env python3
"""
step3_diversity.py — Lexical Diversity (Mode Collapse Check)
============================================================

Measures vocabulary richness and phrasal diversity of the synthetic corpus
to verify the generator did **not** suffer from mode collapse.

Metrics computed
----------------
1. **Type-Token Ratio (TTR)**  — Unique words / total words.
2. **Moving-Average TTR (MSTTR)** — Average TTR over fixed-length windows
   (default 1 000 tokens).  Much more stable than corpus-level TTR,
   which shrinks mechanically as the corpus grows.
3. **Unique N-Gram Ratios** — For n ∈ {2, 3, 4}.
4. **Hapax Legomena Ratio** — Fraction of words that appear only once.
5. **Self-BLEU** (optional) — Average BLEU of each note against the rest;
   low self-BLEU = high diversity.  Expensive for large corpora, so we
   sample a subset.

All metrics are computed for the synthetic corpus.  If MIMIC data is
provided, the same metrics are computed for MIMIC as a reference.

Usage
-----
    python step3_diversity.py \
        --synthetic ../train.jsonl \
        --mimic     ../cleaned_mimic_0_1000.json \
        --sample_size 0         # 0 = use all notes (no subsampling)

Dependencies
------------
    pip install nltk tqdm
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from nltk.util import ngrams as nltk_ngrams
except ImportError:
    sys.exit("ERROR: install nltk  →  pip install nltk")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):  # noqa: D103
        return it

from data_loader import prepare_data, ensure_output_dir, save_json


# ---------------------------------------------------------------------------
# Tokeniser — lightweight, no heavy NLP models required
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase whitespace-and-punctuation tokeniser."""
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def type_token_ratio(tokens: list[str]) -> float:
    """Corpus-level TTR."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def moving_average_ttr(tokens: list[str], window: int = 1000) -> float:
    """MSTTR: average TTR over non-overlapping windows of *window* tokens.

    Much more comparable across corpora of different sizes than plain TTR.
    """
    if len(tokens) < window:
        return type_token_ratio(tokens)
    ttrs = []
    for i in range(0, len(tokens) - window + 1, window):
        chunk = tokens[i : i + window]
        ttrs.append(len(set(chunk)) / len(chunk))
    return statistics.mean(ttrs)


def unique_ngram_ratio(tokens: list[str], n: int = 3) -> float:
    """Unique n-gram count / total n-gram count."""
    grams = list(nltk_ngrams(tokens, n))
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def hapax_legomena_ratio(tokens: list[str]) -> float:
    """Fraction of vocabulary items that occur exactly once."""
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    hapax = sum(1 for c in freq.values() if c == 1)
    return hapax / len(freq)


def per_note_ttr(texts: list[str]) -> list[float]:
    """TTR per individual note — useful for distributions / box-plots."""
    return [type_token_ratio(tokenize(t)) for t in texts]


# ---------------------------------------------------------------------------
# Self-BLEU (optional, expensive)
# ---------------------------------------------------------------------------

def self_bleu(texts: list[str], sample_n: int = 200, seed: int = 42) -> float:
    """Average sentence-level BLEU of each sampled note against the rest.

    Lower = more diverse.
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError:
        print("[WARN] nltk.translate.bleu_score not available — skipping self-BLEU.")
        return float("nan")

    import random as _random
    _random.seed(seed)
    if len(texts) > sample_n:
        indices = sorted(_random.sample(range(len(texts)), sample_n))
    else:
        indices = list(range(len(texts)))

    tokenised = [tokenize(t) for t in texts]
    smooth = SmoothingFunction().method1
    scores = []

    for idx in tqdm(indices, desc="Self-BLEU"):
        hyp = tokenised[idx]
        refs = [tokenised[j] for j in indices if j != idx]
        sc = sentence_bleu(refs, hyp, smoothing_function=smooth)
        scores.append(sc)

    return statistics.mean(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Run all diversity metrics on a single corpus
# ---------------------------------------------------------------------------

def analyse_corpus(
    texts: list[str],
    label: str,
    compute_self_bleu: bool = False,
) -> dict:
    """Compute all diversity metrics for *texts*."""
    print(f"\n  Analysing {label} ({len(texts):,} notes) ...")

    all_tokens: list[str] = []
    for t in texts:
        all_tokens.extend(tokenize(t))

    total_tokens = len(all_tokens)
    unique_tokens = len(set(all_tokens))

    ttr = type_token_ratio(all_tokens)
    msttr = moving_average_ttr(all_tokens, window=1000)
    hapax = hapax_legomena_ratio(all_tokens)

    ngram_ratios = {}
    for n in (2, 3, 4):
        ngram_ratios[f"unique_{n}gram_ratio"] = round(unique_ngram_ratio(all_tokens, n), 6)

    note_ttrs = per_note_ttr(texts)
    note_ttr_mean = statistics.mean(note_ttrs) if note_ttrs else 0.0
    note_ttr_std = statistics.stdev(note_ttrs) if len(note_ttrs) > 1 else 0.0

    result = {
        "label": label,
        "num_notes": len(texts),
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "corpus_ttr": round(ttr, 6),
        "msttr_w1000": round(msttr, 6),
        "hapax_legomena_ratio": round(hapax, 6),
        **ngram_ratios,
        "per_note_ttr_mean": round(note_ttr_mean, 6),
        "per_note_ttr_std": round(note_ttr_std, 6),
    }

    if compute_self_bleu:
        sb = self_bleu(texts)
        result["self_bleu"] = round(sb, 6)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_diversity(
    synthetic_path: str,
    mimic_path: str | None = None,
    sample_size: int = 0,
    output_dir: str = "results",
    compute_self_bleu: bool = False,
    seed: int = 42,
) -> dict:
    out = ensure_output_dir(output_dir)

    # sample_size=0 means "use everything"
    effective_sample = sample_size if sample_size > 0 else 999_999

    synthetic_notes, mimic_notes, records = prepare_data(
        synthetic_path,
        mimic_path,
        sample_size=effective_sample,
        seed=seed,
    )

    print("\n" + "=" * 60)
    print("  STEP 3 — Lexical Diversity (Mode Collapse Check)")
    print("=" * 60)

    syn_stats = analyse_corpus(synthetic_notes, "Synthetic", compute_self_bleu)

    mimic_stats = None
    if mimic_notes:
        mimic_stats = analyse_corpus(mimic_notes, "MIMIC", compute_self_bleu)

    # --- Per-category diversity (synthetic only) ---
    cat_map: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        cat_map[rec["primary_reason"]].append(rec["text"])

    per_category = {}
    for cat, cat_texts in sorted(cat_map.items()):
        cat_tokens = []
        for t in cat_texts:
            cat_tokens.extend(tokenize(t))
        per_category[cat] = {
            "num_notes": len(cat_texts),
            "total_tokens": len(cat_tokens),
            "corpus_ttr": round(type_token_ratio(cat_tokens), 6),
            "msttr_w1000": round(moving_average_ttr(cat_tokens, 1000), 6),
            "unique_3gram_ratio": round(unique_ngram_ratio(cat_tokens, 3), 6),
        }

    # Assemble output
    output = {
        "synthetic": syn_stats,
        "per_category": per_category,
    }
    if mimic_stats:
        output["mimic"] = mimic_stats

    save_json(output, out / "step3_diversity_results.json")

    # --- Pretty-print ---
    def _print_row(label: str, stats: dict) -> None:
        print(f"\n  {'Metric':<30s}  {label}")
        print(f"  {'-'*30}  {'-'*15}")
        for k, v in stats.items():
            if k in ("label", "num_notes"):
                continue
            print(f"  {k:<30s}  {v}")

    _print_row("Synthetic", syn_stats)
    if mimic_stats:
        _print_row("MIMIC", mimic_stats)

    print(f"\n  Per-category diversity (synthetic):")
    print(f"  {'Category':>45s}  {'Notes':>6s}  {'TTR':>8s}  {'MSTTR':>8s}  {'3-gram':>8s}")
    for cat, info in per_category.items():
        print(f"  {cat:>45s}  {info['num_notes']:>6d}  {info['corpus_ttr']:>8.4f}  "
              f"{info['msttr_w1000']:>8.4f}  {info['unique_3gram_ratio']:>8.4f}")
    print("=" * 60)

    # --- Optional: box-plot of per-note TTR ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        data_to_plot = [per_note_ttr(synthetic_notes)]
        labels = ["Synthetic"]
        if mimic_notes:
            data_to_plot.append(per_note_ttr(mimic_notes))
            labels.append("MIMIC")

        fig, ax = plt.subplots(figsize=(6, 4))
        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True,
                        boxprops=dict(facecolor="#4C72B0", alpha=0.7))
        if len(data_to_plot) > 1:
            bp["boxes"][1].set_facecolor("#DD8452")
        ax.set_ylabel("Type-Token Ratio (per note)")
        ax.set_title("Per-Note TTR Distribution")
        fig.tight_layout()
        fig.savefig(out / "step3_diversity_boxplot.pdf", dpi=150)
        plt.close(fig)
        print(f"Box-plot saved → {out / 'step3_diversity_boxplot.pdf'}")
    except ImportError:
        pass

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3: Lexical Diversity / Mode Collapse Check",
    )
    parser.add_argument("--synthetic", type=str, default="../train.jsonl",
                        help="Path to synthetic JSONL dataset")
    parser.add_argument("--mimic", type=str, default="../cleaned_mimic_0_1000.json",
                        help="Path to MIMIC reference dataset (optional)")
    parser.add_argument("--sample_size", type=int, default=0,
                        help="Max notes to sample (0 = all)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--self_bleu", action="store_true",
                        help="Also compute self-BLEU (slow)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    run_diversity(
        synthetic_path=args.synthetic,
        mimic_path=args.mimic,
        sample_size=args.sample_size,
        output_dir=args.output_dir,
        compute_self_bleu=args.self_bleu,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
