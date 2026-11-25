"""
Step 1 – Lexical Diversity: TTR & Unique N-gram Ratios
======================================================
Computes corpus-level and per-category lexical diversity metrics:
  • Type-Token Ratio (TTR)
  • Unique bigram ratio
  • Unique trigram ratio
"""
import json
import pandas as pd
import numpy as np
from collections import Counter
from nltk.util import ngrams

from config import RESULTS_DIR, CATEGORY_ORDER
from data_loader import load_train, load_test, tokenize


def compute_ttr(tokens: list[str]) -> dict:
    """Compute Type-Token Ratio for a list of tokens."""
    total = len(tokens)
    unique = len(set(tokens))
    ttr = unique / total if total > 0 else 0.0
    return {"total_tokens": total, "unique_tokens": unique, "ttr": round(ttr, 6)}


def compute_ngram_ratio(tokens: list[str], n: int) -> dict:
    """Compute unique n-gram ratio."""
    all_ngrams = list(ngrams(tokens, n))
    total = len(all_ngrams)
    unique = len(set(all_ngrams))
    ratio = unique / total if total > 0 else 0.0
    return {
        f"total_{n}grams": total,
        f"unique_{n}grams": unique,
        f"unique_{n}gram_ratio": round(ratio, 6),
    }


def analyze_lexical_diversity(df: pd.DataFrame, split_name: str) -> dict:
    """Run full lexical diversity analysis on a DataFrame."""
    results = {}

    # ── Corpus-level ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Lexical Diversity – {split_name}")
    print(f"{'='*60}")

    all_tokens = []
    for text in df["clinical_note"]:
        all_tokens.extend(tokenize(text))

    corpus_ttr = compute_ttr(all_tokens)
    corpus_bi  = compute_ngram_ratio(all_tokens, 2)
    corpus_tri = compute_ngram_ratio(all_tokens, 3)

    corpus_result = {**corpus_ttr, **corpus_bi, **corpus_tri}
    results["corpus"] = corpus_result

    print(f"\n  Corpus-level ({len(df)} notes):")
    print(f"    Total tokens:        {corpus_ttr['total_tokens']:,}")
    print(f"    Unique tokens:       {corpus_ttr['unique_tokens']:,}")
    print(f"    TTR:                 {corpus_ttr['ttr']:.4f}")
    print(f"    Unique bigram ratio: {corpus_bi['unique_2gram_ratio']:.4f}")
    print(f"    Unique trigram ratio: {corpus_tri['unique_3gram_ratio']:.4f}")

    # ── Per-category ──────────────────────────────────────────────────────
    print(f"\n  Per-Category:")
    print(f"    {'Category':<22} {'TTR':>8} {'Bi-ratio':>10} {'Tri-ratio':>10} {'Tokens':>10}")
    print(f"    {'-'*62}")

    per_cat = {}
    for cat in CATEGORY_ORDER:
        cat_df = df[df["category"] == cat]
        if len(cat_df) == 0:
            continue
        cat_tokens = []
        for text in cat_df["clinical_note"]:
            cat_tokens.extend(tokenize(text))

        cat_ttr = compute_ttr(cat_tokens)
        cat_bi  = compute_ngram_ratio(cat_tokens, 2)
        cat_tri = compute_ngram_ratio(cat_tokens, 3)

        cat_result = {
            "n_notes": len(cat_df),
            **cat_ttr, **cat_bi, **cat_tri,
        }
        per_cat[cat] = cat_result

        print(f"    {cat:<22} {cat_ttr['ttr']:>8.4f} {cat_bi['unique_2gram_ratio']:>10.4f} {cat_tri['unique_3gram_ratio']:>10.4f} {cat_ttr['total_tokens']:>10,}")

    results["per_category"] = per_cat

    # ── Per-note stats (for distribution analysis) ────────────────────────
    note_ttrs = []
    for text in df["clinical_note"]:
        tokens = tokenize(text)
        if len(tokens) > 0:
            note_ttrs.append(len(set(tokens)) / len(tokens))

    note_stats = {
        "mean_per_note_ttr":   round(np.mean(note_ttrs), 6),
        "std_per_note_ttr":    round(np.std(note_ttrs), 6),
        "median_per_note_ttr": round(np.median(note_ttrs), 6),
        "min_per_note_ttr":    round(np.min(note_ttrs), 6),
        "max_per_note_ttr":    round(np.max(note_ttrs), 6),
    }
    results["per_note_ttr_stats"] = note_stats

    print(f"\n  Per-note TTR distribution:")
    print(f"    Mean:   {note_stats['mean_per_note_ttr']:.4f}")
    print(f"    Std:    {note_stats['std_per_note_ttr']:.4f}")
    print(f"    Median: {note_stats['median_per_note_ttr']:.4f}")
    print(f"    Range:  [{note_stats['min_per_note_ttr']:.4f}, {note_stats['max_per_note_ttr']:.4f}]")

    return results


def main():
    train_df = load_train()
    test_df  = load_test()

    all_results = {}

    # Train set analysis
    all_results["train"] = analyze_lexical_diversity(train_df, "Train Set")

    # Test set analysis
    all_results["test"] = analyze_lexical_diversity(test_df, "Test Set")

    # Combined analysis
    combined = pd.concat([train_df, test_df], ignore_index=True)
    all_results["combined"] = analyze_lexical_diversity(combined, "Combined (Train+Test)")

    # Save results
    out_path = f"{RESULTS_DIR}/step1_lexical_diversity.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
