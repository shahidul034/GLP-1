#!/usr/bin/env python3
"""
step4_density.py — Clinical Concept Density (Entity Extraction)
===============================================================

Uses scispaCy's biomedical NER models to measure how densely medical
concepts appear in synthetic vs. real clinical notes.

Metrics computed
----------------
1. **Concept Density** — #entities / #words per note (mean ± std).
2. **Entity-Type Distribution** — Breakdown by entity label
   (e.g., CHEMICAL, DISEASE, etc.) to see if synthetic notes have
   a realistic mix of medical concepts.
3. **Unique Entity Ratio** — Unique entity surface forms / total entities,
   testing whether the generator reuses the same terms excessively.
4. **Per-Category Density** — Concept density broken down by
   discontinuation reason.

Usage
-----
    python step4_density.py \
        --synthetic ../train.jsonl \
        --mimic     ../cleaned_mimic_0_1000.json \
        --sample_size 1000 \
        --model en_core_sci_md

Dependencies
------------
    pip install scispacy spacy tqdm
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import spacy
except ImportError:
    sys.exit("ERROR: install spacy  →  pip install spacy scispacy")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):  # noqa: D103
        return it

from data_loader import prepare_data, ensure_output_dir, save_json


# ---------------------------------------------------------------------------
# NER analysis helpers
# ---------------------------------------------------------------------------

def load_nlp(model_name: str = "en_core_sci_md"):
    """Load a scispaCy model, with a helpful error message on failure."""
    try:
        nlp = spacy.load(model_name)
    except OSError:
        sys.exit(
            f"ERROR: Model '{model_name}' not found.  Install it with:\n"
            f"  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
            f"releases/v0.5.4/{model_name}-0.5.4.tar.gz"
        )
    return nlp


def analyse_notes(
    nlp,
    texts: list[str],
    label: str,
    batch_size: int = 50,
) -> dict:
    """Run NER over *texts* and return density statistics."""
    densities: list[float] = []
    entity_type_counts: Counter = Counter()
    all_entity_texts: list[str] = []
    total_words = 0

    print(f"\n  Processing {label} ({len(texts):,} notes) ...")

    # Use spacy.pipe for efficient batched processing
    for doc in tqdm(
        nlp.pipe(texts, batch_size=batch_size, disable=["parser"]),
        total=len(texts),
        desc=f"  {label} NER",
    ):
        # Count words (excluding punctuation and whitespace)
        words = [tok for tok in doc if not tok.is_punct and not tok.is_space]
        n_words = len(words)
        total_words += n_words

        n_ents = len(doc.ents)
        if n_words > 0:
            densities.append(n_ents / n_words)

        for ent in doc.ents:
            entity_type_counts[ent.label_] += 1
            all_entity_texts.append(ent.text.lower())

    mean_density = statistics.mean(densities) if densities else 0.0
    std_density = statistics.stdev(densities) if len(densities) > 1 else 0.0
    median_density = statistics.median(densities) if densities else 0.0

    total_entities = sum(entity_type_counts.values())
    unique_entities = len(set(all_entity_texts))
    unique_entity_ratio = unique_entities / total_entities if total_entities else 0.0

    # Top 20 most common entity surface forms
    entity_freq = Counter(all_entity_texts)
    top_entities = entity_freq.most_common(20)

    return {
        "label": label,
        "num_notes": len(texts),
        "total_words": total_words,
        "total_entities": total_entities,
        "unique_entities": unique_entities,
        "unique_entity_ratio": round(unique_entity_ratio, 6),
        "mean_density": round(mean_density, 6),
        "std_density": round(std_density, 6),
        "median_density": round(median_density, 6),
        "entity_type_distribution": {
            k: v for k, v in entity_type_counts.most_common()
        },
        "top_20_entities": [{"text": t, "count": c} for t, c in top_entities],
        "densities": [round(d, 6) for d in densities],  # for plotting
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_density(
    synthetic_path: str,
    mimic_path: str | None = None,
    sample_size: int = 1000,
    model_name: str = "en_core_sci_md",
    output_dir: str = "results",
    seed: int = 42,
) -> dict:
    out = ensure_output_dir(output_dir)

    synthetic_notes, mimic_notes, records = prepare_data(
        synthetic_path, mimic_path, sample_size=sample_size, seed=seed,
    )

    nlp = load_nlp(model_name)

    print("\n" + "=" * 60)
    print("  STEP 4 — Clinical Concept Density")
    print("=" * 60)

    syn_stats = analyse_notes(nlp, synthetic_notes, "Synthetic")

    mimic_stats = None
    if mimic_notes:
        mimic_stats = analyse_notes(nlp, mimic_notes, "MIMIC")

    # --- Per-category density (synthetic only) ---
    cat_map: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        cat_map[rec["primary_reason"]].append(rec["text"])

    per_category: dict[str, dict] = {}
    for cat, cat_texts in sorted(cat_map.items()):
        cat_stats = analyse_notes(nlp, cat_texts, cat)
        per_category[cat] = {
            "num_notes": cat_stats["num_notes"],
            "mean_density": cat_stats["mean_density"],
            "std_density": cat_stats["std_density"],
            "unique_entity_ratio": cat_stats["unique_entity_ratio"],
            "total_entities": cat_stats["total_entities"],
        }

    # --- Assemble output ---
    # Remove raw densities list from saved summary (keep it for plotting)
    syn_plot_densities = syn_stats.pop("densities", [])
    mimic_plot_densities = mimic_stats.pop("densities", []) if mimic_stats else []

    output = {
        "model": model_name,
        "synthetic": syn_stats,
        "per_category": per_category,
    }
    if mimic_stats:
        output["mimic"] = mimic_stats

    save_json(output, out / "step4_density_results.json")

    # --- Pretty-print ---
    def _print_stats(stats: dict) -> None:
        print(f"  {'Corpus':<20s}: {stats['label']}")
        print(f"  {'Notes':<20s}: {stats['num_notes']:,}")
        print(f"  {'Total Words':<20s}: {stats['total_words']:,}")
        print(f"  {'Total Entities':<20s}: {stats['total_entities']:,}")
        print(f"  {'Unique Entities':<20s}: {stats['unique_entities']:,}")
        print(f"  {'Unique Entity Ratio':<20s}: {stats['unique_entity_ratio']:.4f}")
        print(f"  {'Mean Density':<20s}: {stats['mean_density']:.4f} ± {stats['std_density']:.4f}")
        print(f"  {'Median Density':<20s}: {stats['median_density']:.4f}")
        print(f"\n  Entity-type distribution:")
        for etype, cnt in stats["entity_type_distribution"].items():
            print(f"    {etype:<25s}: {cnt:>7,}")
        print(f"\n  Top-10 entities:")
        for item in stats["top_20_entities"][:10]:
            print(f"    {item['text']:<30s}: {item['count']:>5,}")

    print("\n" + "-" * 60)
    _print_stats(syn_stats)
    if mimic_stats:
        print("\n" + "-" * 60)
        _print_stats(mimic_stats)

    print(f"\n  Per-category concept density (synthetic):")
    print(f"  {'Category':>45s}  {'Notes':>6s}  {'Density':>10s}  {'UER':>8s}")
    for cat, info in per_category.items():
        print(f"  {cat:>45s}  {info['num_notes']:>6d}  "
              f"{info['mean_density']:>10.4f}  {info['unique_entity_ratio']:>8.4f}")
    print("=" * 60)

    # --- Optional: density distribution histogram ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(syn_plot_densities, bins=40, alpha=0.7, label="Synthetic",
                color="#4C72B0", edgecolor="black")
        if mimic_plot_densities:
            ax.hist(mimic_plot_densities, bins=40, alpha=0.6, label="MIMIC",
                    color="#DD8452", edgecolor="black")
        ax.axvline(syn_stats["mean_density"], color="#4C72B0", linestyle="--",
                   linewidth=1.5)
        if mimic_stats:
            ax.axvline(mimic_stats["mean_density"], color="#DD8452", linestyle="--",
                       linewidth=1.5)
        ax.set_xlabel("Concept Density (entities / word)")
        ax.set_ylabel("Frequency")
        ax.set_title("Clinical Concept Density Distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "step4_density_histogram.pdf", dpi=150)
        plt.close(fig)
        print(f"Histogram saved → {out / 'step4_density_histogram.pdf'}")
    except ImportError:
        pass

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 4: Clinical Concept Density (Entity Extraction)",
    )
    parser.add_argument("--synthetic", type=str, default="../train.jsonl",
                        help="Path to synthetic JSONL dataset")
    parser.add_argument("--mimic", type=str, default="../cleaned_mimic_0_1000.json",
                        help="Path to MIMIC reference dataset (optional)")
    parser.add_argument("--sample_size", type=int, default=1000,
                        help="Max notes to sample from each corpus")
    parser.add_argument("--model", type=str, default="en_core_sci_md",
                        help="scispaCy model name")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    run_density(
        synthetic_path=args.synthetic,
        mimic_path=args.mimic,
        sample_size=args.sample_size,
        model_name=args.model,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
