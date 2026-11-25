"""
Embedding-based diversity analysis and filtering of synthetic clinical notes.

Uses Qwen/Qwen3-Embedding-8B to compute embeddings, measures pairwise
cosine similarity within each category, and filters near-duplicates
using a greedy max-similarity threshold to produce a diverse subset.
"""

import json
import argparse
import time
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

from sentence_transformers import SentenceTransformer


BATCH_SIZE = 32


def load_synthetic_data(input_path: str) -> list[dict]:
    records = []
    with open(input_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                record["_index"] = i
                records.append(record)
            except json.JSONDecodeError:
                print(f"Warning: skipping malformed line {i} in {input_path}")
    return records


def compute_embeddings(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    prompt_name: str | None = None,
) -> np.ndarray:
    """Encode texts in batches and return L2-normalized embeddings."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        prompt_name=prompt_name,
    )
    return embeddings


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity (embeddings assumed L2-normalized)."""
    return embeddings @ embeddings.T


def greedy_diversity_filter(
    sim_matrix: np.ndarray,
    threshold: float,
) -> list[int]:
    """
    Greedy selection: keep a note only if its max similarity to all
    previously selected notes is below the threshold.
    Returns indices of selected notes (relative to the input matrix).
    """
    n = sim_matrix.shape[0]
    if n == 0:
        return []

    selected = [0]
    for i in range(1, n):
        max_sim_to_selected = max(sim_matrix[i, j] for j in selected)
        if max_sim_to_selected < threshold:
            selected.append(i)

    return selected


def compute_diversity_stats(sim_matrix: np.ndarray) -> dict:
    """Compute distribution statistics from the upper triangle of the sim matrix."""
    n = sim_matrix.shape[0]
    if n < 2:
        return {"count": n, "num_pairs": 0}

    upper_tri = sim_matrix[np.triu_indices(n, k=1)]

    return {
        "count": n,
        "num_pairs": len(upper_tri),
        "mean_similarity": round(float(np.mean(upper_tri)), 4),
        "median_similarity": round(float(np.median(upper_tri)), 4),
        "std_similarity": round(float(np.std(upper_tri)), 4),
        "min_similarity": round(float(np.min(upper_tri)), 4),
        "max_similarity": round(float(np.max(upper_tri)), 4),
        "p90_similarity": round(float(np.percentile(upper_tri, 90)), 4),
        "p95_similarity": round(float(np.percentile(upper_tri, 95)), 4),
        "p99_similarity": round(float(np.percentile(upper_tri, 99)), 4),
        "frac_above_0.85": round(float(np.mean(upper_tri > 0.85)), 4),
        "frac_above_0.90": round(float(np.mean(upper_tri > 0.90)), 4),
        "frac_above_0.95": round(float(np.mean(upper_tri > 0.95)), 4),
    }


def save_jsonl(records: list[dict], output_path: str) -> None:
    with open(output_path, "w") as f:
        for item in records:
            clean = {k: v for k, v in item.items() if not k.startswith("_")}
            f.write(json.dumps(clean) + "\n")


def print_report(report: dict) -> None:
    print("\n" + "=" * 70)
    print("DIVERSITY ANALYSIS REPORT")
    print("=" * 70)
    print(f"Similarity threshold:  {report['threshold']}")
    print(f"Total before:          {report['total_before']}")
    print(f"Total after:           {report['total_after']}")
    print(f"Removed:               {report['total_removed']} "
          f"({report['removal_rate']:.1%})")
    print()

    print("Per-category results:")
    for cat, info in report["categories"].items():
        print(f"\n  {cat}:")
        print(f"    Before:  {info['before_count']}")
        print(f"    After:   {info['after_count']}")
        print(f"    Removed: {info['removed']} ({info['removal_rate']:.1%})")

        b = info["before_stats"]
        print(f"    [BEFORE] mean_sim={b['mean_similarity']:.4f}  "
              f"median={b['median_similarity']:.4f}  "
              f"max={b['max_similarity']:.4f}  "
              f"p95={b['p95_similarity']:.4f}  "
              f">0.90={b['frac_above_0.90']:.1%}")

        a = info["after_stats"]
        if a["num_pairs"] > 0:
            print(f"    [AFTER]  mean_sim={a['mean_similarity']:.4f}  "
                  f"median={a['median_similarity']:.4f}  "
                  f"max={a['max_similarity']:.4f}  "
                  f"p95={a['p95_similarity']:.4f}  "
                  f">0.90={a['frac_above_0.90']:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Diversity analysis and filtering of synthetic clinical notes"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-Embedding-8B",
        help="Embedding model ID",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file with synthetic notes",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="diversity_results",
        help="Output directory for filtered data and report",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Max cosine similarity threshold; notes above this vs. any "
             "already-selected note are removed (default: 0.90)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for embedding computation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for the embedding model (default: auto-detect)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    input_path = project_root / args.input
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Embedding model: {args.model}")
    print(f"Input:           {input_path}")
    print(f"Threshold:       {args.threshold}")
    print(f"Batch size:      {args.batch_size}")
    print()

    records = load_synthetic_data(input_path)
    print(f"Loaded {len(records)} synthetic notes\n")

    categories = defaultdict(list)
    for r in records:
        categories[r["primary_reason"]].append(r)
    for cat, recs in sorted(categories.items()):
        print(f"  {cat}: {len(recs)}")
    print()

    print("Loading embedding model...")
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(args.model, trust_remote_code=True, device=device)
    print(f"Model loaded on {device}.\n")

    total_start = time.time()

    report = {
        "embedding_model": args.model,
        "input_file": str(input_path.name),
        "threshold": args.threshold,
        "categories": {},
    }

    all_filtered = []
    all_removed = []

    for cat, cat_records in sorted(categories.items()):
        print(f"=== Processing: {cat} ({len(cat_records)} notes) ===")
        cat_start = time.time()

        texts = [r["clinical_note"] for r in cat_records]

        print("  Computing embeddings...")
        embeddings = compute_embeddings(model, texts, args.batch_size)

        print("  Computing similarity matrix...")
        sim_matrix = cosine_similarity_matrix(embeddings)

        before_stats = compute_diversity_stats(sim_matrix)

        print(f"  Filtering with threshold={args.threshold}...")
        selected_indices = greedy_diversity_filter(sim_matrix, args.threshold)
        removed_indices = [i for i in range(len(cat_records)) if i not in selected_indices]

        selected_embeddings = embeddings[selected_indices]
        after_sim = cosine_similarity_matrix(selected_embeddings)
        after_stats = compute_diversity_stats(after_sim)

        selected_records = [cat_records[i] for i in selected_indices]
        removed_records = [cat_records[i] for i in removed_indices]
        all_filtered.extend(selected_records)
        all_removed.extend(removed_records)

        cat_elapsed = time.time() - cat_start
        removal_rate = len(removed_indices) / max(1, len(cat_records))

        report["categories"][cat] = {
            "before_count": len(cat_records),
            "after_count": len(selected_records),
            "removed": len(removed_indices),
            "removal_rate": round(removal_rate, 4),
            "before_stats": before_stats,
            "after_stats": after_stats,
            "elapsed_seconds": round(cat_elapsed, 1),
        }

        print(
            f"  Kept {len(selected_records)}/{len(cat_records)} "
            f"(removed {len(removed_indices)}, {removal_rate:.1%}) "
            f"[{cat_elapsed:.1f}s]\n"
        )

    total_elapsed = time.time() - total_start

    report["total_before"] = len(records)
    report["total_after"] = len(all_filtered)
    report["total_removed"] = len(all_removed)
    report["removal_rate"] = round(len(all_removed) / max(1, len(records)), 4)
    report["elapsed_seconds"] = round(total_elapsed, 1)

    filtered_path = output_dir / "filtered_notes.jsonl"
    save_jsonl(all_filtered, filtered_path)

    removed_path = output_dir / "removed_notes.jsonl"
    save_jsonl(all_removed, removed_path)

    per_cat_dir = output_dir / "per_category"
    per_cat_dir.mkdir(exist_ok=True)
    cat_filtered = defaultdict(list)
    for r in all_filtered:
        cat_filtered[r["primary_reason"]].append(r)
    for cat, recs in cat_filtered.items():
        save_jsonl(recs, per_cat_dir / f"{cat.lower()}.jsonl")

    report_path = output_dir / "diversity_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print_report(report)

    print(f"\nElapsed:         {total_elapsed:.1f}s")
    print(f"Filtered output: {filtered_path}")
    print(f"Removed notes:   {removed_path}")
    print(f"Per-category:    {per_cat_dir}/")
    print(f"Report:          {report_path}")


if __name__ == "__main__":
    main()
