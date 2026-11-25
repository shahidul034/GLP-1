"""
Filter synthetic clinical notes based on LLM judge scores.

Joins judgment results with the original synthetic data, applies
configurable score thresholds, and outputs a filtered dataset.
Supports processing a single model or all models at once.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

DIMENSIONS = [
    "clinical_realism",
    "category_alignment",
    "clinical_coherence",
    "note_variability",
    "glp1_relevance",
]


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: str) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def passes_filter(
    judgment: dict,
    min_score: int,
    min_avg: float,
    accept_only: bool,
    dim_mins: dict[str, int],
) -> bool:
    if judgment.get("_parse_error"):
        return False

    if accept_only and not judgment.get("overall_accept"):
        return False

    scores = judgment["scores"]

    for dim in DIMENSIONS:
        if scores.get(dim, -1) < min_score:
            return False

    for dim, threshold in dim_mins.items():
        if scores.get(dim, -1) < threshold:
            return False

    avg = sum(scores[d] for d in DIMENSIONS) / len(DIMENSIONS)
    if avg < min_avg:
        return False

    return True


def process_model(
    synthetic_path: Path,
    judgments_path: Path,
    output_dir: Path,
    min_score: int,
    min_avg: float,
    accept_only: bool,
    dim_mins: dict[str, int],
    model_name: str,
) -> dict:
    """Filter one model's data and return statistics."""
    notes = load_jsonl(synthetic_path)
    judgments = load_jsonl(judgments_path)

    notes_by_index = {}
    for i, note in enumerate(notes):
        notes_by_index[i] = note

    kept = []
    rejected = []
    reject_reasons = defaultdict(int)

    for j in judgments:
        idx = j["_index"]
        note = notes_by_index.get(idx)
        if note is None:
            reject_reasons["missing_note"] += 1
            continue

        if j.get("_parse_error"):
            rejected.append(note)
            reject_reasons["parse_error"] += 1
            continue

        if passes_filter(j, min_score, min_avg, accept_only, dim_mins):
            merged = {
                "primary_reason": note["primary_reason"],
                "clinical_note": note["clinical_note"],
                "judge_scores": j["scores"],
                "judge_avg": round(
                    sum(j["scores"][d] for d in DIMENSIONS) / len(DIMENSIONS), 2
                ),
            }
            kept.append(merged)
        else:
            reasons = []
            scores = j["scores"]
            for dim in DIMENSIONS:
                effective_min = max(min_score, dim_mins.get(dim, 0))
                if scores.get(dim, -1) < effective_min:
                    reasons.append(f"{dim}<{effective_min}")
            avg = sum(scores[d] for d in DIMENSIONS) / len(DIMENSIONS)
            if avg < min_avg:
                reasons.append(f"avg={avg:.2f}<{min_avg}")
            if accept_only and not j.get("overall_accept"):
                reasons.append("not_accepted")

            rejected_entry = {
                "primary_reason": note["primary_reason"],
                "clinical_note": note["clinical_note"],
                "judge_scores": j["scores"],
                "reject_reasons": reasons,
            }
            rejected.append(rejected_entry)
            for r in reasons:
                key = r.split("=")[0].split("<")[0]
                reject_reasons[key] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(kept, output_dir / "filtered_notes.jsonl")
    save_jsonl(rejected, output_dir / "rejected_notes.jsonl")

    per_cat_dir = output_dir / "per_category"
    per_cat_dir.mkdir(exist_ok=True)
    cat_kept = defaultdict(list)
    for r in kept:
        cat_kept[r["primary_reason"]].append(r)
    for cat, recs in cat_kept.items():
        save_jsonl(recs, per_cat_dir / f"{cat.lower()}.jsonl")

    cat_stats = {}
    cat_total = defaultdict(int)
    for j in judgments:
        cat_total[j["primary_reason"]] += 1
    for cat in sorted(set(cat_total.keys()) | set(cat_kept.keys())):
        total = cat_total.get(cat, 0)
        kept_count = len(cat_kept.get(cat, []))
        cat_stats[cat] = {
            "total": total,
            "kept": kept_count,
            "rejected": total - kept_count,
            "keep_rate": round(kept_count / max(1, total), 4),
        }
        if cat_kept.get(cat):
            avgs = [r["judge_avg"] for r in cat_kept[cat]]
            cat_stats[cat]["mean_avg_score"] = round(sum(avgs) / len(avgs), 3)

    stats = {
        "model": model_name,
        "synthetic_file": str(synthetic_path.name),
        "judgments_file": str(judgments_path.name),
        "total_judged": len(judgments),
        "kept": len(kept),
        "rejected": len(rejected),
        "keep_rate": round(len(kept) / max(1, len(judgments)), 4),
        "filter_settings": {
            "min_score": min_score,
            "min_avg": min_avg,
            "accept_only": accept_only,
            "dim_mins": dim_mins if dim_mins else "none",
        },
        "reject_reason_counts": dict(reject_reasons),
        "per_category": cat_stats,
    }

    with open(output_dir / "filter_report.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def print_report(all_stats: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("JUDGMENT-BASED FILTERING REPORT")
    print("=" * 70)

    for stats in all_stats:
        print(f"\n--- {stats['model']} ---")
        print(f"  Total judged:  {stats['total_judged']}")
        print(f"  Kept:          {stats['kept']}")
        print(f"  Rejected:      {stats['rejected']}")
        print(f"  Keep rate:     {stats['keep_rate']:.1%}")

        if stats["reject_reason_counts"]:
            print(f"  Reject reasons:")
            for reason, count in sorted(
                stats["reject_reason_counts"].items(), key=lambda x: -x[1]
            ):
                print(f"    {reason}: {count}")

        print(f"  Per-category:")
        for cat, cs in stats["per_category"].items():
            avg_str = f"  avg={cs['mean_avg_score']:.2f}" if "mean_avg_score" in cs else ""
            print(
                f"    {cat}: {cs['kept']}/{cs['total']} kept "
                f"({cs['keep_rate']:.0%}){avg_str}"
            )

    total_kept = sum(s["kept"] for s in all_stats)
    total_all = sum(s["total_judged"] for s in all_stats)
    print(f"\n{'=' * 70}")
    print(f"GRAND TOTAL: {total_kept}/{total_all} kept "
          f"({total_kept / max(1, total_all):.1%})")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Filter synthetic notes by judge scores"
    )
    parser.add_argument(
        "--judge-dir",
        type=str,
        default="judge_results",
        help="Base directory containing judge results (with model subdirs)",
    )
    parser.add_argument(
        "--synthetic-dir",
        type=str,
        default="synthetic_data",
        help="Base directory containing synthetic data (with model subdirs)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="filtered_data",
        help="Base output directory (model subdirs created automatically)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Model subdirectory names to process (default: all found in judge-dir)",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=3,
        help="Minimum score required on ALL dimensions (default: 3)",
    )
    parser.add_argument(
        "--min-avg",
        type=float,
        default=3.0,
        help="Minimum average score across all dimensions (default: 3.0)",
    )
    parser.add_argument(
        "--accept-only",
        action="store_true",
        default=True,
        help="Only keep notes where overall_accept is true (default: True)",
    )
    parser.add_argument(
        "--no-accept-only",
        action="store_true",
        help="Disable the accept-only filter",
    )
    parser.add_argument(
        "--min-realism",
        type=int,
        default=0,
        help="Minimum clinical_realism score",
    )
    parser.add_argument(
        "--min-alignment",
        type=int,
        default=0,
        help="Minimum category_alignment score",
    )
    parser.add_argument(
        "--min-coherence",
        type=int,
        default=0,
        help="Minimum clinical_coherence score",
    )
    parser.add_argument(
        "--min-variability",
        type=int,
        default=0,
        help="Minimum note_variability score",
    )
    parser.add_argument(
        "--min-glp1",
        type=int,
        default=0,
        help="Minimum glp1_relevance score",
    )
    args = parser.parse_args()

    if args.no_accept_only:
        args.accept_only = False

    project_root = Path(__file__).parent.parent
    judge_base = project_root / args.judge_dir
    synthetic_base = project_root / args.synthetic_dir
    output_base = project_root / args.output_dir

    dim_mins = {}
    if args.min_realism > 0:
        dim_mins["clinical_realism"] = args.min_realism
    if args.min_alignment > 0:
        dim_mins["category_alignment"] = args.min_alignment
    if args.min_coherence > 0:
        dim_mins["clinical_coherence"] = args.min_coherence
    if args.min_variability > 0:
        dim_mins["note_variability"] = args.min_variability
    if args.min_glp1 > 0:
        dim_mins["glp1_relevance"] = args.min_glp1

    if args.models:
        model_dirs = args.models
    else:
        model_dirs = sorted(
            d.name for d in judge_base.iterdir() if d.is_dir()
        )

    print(f"Filter settings:")
    print(f"  min_score (all dims): {args.min_score}")
    print(f"  min_avg:              {args.min_avg}")
    print(f"  accept_only:          {args.accept_only}")
    if dim_mins:
        for dim, val in dim_mins.items():
            print(f"  {dim}:  >= {val}")
    print(f"  Models: {model_dirs}")
    print()

    all_stats = []

    for model_name in model_dirs:
        judgments_path = judge_base / model_name / "judgments.jsonl"
        synthetic_path = synthetic_base / model_name / "all_synthetic_notes.jsonl"

        if not judgments_path.exists():
            print(f"Warning: {judgments_path} not found, skipping {model_name}")
            continue
        if not synthetic_path.exists():
            print(f"Warning: {synthetic_path} not found, skipping {model_name}")
            continue

        print(f"Processing: {model_name}")
        model_output = output_base / model_name

        stats = process_model(
            synthetic_path=synthetic_path,
            judgments_path=judgments_path,
            output_dir=model_output,
            min_score=args.min_score,
            min_avg=args.min_avg,
            accept_only=args.accept_only,
            dim_mins=dim_mins,
            model_name=model_name,
        )
        all_stats.append(stats)

    if not all_stats:
        print("No models found to process.")
        return

    combined = []
    for model_name in model_dirs:
        filtered_path = output_base / model_name / "filtered_notes.jsonl"
        if filtered_path.exists():
            notes = load_jsonl(filtered_path)
            for note in notes:
                note["source_model"] = model_name
            combined.extend(notes)

    if combined:
        output_base.mkdir(parents=True, exist_ok=True)
        save_jsonl(combined, output_base / "all_filtered_notes.jsonl")

    summary = {
        "filter_settings": {
            "min_score": args.min_score,
            "min_avg": args.min_avg,
            "accept_only": args.accept_only,
            "dim_mins": dim_mins if dim_mins else "none",
        },
        "models": {s["model"]: s for s in all_stats},
        "grand_total_judged": sum(s["total_judged"] for s in all_stats),
        "grand_total_kept": sum(s["kept"] for s in all_stats),
        "grand_total_rejected": sum(s["rejected"] for s in all_stats),
        "grand_keep_rate": round(
            sum(s["kept"] for s in all_stats)
            / max(1, sum(s["total_judged"] for s in all_stats)),
            4,
        ),
    }
    with open(output_base / "filter_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print_report(all_stats)
    print(f"\nCombined filtered: {output_base / 'all_filtered_notes.jsonl'} "
          f"({len(combined)} notes)")
    print(f"Summary: {output_base / 'filter_summary.json'}")


if __name__ == "__main__":
    main()
