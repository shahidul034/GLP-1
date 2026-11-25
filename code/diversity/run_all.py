"""
run_all.py – Master runner for the full Diversity Evaluation Pipeline
=====================================================================
Runs all four steps sequentially and generates a final summary report.

Usage:
    python run_all.py              # Run all steps
    python run_all.py --skip 3     # Run all except step 3 (UMAP)
    python run_all.py --only 1 4   # Run only steps 1 and 4
"""
import argparse
import json
import os
import time
import sys

from config import RESULTS_DIR


def run_step(step_num: int) -> float:
    """Run a single step and return elapsed time."""
    step_map = {
        1: ("Lexical Diversity", "step1_lexical_diversity"),
        2: ("Semantic Diversity (BERTScore)", "step2_semantic_diversity"),
        3: ("UMAP Visualization", "step3_umap_visualization"),
        4: ("Demographic Coverage", "step4_demographic_coverage"),
    }

    name, module = step_map[step_num]
    print(f"\n{'#' * 70}")
    print(f"#  STEP {step_num}: {name}")
    print(f"{'#' * 70}\n")

    start = time.time()
    mod = __import__(module)
    mod.main()
    elapsed = time.time() - start

    print(f"\n  ⏱  Step {step_num} completed in {elapsed:.1f}s")
    return elapsed


def generate_summary(timings: dict[int, float]):
    """Generate a final summary report."""
    print(f"\n{'=' * 70}")
    print("  DIVERSITY EVALUATION – FINAL SUMMARY")
    print(f"{'=' * 70}")

    summary = {"timings": {}}

    # Load each step's results if available
    step_files = {
        1: "step1_lexical_diversity.json",
        2: "step2_semantic_diversity.json",
        3: "step3_umap_metadata.json",
        4: "step4_demographic_coverage.json",
    }

    for step, fname in step_files.items():
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)

            if step == 1:
                corpus = data.get("train", {}).get("corpus", {})
                print(f"\n  Step 1 – Lexical Diversity (Train):")
                print(f"    TTR:                 {corpus.get('ttr', 'N/A')}")
                print(f"    Unique bigram ratio: {corpus.get('unique_2gram_ratio', 'N/A')}")
                print(f"    Unique trigram ratio: {corpus.get('unique_3gram_ratio', 'N/A')}")
                summary["lexical_diversity"] = corpus

            elif step == 2:
                print(f"\n  Step 2 – Semantic Diversity (Pairwise Semantic Similarity):")
                for cat, res in data.items():
                    mean_sim = res.get("mean_sim", 0)
                    print(f"    {cat:<22}: mean Sim = {mean_sim:.4f}")
                summary["semantic_diversity"] = {
                    k: v.get("mean_sim") for k, v in data.items()
                }

            elif step == 3:
                print(f"\n  Step 3 – UMAP Visualization:")
                print(f"    Model: {data.get('model', 'N/A')}")
                print(f"    Embedding dim: {data.get('embedding_dim', 'N/A')}")
                print(f"    Train notes: {data.get('n_train', 'N/A')}")
                print(f"    Test notes: {data.get('n_test', 'N/A')}")
                summary["umap"] = data

            elif step == 4:
                print(f"\n  Step 4 – Demographic Coverage:")
                demos = data.get("demographics", {})
                for field in ["age", "sex", "bmi", "race", "a1c"]:
                    info = demos.get(field, {})
                    n = info.get("n_extracted", 0)
                    print(f"    {field:<8}: {n} notes extracted")
                summary["demographics"] = demos

        if step in timings:
            summary["timings"][f"step_{step}"] = round(timings[step], 1)

    total_time = sum(timings.values())
    summary["timings"]["total"] = round(total_time, 1)
    print(f"\n  ⏱  Total time: {total_time:.1f}s")

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "diversity_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Summary saved to {summary_path}")

    # List all output files
    print(f"\n  Output files:")
    for root, dirs, files in os.walk(RESULTS_DIR):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            size = os.path.getsize(fpath)
            rel = os.path.relpath(fpath, RESULTS_DIR)
            if size > 1024 * 1024:
                size_str = f"{size / (1024*1024):.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"    {rel:<45} {size_str:>10}")


def main():
    parser = argparse.ArgumentParser(description="Run Diversity Evaluation Pipeline")
    parser.add_argument("--skip", nargs="+", type=int, default=[],
                        help="Steps to skip (e.g., --skip 3)")
    parser.add_argument("--only", nargs="+", type=int, default=[],
                        help="Run only these steps (e.g., --only 1 4)")
    args = parser.parse_args()

    all_steps = [1, 2, 3, 4]

    if args.only:
        steps = [s for s in all_steps if s in args.only]
    else:
        steps = [s for s in all_steps if s not in args.skip]

    print("=" * 70)
    print("  DIVERSITY EVALUATION PIPELINE")
    print(f"  Steps to run: {steps}")
    print("=" * 70)

    timings = {}
    for step in steps:
        timings[step] = run_step(step)

    generate_summary(timings)


if __name__ == "__main__":
    main()
