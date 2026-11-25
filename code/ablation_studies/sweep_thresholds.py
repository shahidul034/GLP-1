#!/usr/bin/env python3
"""
Phase 2: Threshold Sweeping and Dataset Construction
Applies greedy diversity filtering at multiple thresholds and generates
filtered training datasets for each threshold setting.
"""

import json
import time
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda"
BATCH_SIZE = 32
SIMILARITY_MODEL = "thomas-sounack/BioClinical-ModernBERT-large"
EMBEDDINGS_DIR = Path("/home/user/GLP_project/GLP_project_ad/code/ablation_studies/embeddings")
OUTPUT_DIR = Path("/home/user/GLP_project/GLP_project_ad/code/ablation_studies")
DATASETS_DIR = OUTPUT_DIR / "datasets"
RESULTS_DIR = OUTPUT_DIR / "results" / "filtering_reports"

THRESHOLDS = [1.0, 0.95, 0.90, 0.85, 0.80]
SAMPLE_PAIRS_PER_CATEGORY = 200


def cosine_similarity_matrix(embeddings):
    return embeddings @ embeddings.T


def greedy_diversity_filter(sim_matrix, threshold):
    n = sim_matrix.shape[0]
    if n == 0:
        return []
    selected = [0]
    for i in range(1, n):
        max_sim = max(sim_matrix[i, j] for j in selected)
        if max_sim < threshold:
            selected.append(i)
    return selected


def compute_diversity_stats(sim_matrix):
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


def load_records_with_embeddings():
    emb_path = EMBEDDINGS_DIR / "notes_emb.pt"
    ids_path = EMBEDDINGS_DIR / "notes_ids.json"
    cats_path = EMBEDDINGS_DIR / "notes_categories.json"

    embeddings = torch.load(emb_path)
    note_ids = json.loads(Path(ids_path).read_text())
    categories = json.loads(Path(cats_path).read_text())

    input_path = Path("/home/user/GLP_project/GLP_project_ad/data/combined_synthetic/combined.jsonl")
    records = []
    with open(input_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_index"] = i
            records.append(record)

    assert len(records) == len(note_ids) == embeddings.shape[0]

    for i, (r, e, c) in enumerate(zip(records, embeddings, categories)):
        r["_embedding"] = e.to(torch.float32).numpy()
        r["_category"] = c
        r["_note_id"] = note_ids[i]

    return records


def compute_semantic_leakage(records_by_cat, model, tokenizer):
    print("\nComputing semantic leakage (mean within-category pairwise similarity)...")
    results = {}

    for cat, cat_records in records_by_cat.items():
        if len(cat_records) < 2:
            results[cat] = {"mean_similarity": None, "num_pairs": 0}
            continue

        texts = [r["clinical_note"] for r in cat_records]
        n_pairs = min(SAMPLE_PAIRS_PER_CATEGORY, len(texts) * (len(texts) - 1) // 2)
        if n_pairs == 0:
            results[cat] = {"mean_similarity": None, "num_pairs": 0}
            continue

        rng = np.random.default_rng(42)
        pairs = []
        indices = list(range(len(texts)))
        while len(pairs) < SAMPLE_PAIRS_PER_CATEGORY:
            i, j = rng.choice(indices, 2, replace=False)
            if i != j:
                pairs.append((min(i, j), max(i, j)))
            if len(set(pairs)) == len(indices) * (len(indices) - 1) // 2:
                break
        pairs = list(set(pairs))[:SAMPLE_PAIRS_PER_CATEGORY]

        with torch.no_grad():
            encoded = tokenizer(
                [texts[i] for i, j in pairs] + [texts[j] for i, j in pairs],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
            outputs = model(**encoded)
            emb1 = outputs.last_hidden_state[:, 0, :]
            emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)

        half = len(pairs)
        emb2 = emb1[half:]
        emb1 = emb1[:half]

        sims = (emb1 * emb2).sum(dim=1).cpu().numpy()
        results[cat] = {
            "mean_similarity": round(float(np.mean(sims)), 4),
            "std_similarity": round(float(np.std(sims)), 4),
            "num_pairs": len(pairs),
        }

    overall_mean = np.mean([v["mean_similarity"] for v in results.values() if v["mean_similarity"] is not None])
    results["_overall_mean"] = round(float(overall_mean), 4)
    return results


def run_threshold_sweep(records_by_cat, thresholds):
    reports = {}

    for threshold in thresholds:
        print(f"\n{'=' * 60}")
        print(f"Threshold: {threshold}")
        print(f"{'=' * 60}")

        start_time = time.time()
        filtered_by_cat = {}
        global_report = {
            "threshold": threshold,
            "categories": {},
            "total_before": 0,
            "total_after": 0,
        }

        for cat, cat_records in sorted(records_by_cat.items()):
            before_count = len(cat_records)
            global_report["total_before"] += before_count

            embeddings = np.stack([r["_embedding"] for r in cat_records])
            sim_matrix = cosine_similarity_matrix(embeddings)

            before_stats = compute_diversity_stats(sim_matrix)

            selected_indices = greedy_diversity_filter(sim_matrix, threshold)
            selected_records = [cat_records[i] for i in selected_indices]

            if len(selected_records) > 1:
                sel_emb = embeddings[selected_indices]
                after_sim = cosine_similarity_matrix(sel_emb)
                after_stats = compute_diversity_stats(after_sim)
            else:
                after_stats = {"count": len(selected_records), "num_pairs": 0}

            filtered_by_cat[cat] = selected_records

            global_report["categories"][cat] = {
                "before_count": before_count,
                "after_count": len(selected_records),
                "removed": before_count - len(selected_records),
                "removal_rate": round((before_count - len(selected_records)) / max(1, before_count), 4),
                "before_stats": before_stats,
                "after_stats": after_stats,
            }
            global_report["total_after"] += len(selected_records)

        global_report["elapsed_seconds"] = round(time.time() - start_time, 1)
        global_report["total_removed"] = global_report["total_before"] - global_report["total_after"]
        global_report["removal_rate"] = round(global_report["total_removed"] / max(1, global_report["total_before"]), 4)

        label = str(threshold).replace(".", "_")
        out_path = DATASETS_DIR / f"train_{label}.jsonl"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = RESULTS_DIR / f"threshold_{label}.json"

        all_filtered = []
        for cat_records in filtered_by_cat.values():
            all_filtered.extend(cat_records)

        with open(out_path, "w") as f:
            for r in all_filtered:
                clean = {k: v for k, v in r.items() if not k.startswith("_")}
                f.write(json.dumps(clean) + "\n")

        with open(report_path, "w") as f:
            json.dump(global_report, f, indent=2)

        reports[threshold] = global_report
        print(f"  Before: {global_report['total_before']}, After: {global_report['total_after']}")
        print(f"  Removed: {global_report['total_removed']} ({global_report['removal_rate']:.1%})")
        print(f"  Time: {global_report['elapsed_seconds']}s")
        print(f"  Output: {out_path}")

    return reports


def main():
    print("=" * 70)
    print("Phase 2: Threshold Sweeping and Dataset Construction")
    print("=" * 70)
    print(f"Thresholds: {THRESHOLDS}")
    print()

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading records and embeddings...")
    records = load_records_with_embeddings()
    print(f"Loaded {len(records)} records\n")

    records_by_cat = defaultdict(list)
    for r in records:
        records_by_cat[r["_category"]].append(r)

    print("Per-category counts:")
    for cat, recs in sorted(records_by_cat.items()):
        print(f"  {cat}: {len(recs)}")
    print()

    reports = run_threshold_sweep(records_by_cat, THRESHOLDS)

    print("\nLoading BioClinical-ModernBERT-large for semantic leakage...")
    tok = AutoTokenizer.from_pretrained(SIMILARITY_MODEL, trust_remote_code=True)
    mod = AutoModel.from_pretrained(SIMILARITY_MODEL, trust_remote_code=True)
    mod = mod.to(DEVICE)
    mod.eval()

    leakage_results = {}
    for threshold in THRESHOLDS:
        label = str(threshold).replace(".", "_")
        out_path = DATASETS_DIR / f"train_{label}.jsonl"

        filtered_records = []
        with open(out_path) as f:
            for line in f:
                filtered_records.append(json.loads(line.strip()))

        filtered_by_cat = defaultdict(list)
        for r in filtered_records:
            filtered_by_cat[r["primary_reason"]].append(r)

        label = str(threshold).replace(".", "_")
        print(f"\nSemantic leakage for threshold={threshold}:")
        cat_results = compute_semantic_leakage(filtered_by_cat, mod, tok)
        leakage_results[threshold] = cat_results
        for cat, res in cat_results.items():
            if cat != "_overall_mean" and res["mean_similarity"] is not None:
                print(f"  {cat}: mean_sim={res['mean_similarity']:.4f} (n={res['num_pairs']})")
        print(f"  OVERALL MEAN: {cat_results.get('_overall_mean', 'N/A')}")

    leakage_path = RESULTS_DIR / "semantic_leakage_report.json"
    with open(leakage_path, "w") as f:
        json.dump(leakage_results, f, indent=2)
    print(f"\nSemantic leakage report saved to {leakage_path}")

    del mod
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Phase 2 Complete - Dataset Summary")
    print("=" * 70)
    print(f"{'Threshold':<12} {'Before':<10} {'After':<10} {'Removed':<10} {'RemRate':<10}")
    for threshold in THRESHOLDS:
        r = reports[threshold]
        print(f"{threshold:<12} {r['total_before']:<10} {r['total_after']:<10} {r['total_removed']:<10} {r['removal_rate']:<10.2%}")
    print("Phase 2 complete.")


if __name__ == "__main__":
    main()