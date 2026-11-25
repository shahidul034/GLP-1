#!/usr/bin/env python3
"""
Phase 5: Generate Final Report for Ablation Study
Compiles results from all phases into a comprehensive Markdown table.
"""

import json
from pathlib import Path

RESULTS_DIR = Path("/home/user/GLP_project/GLP_project_ad/code/ablation_studies/results")
FILTERING_REPORTS_DIR = RESULTS_DIR / "filtering_reports"
REPORT_OUTPUT = Path("/home/user/GLP_project/GLP_project_ad/results_summary/results/ablation_study_results.md")

THRESHOLDS = [1.0, 0.95, 0.90, 0.85, 0.80]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("Phase 5: Generating Ablation Study Report")
    print("=" * 70)

    eval_results_path = RESULTS_DIR / "evaluation_results.json"
    leakage_path = RESULTS_DIR / "filtering_reports" / "semantic_leakage_report.json"

    if not eval_results_path.exists():
        print("ERROR: evaluation_results.json not found. Run evaluation first.")
        return
    if not leakage_path.exists():
        print("WARNING: semantic_leakage_report.json not found.")

    eval_results = load_json(eval_results_path)
    leakage = load_json(leakage_path) if leakage_path.exists() else {}

    threshold_data = []

    for threshold in THRESHOLDS:
        label = str(threshold).replace(".", "_")

        filter_report_path = FILTERING_REPORTS_DIR / f"threshold_{label}.json"
        filter_report = load_json(filter_report_path) if filter_report_path.exists() else {}

        retained = filter_report.get("total_after", "N/A")
        removed = filter_report.get("total_removed", "N/A")
        removal_rate = filter_report.get("removal_rate", "N/A")

        leakage_key = threshold
        leakage_info = leakage.get(str(threshold), leakage.get(threshold, {}))
        if isinstance(leakage_info, dict):
            mean_sim = leakage_info.get("_overall_mean", leakage_info.get("overall_mean", "N/A"))
        else:
            mean_sim = "N/A"

        modernbert_key = f"modernbert_{threshold}"
        modernbert_result = eval_results.get(modernbert_key, {})
        modernbert_metrics = modernbert_result.get("metrics", {})
        modernbert_f1 = modernbert_metrics.get("f1_macro", "N/A")
        modernbert_acc = modernbert_metrics.get("accuracy", "N/A")

        qwen_key = f"qwen_{threshold}"
        qwen_result = eval_results.get(qwen_key, {})
        qwen_metrics = qwen_result.get("metrics", {})
        qwen_f1 = qwen_metrics.get("f1_macro", "N/A")
        qwen_acc = qwen_metrics.get("accuracy", "N/A")

        threshold_data.append({
            "threshold": threshold,
            "retained": retained,
            "removed": removed,
            "removal_rate": removal_rate,
            "mean_within_cat_sim": mean_sim,
            "modernbert_f1": modernbert_f1,
            "modernbert_acc": modernbert_acc,
            "qwen_f1": qwen_f1,
            "qwen_acc": qwen_acc,
        })

    report_lines = []
    report_lines.append("# Ablation Study: Semantic Similarity Threshold Sweep\n")
    report_lines.append("## Objective\n")
    report_lines.append("Measure how training data diversity (cosine similarity threshold) impacts model performance on GLP-1 receptor agonist discontinuation classification.\n")
    report_lines.append("## Configuration\n")
    report_lines.append("| Parameter | Value |\n| --- | --- |\n")
    report_lines.append(f"| Raw training notes | 50,885 |\n")
    report_lines.append(f"| Test set | 200 notes (sampled from gpt5.4.jsonl) |\n")
    report_lines.append(f"| Encoder model | thomas-sounack/BioClinical-ModernBERT-large |\n")
    report_lines.append(f"| Decoder model | Qwen/Qwen3.5-9B (Unsloth LoRA) |\n")
    report_lines.append(f"| Embedding model | Qwen/Qwen3-Embedding-8B |\n")
    report_lines.append(f"| Semantic leakage model | thomas-sounack/BioClinical-ModernBERT-large |\n")
    report_lines.append(f"| Thresholds swept | {THRESHOLDS} |\n")
    report_lines.append("\n## Results\n")

    report_lines.append("### Main Comparison Table\n\n")
    report_lines.append("| Threshold | Retained Notes | Mean Within-Category Similarity | ModernBERT-large (Macro-F1) | Qwen3.5-9B (Macro-F1) |\n")
    report_lines.append("| --- | --- | --- | --- | --- |\n")
    for d in threshold_data:
        th = d["threshold"]
        ret = d["retained"]
        sim = d["mean_within_cat_sim"]
        m_f1 = f"{d['modernbert_f1']:.4f}" if isinstance(d["modernbert_f1"], float) else str(d["modernbert_f1"])
        q_f1 = f"{d['qwen_f1']:.4f}" if isinstance(d["qwen_f1"], float) else str(d["qwen_f1"])
        report_lines.append(f"| {th} | {ret} | {sim} | {m_f1} | {q_f1} |\n")

    report_lines.append("\n### Detailed Results\n\n")
    report_lines.append("| Threshold | Retained | Removed | Removal Rate | ModernBERT Acc | ModernBERT F1 | Qwen Acc | Qwen F1 |\n")
    report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
    for d in threshold_data:
        th = d["threshold"]
        ret = d["retained"]
        rem = d["removed"]
        rr = f"{d['removal_rate']:.2%}" if isinstance(d["removal_rate"], float) else str(d["removal_rate"])
        m_acc = f"{d['modernbert_acc']:.4f}" if isinstance(d["modernbert_acc"], float) else str(d["modernbert_acc"])
        m_f1 = f"{d['modernbert_f1']:.4f}" if isinstance(d["modernbert_f1"], float) else str(d["modernbert_f1"])
        q_acc = f"{d['qwen_acc']:.4f}" if isinstance(d["qwen_acc"], float) else str(d["qwen_acc"])
        q_f1 = f"{d['qwen_f1']:.4f}" if isinstance(d["qwen_f1"], float) else str(d["qwen_f1"])
        report_lines.append(f"| {th} | {ret} | {rem} | {rr} | {m_acc} | {m_f1} | {q_acc} | {q_f1} |\n")

    report_lines.append("\n### Semantic Leakage Details (per category)\n\n")
    for threshold in THRESHOLDS:
        label = str(threshold).replace(".", "_")
        leakage_info = leakage.get(str(threshold), {})
        if not leakage_info:
            continue
        report_lines.append(f"#### Threshold {threshold}\n\n")
        report_lines.append("| Category | Mean Similarity | Std | Sample Pairs |\n")
        report_lines.append("| --- | --- | --- | --- |\n")
        for cat, info in leakage_info.items():
            if cat.startswith("_"):
                continue
            if isinstance(info, dict) and "mean_similarity" in info:
                ms = info["mean_similarity"]
                std = info.get("std_similarity", "N/A")
                np_ = info.get("num_pairs", "N/A")
                ms_str = f"{ms:.4f}" if isinstance(ms, float) else str(ms)
                std_str = f"{std:.4f}" if isinstance(std, float) else str(std)
                report_lines.append(f"| {cat} | {ms_str} | {std_str} | {np_} |\n")
        report_lines.append("\n")

    report_lines.append("---\n*Generated by ablation_studies pipeline*\n")

    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    report_text = "".join(report_lines)
    REPORT_OUTPUT.write_text(report_text)

    print(f"Report saved to: {REPORT_OUTPUT}")
    print("\n" + "=" * 70)
    print("Report Preview")
    print("=" * 70)
    print(report_text[:3000])
    print("\nPhase 5 complete.")


if __name__ == "__main__":
    main()