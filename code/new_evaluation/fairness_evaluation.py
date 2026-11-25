#!/usr/bin/env python3
"""
Fairness Evaluation for GLP-1 Discontinuation Classification

Evaluates model performance across demographic subgroups (age quartile, implied sex)
to identify potential disparities in model performance.

Models evaluated:
- Qwen3.5-9B (fine-tuned) - Decoder
- Llama3.2-3B (fine-tuned) - Decoder
- BioClinical ModernBERT-large (fine-tuned) - Encoder
"""

import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import classification_report, f1_score, accuracy_score
from typing import Dict, List, Tuple, Optional, Any

np.random.seed(42)

PATHS = {
    "test_data": "/home/user/GLP_project/GLP_project_ad/code/finetune/training_data/train_test/test.jsonl",
    "qwen3.5-9b": "/home/user/GLP_project/GLP_project_ad/code/finetune/testing_results/qwen3.5-9b_finetuned_results.json",
    "llama3.2-3b": "/home/user/GLP_project/GLP_project_ad/code/finetune/testing_results/llama3.2-3b_finetuned_results.json",
    "modernbert-large": "/home/user/GLP_project/GLP_project_ad/code/finetune/testing_results/modernbert-large_finetuned_results.json",
}

OUTPUT_DIR = Path("/home/user/GLP_project/GLP_project_ad/code/new_evaluation")
FEATURES_DIR = OUTPUT_DIR / "features"
METRICS_DIR = OUTPUT_DIR / "metrics"
BOOTSTRAP_DIR = OUTPUT_DIR / "bootstrap"
REPORT_DIR = OUTPUT_DIR / "report"

for d in [FEATURES_DIR, METRICS_DIR, BOOTSTRAP_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    "ADVERSE_EFFECTS_TOLERABILITY",
    "ACCESS_COST_SYSTEM_BARRIERS",
    "EFFICACY_EXPECTATION_MISMATCH",
    "PATIENT_PREFERENCE_BELIEFS",
    "PLANNED_GOAL_BASED_DISCONTINUATION",
    "CLINICAL_DECISION_MEDICAL_APPROPRIATENESS",
    "OTHER_UNKNOWN"
]

MALE_PRONOUNS = ['he', 'him', 'his', "he'd", "he's", "himself"]
FEMALE_PRONOUNS = ['she', 'her', 'hers', "she'd", "she's", "herself"]

AGE_REGEX_PATTERNS = [
    r'(\d{1,3})\s*y/?o\b',
    r'(\d{1,3})\s*year[-\s]?old',
    r'\bage\s*:?\s*(\d{1,3})\b',
    r'\bage\s+of\s+(\d{1,3})\b',
    r'(\d{1,3})\s+year\s+old',
]

SPARSITY_THRESHOLD = 10
BOOTSTRAP_ITERATIONS = 1000


def load_test_data() -> pd.DataFrame:
    """Load test data from JSONL file."""
    data = []
    with open(PATHS["test_data"], 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} test samples")
    return df


def load_predictions(model_key: str) -> Dict[str, Any]:
    """Load prediction results for a model."""
    with open(PATHS[model_key], 'r') as f:
        data = json.load(f)
    print(f"Loaded predictions for {model_key}: {len(data.get('predictions', []))} samples")
    return data


def extract_age(text: str) -> Optional[int]:
    """Extract age from clinical note text using regex patterns."""
    text_lower = text.lower()
    for pattern in AGE_REGEX_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            age = int(match.group(1))
            if 0 < age < 120:
                return age
    return None


def count_pronouns(text: str) -> Tuple[int, int]:
    """Count male and female pronouns in text."""
    text_lower = text.lower()
    male_pattern = r"\b(?:he|him|his|himself)\b"
    female_pattern = r"\b(?:she|her|hers|herself)\b"

    male_count = len(re.findall(male_pattern, text_lower))
    female_count = len(re.findall(female_pattern, text_lower))

    return male_count, female_count


def determine_implied_sex(male_count: int, female_count: int) -> str:
    """Determine implied sex from pronoun counts."""
    if male_count > female_count:
        return "Male"
    elif female_count > male_count:
        return "Female"
    else:
        return "Unknown"


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract age and implied sex features from clinical notes."""
    print("\n--- Phase 1: Feature Extraction ---")
    
    ages = []
    male_counts = []
    female_counts = []
    
    for idx, row in df.iterrows():
        age = extract_age(row['clinical_note'])
        ages.append(age)
        male_cnt, female_cnt = count_pronouns(row['clinical_note'])
        male_counts.append(male_cnt)
        female_counts.append(female_cnt)
    
    df['extracted_age'] = ages
    df['male_pronoun_count'] = male_counts
    df['female_pronoun_count'] = female_counts
    df['implied_sex'] = df.apply(
        lambda r: determine_implied_sex(r['male_pronoun_count'], r['female_pronoun_count']), 
        axis=1
    )
    
    valid_ages = [a for a in ages if a is not None]

    print(f"Age extraction: {sum(1 for a in ages if a is not None)}/{len(ages)} successful")
    if valid_ages:
        print(f"Age range: {min(valid_ages)}-{max(valid_ages)}")
    else:
        print("Age range: unavailable")

    df['age_quartile'] = "Unknown"
    if valid_ages:
        age_quartiles = pd.qcut(
            df.loc[df['extracted_age'].notna(), 'extracted_age'],
            q=4,
            labels=['Q1', 'Q2', 'Q3', 'Q4'],
            duplicates='drop'
        )
        df.loc[df['extracted_age'].notna(), 'age_quartile'] = age_quartiles.astype(str)
    
    print(f"\nPronoun-based implied sex distribution:")
    print(df['implied_sex'].value_counts())
    print(f"\nAge quartile distribution:")
    print(df['age_quartile'].value_counts().sort_index())
    
    return df


def merge_predictions(df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    """Merge test data with model predictions."""
    pred_data = load_predictions(model_key)
    predictions = pred_data.get('predictions', [])
    
    if not predictions:
        print(f"  Warning: No prediction-level data available for {model_key}")
        df[f'pred_{model_key}'] = None
        return df
    
    pred_df = pd.DataFrame(predictions)
    pred_df = pred_df.rename(columns={
        'true_label': 'true_label',
        'predicted_label': f'pred_{model_key}'
    })
    
    merged = df.merge(pred_df[['note_id', f'pred_{model_key}']], on='note_id', how='left')
    return merged


def calculate_subgroup_metrics(
    df: pd.DataFrame, 
    pred_col: str, 
    group_col: str,
    group_name: str
) -> Dict[str, Any]:
    """Calculate metrics for a specific subgroup."""
    results = {}
    
    for subgroup in sorted(df[group_col].unique()):
        if pd.isna(subgroup):
            continue
        
        subgroup_df = df[df[group_col] == subgroup].dropna(subset=[pred_col])
        n = len(subgroup_df)
        
        if n == 0:
            continue
        
        y_true = subgroup_df['primary_reason'].tolist()
        y_pred = subgroup_df[pred_col].tolist()
        
        valid_mask = [p is not None and p in CATEGORIES for p in y_pred]
        y_true_valid = [t for t, v in zip(y_true, valid_mask) if v]
        y_pred_valid = [p for p, v in zip(y_pred, valid_mask) if v]
        
        if len(y_true_valid) < SPARSITY_THRESHOLD:
            f1_reliable = False
        else:
            f1_reliable = True
        
        try:
            accuracy = accuracy_score(y_true_valid, y_pred_valid)
        except:
            accuracy = 0.0
        
        if f1_reliable:
            try:
                f1_macro = f1_score(y_true_valid, y_pred_valid, average='macro', labels=CATEGORIES, zero_division=0)
            except:
                f1_macro = 0.0
        else:
            f1_macro = None
        
        results[str(subgroup)] = {
            'accuracy': round(accuracy, 4),
            'f1_macro': round(f1_macro, 4) if f1_macro is not None else None,
            'support': n,
            'f1_reliable': f1_reliable
        }
    
    return results


def calculate_all_subgroup_metrics(df: pd.DataFrame, model_key: str) -> Dict[str, Any]:
    """Calculate metrics for all subgroups for a model."""
    print(f"\n--- Phase 2: Stratified Metric Computation for {model_key} ---")
    
    pred_col = f'pred_{model_key}'
    
    all_metrics = {}
    
    print("  Age Quartile Metrics:")
    age_metrics = calculate_subgroup_metrics(df, pred_col, 'age_quartile', 'age')
    all_metrics['age_quartiles'] = age_metrics
    for q, m in sorted(age_metrics.items()):
        status = "(reliable)" if m['f1_reliable'] else "(unreliable, n<10)"
        print(f"    {q}: Acc={m['accuracy']:.4f}, F1={m['f1_macro']}, n={m['support']} {status}")
    
    print("\n  Implied Sex Metrics:")
    sex_metrics = calculate_subgroup_metrics(df, pred_col, 'implied_sex', 'sex')
    all_metrics['implied_sex'] = sex_metrics
    for s, m in sorted(sex_metrics.items()):
        status = "(reliable)" if m['f1_reliable'] else "(unreliable, n<10)"
        print(f"    {s}: Acc={m['accuracy']:.4f}, F1={m['f1_macro']}, n={m['support']} {status}")
    
    return all_metrics


def bootstrap_delta_f1(
    df: pd.DataFrame,
    pred_col: str,
    group_col: str,
    majority_group: str,
    minority_group: str,
    n_iterations: int = BOOTSTRAP_ITERATIONS
) -> Dict[str, Any]:
    """Perform bootstrap resampling to calculate delta F1 confidence intervals."""
    print(f"\n--- Phase 3: Bootstrap Resampling ---")
    print(f"  Comparing {majority_group} vs {minority_group}")
    
    majority_df = df[df[group_col] == majority_group].dropna(subset=[pred_col])
    minority_df = df[df[group_col] == minority_group].dropna(subset=[pred_col])
    
    majority_size = len(majority_df)
    minority_size = len(minority_df)
    
    print(f"  Majority group ({majority_group}): n={majority_size}")
    print(f"  Minority group ({minority_group}): n={minority_size}")
    
    delta_f1_samples = []
    
    for i in range(n_iterations):
        maj_sample = majority_df.sample(n=majority_size, replace=True)
        min_sample = minority_df.sample(n=minority_size, replace=True)
        
        y_true_maj = maj_sample['primary_reason'].tolist()
        y_pred_maj = maj_sample[pred_col].tolist()
        y_true_maj = [t for t, p in zip(y_true_maj, y_pred_maj) if p is not None and p in CATEGORIES]
        y_pred_maj = [p for p in y_pred_maj if p is not None and p in CATEGORIES]
        
        y_true_min = min_sample['primary_reason'].tolist()
        y_pred_min = min_sample[pred_col].tolist()
        y_true_min = [t for t, p in zip(y_true_min, y_pred_min) if p is not None and p in CATEGORIES]
        y_pred_min = [p for p in y_pred_min if p is not None and p in CATEGORIES]
        
        if len(y_true_maj) >= 2 and len(y_pred_maj) >= 2:
            f1_maj = f1_score(y_true_maj, y_pred_maj, average='macro', labels=CATEGORIES, zero_division=0)
        else:
            f1_maj = 0.0
            
        if len(y_true_min) >= 2 and len(y_pred_min) >= 2:
            f1_min = f1_score(y_true_min, y_pred_min, average='macro', labels=CATEGORIES, zero_division=0)
        else:
            f1_min = 0.0
        
        delta_f1_samples.append(f1_maj - f1_min)
    
    delta_f1_samples = np.array(delta_f1_samples)
    
    mean_delta = np.mean(delta_f1_samples)
    std_delta = np.std(delta_f1_samples)
    ci_lower = np.percentile(delta_f1_samples, 2.5)
    ci_upper = np.percentile(delta_f1_samples, 97.5)
    p_value = np.mean(delta_f1_samples <= 0)
    
    print(f"  Bootstrap results ({n_iterations} iterations):")
    print(f"    Mean ΔF1: {mean_delta:.4f} ± {std_delta:.4f}")
    print(f"    95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"    p-value: {p_value:.4f}")
    print(f"    Significant (p<0.05): {'Yes' if p_value < 0.05 else 'No'}")
    
    return {
        'mean_delta_f1': round(mean_delta, 4),
        'std_delta_f1': round(std_delta, 4),
        'ci_lower': round(ci_lower, 4),
        'ci_upper': round(ci_upper, 4),
        'p_value': round(p_value, 4),
        'significant': bool(p_value < 0.05),
        'n_iterations': n_iterations,
        'majority_group': majority_group,
        'minority_group': minority_group,
        'majority_n': majority_size,
        'minority_n': minority_size
    }


def run_bootstrap_analysis(df: pd.DataFrame, model_key: str) -> Dict[str, Any]:
    """Run bootstrap analysis for a model's predictions."""
    pred_col = f'pred_{model_key}'
    results = {}
    
    implied_sex_metrics = calculate_subgroup_metrics(df, pred_col, 'implied_sex', 'sex')
    age_quartile_metrics = calculate_subgroup_metrics(df, pred_col, 'age_quartile', 'age')
    
    if 'Male' in implied_sex_metrics and 'Female' in implied_sex_metrics:
        male_f1 = implied_sex_metrics['Male']['f1_macro']
        female_f1 = implied_sex_metrics['Female']['f1_macro']
        
        if male_f1 is not None and female_f1 is not None:
            if male_f1 >= female_f1:
                majority, minority = 'Male', 'Female'
            else:
                majority, minority = 'Female', 'Male'
            
            results['implied_sex'] = bootstrap_delta_f1(df, pred_col, 'implied_sex', majority, minority)
    
    age_keys = sorted(age_quartile_metrics.keys())
    if len(age_keys) >= 2:
        q1_f1 = age_quartile_metrics[age_keys[0]]['f1_macro']
        q4_f1 = age_quartile_metrics[age_keys[-1]]['f1_macro']
        
        if q1_f1 is not None and q4_f1 is not None:
            if q1_f1 >= q4_f1:
                majority_q, minority_q = age_keys[0], age_keys[-1]
            else:
                majority_q, minority_q = age_keys[-1], age_keys[0]
            
            results['age_quartiles'] = bootstrap_delta_f1(df, pred_col, 'age_quartile', majority_q, minority_q)
    
    return results


def generate_fairness_report(
    all_metrics: Dict[str, Dict],
    all_bootstrap: Dict[str, Dict]
) -> str:
    """Generate a comprehensive fairness report."""
    
    report = """# Exploratory Subgroup Evaluation Report

## Overview

This report evaluates model performance across proxy demographic subgroups (age quartile, implied sex)
to characterize subgroup robustness in the GLP-1 discontinuation classification task.

**Models Evaluated:**
- Qwen3.5-9B (fine-tuned) - Decoder LLM
- Llama3.2-3B (fine-tuned) - Decoder LLM  
- BioClinical ModernBERT-large (fine-tuned) - Encoder LLM

**Test Set Size:** 320 samples

**Sparsity Threshold:** n < 10 → F1 score flagged as unreliable

---

## 1. Feature Extraction Summary

### Age Extraction
- Used regex patterns targeting forms such as `(\\d+)y/?o`, `(\\d+)year-old`, and `age (\\d+)`
- Age quartiles were computed only for notes with extractable ages
- Notes without an extracted age were assigned to an `Unknown` group

### Pronoun-Based Implied Sex
- Male pronouns: he, him, his, he'd, he's, himself
- Female pronouns: she, her, hers, she'd, she's, herself
- Assigned to majority pronoun sex; "Unknown" if tied or zero

---

## 2. Subgroup Metrics

"""
    
    for model_key in all_metrics:
        report += f"### {model_key}\n\n"
        
        report += "#### Age Quartile Performance\n"
        report += "| Quartile | Accuracy | Macro-F1 | Support | Reliable |\n"
        report += "|----------|----------|----------|---------|----------|\n"
        
        age_metrics = all_metrics[model_key].get('age_quartiles', {})
        for q in sorted(age_metrics.keys()):
            m = age_metrics[q]
            reliable = "Yes" if m['f1_reliable'] else "No (n<10)"
            f1_str = f"{m['f1_macro']:.4f}" if m['f1_macro'] is not None else "N/A"
            report += f"| {q} | {m['accuracy']:.4f} | {f1_str} | {m['support']} | {reliable} |\n"
        
        report += "\n#### Implied Sex Performance\n"
        report += "| Sex | Accuracy | Macro-F1 | Support | Reliable |\n"
        report += "|-----|----------|----------|---------|----------|\n"
        
        sex_metrics = all_metrics[model_key].get('implied_sex', {})
        for s in sorted(sex_metrics.keys()):
            m = sex_metrics[s]
            reliable = "Yes" if m['f1_reliable'] else "No (n<10)"
            f1_str = f"{m['f1_macro']:.4f}" if m['f1_macro'] is not None else "N/A"
            report += f"| {s} | {m['accuracy']:.4f} | {f1_str} | {m['support']} | {reliable} |\n"
        
        report += "\n"

    report += """---

## 3. Statistical Significance Testing

Bootstrap resampling (1,000 iterations) was used to compute 95% confidence intervals
for the F1 score difference (ΔF1) between demographic groups.

### Interpretation
- **p-value < 0.05**: Statistically significant performance disparity between groups
- **95% CI does not include 0**: Indicates significant difference in F1 scores
- **Mean ΔF1**: Average difference in F1 (positive = majority group performs better)

"""
    
    for model_key in all_bootstrap:
        report += f"### {model_key}\n\n"
        
        bootstrap_results = all_bootstrap[model_key]
        
        if 'implied_sex' in bootstrap_results:
            bs = bootstrap_results['implied_sex']
            sig_marker = "**" if bs['significant'] else ""
            report += f"#### Implied Sex: Male vs Female{sig_marker}\n"
            report += f"- Majority: {bs['majority_group']} (n={bs['majority_n']})\n"
            report += f"- Minority: {bs['minority_group']} (n={bs['minority_n']})\n"
            report += f"- Mean ΔF1: {bs['mean_delta_f1']:.4f} ± {bs['std_delta_f1']:.4f}\n"
            report += f"- 95% CI: [{bs['ci_lower']:.4f}, {bs['ci_upper']:.4f}]\n"
            report += f"- p-value: {bs['p_value']:.4f}\n"
            report += f"- Statistically Significant: {'Yes' if bs['significant'] else 'No'}\n\n"
        
        if 'age_quartiles' in bootstrap_results:
            bs = bootstrap_results['age_quartiles']
            sig_marker = "**" if bs['significant'] else ""
            report += f"#### Age: {bs['majority_group']} vs {bs['minority_group']}{sig_marker}\n"
            report += f"- Majority: {bs['majority_group']} (n={bs['majority_n']})\n"
            report += f"- Minority: {bs['minority_group']} (n={bs['minority_n']})\n"
            report += f"- Mean ΔF1: {bs['mean_delta_f1']:.4f} ± {bs['std_delta_f1']:.4f}\n"
            report += f"- 95% CI: [{bs['ci_lower']:.4f}, {bs['ci_upper']:.4f}]\n"
            report += f"- p-value: {bs['p_value']:.4f}\n"
            report += f"- Statistically Significant: {'Yes' if bs['significant'] else 'No'}\n\n"
        
        report += "---\n\n"

    report += """## 4. Key Findings

### Exploratory Summary

| Model | Sex Disparity | Age Disparity |
|-------|---------------|---------------|
"""
    
    for model_key in all_bootstrap:
        bootstrap_results = all_bootstrap[model_key]
        sex_sig = "Yes" if bootstrap_results.get('implied_sex', {}).get('significant') else "No"
        age_sig = "Yes" if bootstrap_results.get('age_quartiles', {}).get('significant') else "No"
        report += f"| {model_key} | {sex_sig} | {age_sig} |\n"
    
    report += """

### Interpretation Guidelines

- **Statistically Significant disparity (p < 0.05)**: The observed performance difference
  between proxy-defined groups is unlikely under the bootstrap procedure used here.
  
- **Non-significant disparity**: Any observed difference could be due to sampling variability.
  The model appears to perform similarly across groups.

### Notes

- Age is extracted from clinical note text using pattern matching and may be incomplete
- Implied sex is based on pronoun usage and may not reflect self-identified gender
- These subgroup definitions are proxy variables and should not be treated as validated demographics
- Small subgroup sizes limit the reliability of subgroup-level comparisons

---

*Report generated automatically by fairness_evaluation.py*
"""
    
    return report


def main():
    print("=" * 80)
    print("Fairness Evaluation for GLP-1 Discontinuation Classification")
    print("=" * 80)
    
    df = load_test_data()
    df = extract_features(df)
    
    features_path = FEATURES_DIR / "test_with_subgroups.csv"
    df.to_csv(features_path, index=False)
    print(f"\nFeatures saved to: {features_path}")
    
    all_metrics = {}
    all_bootstrap = {}
    
    model_keys = ['qwen3.5-9b', 'llama3.2-3b', 'modernbert-large']
    
    for model_key in model_keys:
        df_model = merge_predictions(df.copy(), model_key)
        
        metrics = calculate_all_subgroup_metrics(df_model, model_key)
        all_metrics[model_key] = metrics
        
        metrics_path = METRICS_DIR / f"{model_key}_subgroup_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")
        
        bootstrap_results = run_bootstrap_analysis(df_model, model_key)
        all_bootstrap[model_key] = bootstrap_results
        
        bootstrap_path = BOOTSTRAP_DIR / f"{model_key}_bootstrap_results.json"
        with open(bootstrap_path, 'w') as f:
            json.dump(bootstrap_results, f, indent=2)
        print(f"Bootstrap results saved to: {bootstrap_path}")
    
    report = generate_fairness_report(all_metrics, all_bootstrap)
    
    report_path = REPORT_DIR / "fairness_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nFairness report saved to: {report_path}")
    
    print("\n" + "=" * 80)
    print("Fairness Evaluation Complete!")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  Features: {FEATURES_DIR / 'test_with_subgroups.csv'}")
    print(f"  Metrics: {METRICS_DIR}/<model>_subgroup_metrics.json")
    print(f"  Bootstrap: {BOOTSTRAP_DIR}/<model>_bootstrap_results.json")
    print(f"  Report: {REPORT_DIR / 'fairness_report.md'}")


if __name__ == "__main__":
    main()