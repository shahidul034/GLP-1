#!/usr/bin/env python3
"""
Evaluate GPT-5.4, GPT-5.4-mini, and GPT-5.4-nano using OpenAI API for GLP-1 discontinuation classification.
Supports zero-shot and few-shot prompting modes.
"""

import os
import sys
import json
import re
import time
import asyncio
import argparse
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import AsyncOpenAI
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

# Configuration
TEST_DATA_PATH = "/home/user/GLP_project/GLP_project_ad/code/finetune/training_data/train_test/test.jsonl"
TRAIN_DATA_PATH = "/home/user/GLP_project/GLP_project_ad/code/finetune/training_data/train_test/train.jsonl"
RESULTS_DIR = "/home/user/GLP_project/GLP_project_ad/code/gpt5.4/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CATEGORIES = [
    "ADVERSE_EFFECTS_TOLERABILITY",
    "ACCESS_COST_SYSTEM_BARRIERS",
    "EFFICACY_EXPECTATION_MISMATCH",
    "PATIENT_PREFERENCE_BELIEFS",
    "PLANNED_GOAL_BASED_DISCONTINUATION",
    "CLINICAL_DECISION_MEDICAL_APPROPRIATENESS",
    "OTHER_UNKNOWN"
]

INSTRUCTION = """You are a clinical NLP specialist. Analyze the following clinical note and classify the primary reason for GLP-1 receptor agonist discontinuation.

Categories:
- ADVERSE_EFFECTS_TOLERABILITY: GI side effects, systemic symptoms, serious adverse events, injection site reactions
- ACCESS_COST_SYSTEM_BARRIERS: Insurance issues, high costs, prior authorization, supply shortages
- EFFICACY_EXPECTATION_MISMATCH: Weight plateau, insufficient results, medication not working as expected
- PATIENT_PREFERENCE_BELIEFS: Safety fears, misinformation, stigma, preference for natural approaches
- PLANNED_GOAL_BASED_DISCONTINUATION: Reached goal weight, planned tapering, underlying condition improved
- CLINICAL_DECISION_MEDICAL_APPROPRIATENESS: Contraindications, drug interactions, switching therapy, pregnancy
- OTHER_UNKNOWN: Reason unclear, lost to follow-up, ambiguous documentation

Respond with ONLY a JSON object containing:
1. "primary_reason": The main discontinuation category
2. "confidence": Your confidence level (high/medium/low)
3. "supporting_evidence": Key phrases from the note supporting your classification
4. "secondary_factors": Any additional contributing factors (if applicable)

DO NOT show your thinking process. Output ONLY the JSON object."""

MODEL_CONFIGS = [
    {"name": "gpt-5.4", "model_id": "gpt-5.4"},
    {"name": "gpt-5.4-mini", "model_id": "gpt-5.4-mini"},
    {"name": "gpt-5.4-nano", "model_id": "gpt-5.4-nano"},
]


def load_test_data():
    test_data = []
    with open(TEST_DATA_PATH, 'r') as f:
        for line in f:
            test_data.append(json.loads(line.strip()))
    print(f"Loaded {len(test_data)} test samples")
    return test_data


def load_few_shot_examples():
    """Load 7 fixed few-shot examples, one per category."""
    target_ids = {
        'ADVERSE_EFFECTS_TOLERABILITY': 'oss_400',
        'ACCESS_COST_SYSTEM_BARRIERS': 'oss_94',
        'EFFICACY_EXPECTATION_MISMATCH': 'oss_1196',
        'PATIENT_PREFERENCE_BELIEFS': 'oss_1679',
        'PLANNED_GOAL_BASED_DISCONTINUATION': 'oss_1872',
        'CLINICAL_DECISION_MEDICAL_APPROPRIATENESS': 'oss_988',
        'OTHER_UNKNOWN': 'oss_1432',
    }
    
    examples = []
    with open(TRAIN_DATA_PATH, 'r') as f:
        for line in f:
            d = json.loads(line.strip())
            note_id = d.get('note_id')
            if note_id in target_ids.values():
                examples.append(d)
            if len(examples) == 7:
                break
    
    # Sort by category order for consistency
    examples_sorted = []
    for cat in CATEGORIES:
        for ex in examples:
            if ex.get('primary_reason') == cat:
                examples_sorted.append(ex)
                break
    
    print(f"Loaded {len(examples_sorted)} few-shot examples")
    return examples_sorted


def is_result_saved(config, mode="few_shot"):
    filename = f"{config['name']}_{mode}_results.json"
    filepath = os.path.join(RESULTS_DIR, filename)
    return os.path.exists(filepath)


def build_messages(examples, test_note, mode="few_shot"):
    """Build messages for OpenAI API with zero-shot or few-shot examples."""
    messages = [{"role": "system", "content": INSTRUCTION}]
    
    if mode == "few_shot":
        for ex in examples:
            messages.append({
                "role": "user",
                "content": f"Clinical Note:\n{ex['clinical_note']}"
            })
            messages.append({
                "role": "assistant",
                "content": f'```json\n{{\n  "primary_reason": "{ex["primary_reason"]}",\n  "confidence": "high",\n  "supporting_evidence": "See clinical note for details",\n  "secondary_factors": []\n}}\n```'
            })
    
    messages.append({
        "role": "user",
        "content": f"Clinical Note:\n{test_note}"
    })
    
    return messages


def parse_output(raw_output):
    """Parse the model output to extract primary_reason."""
    try:
        # Remove thinking/reasoning sections
        cleaned = re.sub(r'Thinking Process:.*?(?=\n\n|\{)', '', raw_output, flags=re.DOTALL)
        
        # Remove markdown code blocks
        cleaned = re.sub(r'```json\s*', '', cleaned)
        cleaned = re.sub(r'```\s*', '', cleaned)
        cleaned = cleaned.strip()
        
        # Find JSON object
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            # Normalize whitespace for JSON parsing
            json_str = re.sub(r'\s+', ' ', json_str)
            result = json.loads(json_str)
            primary_reason = result.get("primary_reason", "").upper().strip()
            
            for cat in CATEGORIES:
                if cat in primary_reason or primary_reason in cat:
                    return cat, result.get("confidence", "medium")
            
            return primary_reason if primary_reason in CATEGORIES else None, result.get("confidence", "medium")
        else:
            return None, "low"
            
    except (json.JSONDecodeError, AttributeError) as e:
        return None, "low"
    except Exception as e:
        return None, "low"


def compute_metrics(y_true, y_pred, y_scores=None, model_name=""):
    """Compute evaluation metrics matching the reference format."""
    results = {}
    
    valid_mask = [p is not None for p in y_pred]
    y_true_valid = [t for t, v in zip(y_true, valid_mask) if v]
    y_pred_valid = [p for p, v in zip(y_pred, valid_mask) if v]
    
    if not y_true_valid:
        print(f"  Warning: No valid predictions for {model_name}")
        return None
    
    results["accuracy"] = accuracy_score(y_true_valid, y_pred_valid)
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_valid, y_pred_valid, average='macro', labels=CATEGORIES, zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_valid, y_pred_valid, average='weighted', labels=CATEGORIES, zero_division=0
    )
    
    results["precision_macro"] = precision_macro
    results["recall_macro"] = recall_macro
    results["f1_macro"] = f1_macro
    results["precision_weighted"] = precision_weighted
    results["recall_weighted"] = recall_weighted
    results["f1_weighted"] = f1_weighted
    
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        y_true_valid, y_pred_valid, labels=CATEGORIES, zero_division=0
    )
    
    results["per_class"] = {}
    for i, cat in enumerate(CATEGORIES):
        results["per_class"][cat] = {
            "precision": float(precision_per[i]),
            "recall": float(recall_per[i]),
            "f1": float(f1_per[i]),
            "support": int(support_per[i])
        }
    
    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=CATEGORIES)
    results["confusion_matrix"] = cm.tolist()
    
    if y_scores is not None:
        try:
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true_valid, classes=CATEGORIES)
            
            if y_scores.shape[1] == len(CATEGORIES):
                results["roc_auc_macro"] = roc_auc_score(
                    y_true_bin, y_scores, multi_class='ovr', average='macro'
                )
            else:
                results["roc_auc_macro"] = None
        except Exception as e:
            results["roc_auc_macro"] = None
    else:
        results["roc_auc_macro"] = None
    
    results["classification_report"] = classification_report(
        y_true_valid, y_pred_valid, labels=CATEGORIES, zero_division=0
    )
    
    return results


def evaluate_model(config, test_data, few_shot_examples, client, mode="few_shot"):
    """Evaluate a single model using OpenAI API with zero-shot or few-shot prompting and batch processing."""
    print(f"\n{'#'*80}")
    print(f"# Evaluating {config['name']} ({mode})")
    print(f"{'#'*80}")
    
    if is_result_saved(config, mode):
        print(f"  SKIPPING: Results already exist for {config['name']} ({mode})")
        return None
    
    # Build all messages upfront
    all_messages = []
    for sample in test_data:
        clinical_note = sample.get("clinical_note", "")
        messages = build_messages(few_shot_examples, clinical_note, mode)
        all_messages.append(messages)
    
    print(f"  Processing {len(test_data)} samples in batches...")
    
    # Process in batches using async
    async def process_batch(messages_batch, indices):
        tasks = []
        for idx, messages in enumerate(messages_batch):
            task = client.chat.completions.create(
                model=config["model_id"],
                messages=messages,
            )
            tasks.append((task, indices[idx]))
        return await asyncio.gather(*[t[0] for t in tasks])
    
    # Run async processing
    async def run_all():
        batch_size = 50
        results = []
        for i in range(0, len(all_messages), batch_size):
            batch = all_messages[i:i+batch_size]
            indices = list(range(i, i+len(batch)))
            print(f"  Processing batch {i//batch_size + 1}/{(len(all_messages) + batch_size - 1)//batch_size}...")
            batch_results = await asyncio.gather(*[
                client.chat.completions.create(
                    model=config["model_id"],
                    messages=messages,
                )
                for messages in batch
            ])
            for idx, response in zip(indices, batch_results):
                results.append((idx, response))
            # Small delay between batches to avoid rate limiting
            await asyncio.sleep(0.5)
        return results
    
    responses_with_idx = asyncio.run(run_all())
    # Sort by index to maintain order
    responses_with_idx.sort(key=lambda x: x[0])
    
    y_true = []
    y_pred = []
    y_scores = []
    predictions_detail = []
    
    for idx, response in responses_with_idx:
        sample = test_data[idx]
        note_id = sample.get("note_id", f"sample_{idx}")
        true_label = sample.get("primary_reason", "")
        raw_output = response.choices[0].message.content
        
        pred_label, confidence = parse_output(raw_output)
        
        confidence_map = {"high": 0.9, "medium": 0.6, "low": 0.3}
        score = confidence_map.get(confidence, 0.5) if isinstance(confidence, str) else float(confidence)
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        
        score_vec = np.zeros(len(CATEGORIES))
        if pred_label in CATEGORIES:
            cat_idx = CATEGORIES.index(pred_label)
            score_vec[cat_idx] = score
        y_scores.append(score_vec)
        
        predictions_detail.append({
            "note_id": note_id,
            "true_label": true_label,
            "predicted_label": pred_label,
            "confidence": confidence,
            "raw_output": raw_output[:500],
        })
    
    y_scores_array = np.array(y_scores) if y_scores else None
    metrics = compute_metrics(y_true, y_pred, y_scores_array, model_name=config["name"])
    
    if metrics is None:
        return None
    
    results = {
        "model_name": config["name"],
        "model_type": mode,
        "metrics": {k: v for k, v in metrics.items() if k not in ["confusion_matrix", "classification_report", "per_class"]},
        "per_class_metrics": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report"],
        "predictions": predictions_detail,
    }
    
    return results


def save_results(results, config, mode="few_shot"):
    """Save results in the same format as reference."""
    filename = f"{config['name']}_{mode}_results.json"
    filepath = os.path.join(RESULTS_DIR, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")
    
    cm = results.get("confusion_matrix")
    if cm:
        cm_df = pd.DataFrame(cm, columns=CATEGORIES, index=CATEGORIES)
        cm_filepath = os.path.join(RESULTS_DIR, f"{config['name']}_{mode}_confusion_matrix.csv")
        cm_df.to_csv(cm_filepath)
        print(f"Confusion matrix saved to: {cm_filepath}")


def create_summary():
    """Create summary of all results."""
    all_results = []
    
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith("_results.json") and ("zero_shot" in filename or "few_shot" in filename):
            filepath = os.path.join(RESULTS_DIR, filename)
            with open(filepath, 'r') as f:
                result = json.load(f)
                all_results.append(result)
    
    summary_data = []
    
    for result in all_results:
        if result is None:
            continue
        
        row = {
            "model_name": result["model_name"],
            "model_type": result["model_type"],
            "accuracy": result["metrics"].get("accuracy"),
            "precision_macro": result["metrics"].get("precision_macro"),
            "recall_macro": result["metrics"].get("recall_macro"),
            "f1_macro": result["metrics"].get("f1_macro"),
            "precision_weighted": result["metrics"].get("precision_weighted"),
            "recall_weighted": result["metrics"].get("recall_weighted"),
            "f1_weighted": result["metrics"].get("f1_weighted"),
            "roc_auc_macro": result["metrics"].get("roc_auc_macro"),
        }
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(RESULTS_DIR, "evaluation_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSummary CSV saved to: {csv_path}")
    
    json_path = os.path.join(RESULTS_DIR, "evaluation_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary JSON saved to: {json_path}")
    
    print("\n" + "="*120)
    print("EVALUATION SUMMARY")
    print("="*120)
    print(df.to_string(index=False))
    print("="*120)


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT-5.4 models for GLP-1 discontinuation classification")
    parser.add_argument("--mode", type=str, choices=["zero_shot", "few_shot"], default="few_shot",
                        help="Prompting mode: zero_shot or few_shot (default: few_shot)")
    args = parser.parse_args()
    
    mode = args.mode
    
    print("="*80)
    print("GLP-1 Discontinuation Classification - GPT-5.4 Evaluation")
    print("="*80)
    print(f"\nTest dataset: {TEST_DATA_PATH}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Number of models: {len(MODEL_CONFIGS)}")
    print(f"Categories: {len(CATEGORIES)}")
    print(f"Mode: {mode}")
    if mode == "few_shot":
        print(f"Few-shot examples: 7 (one per category)")
    print(f"Batch size: 50 concurrent requests")
    
    # Check for OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\nERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set it using: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    client = AsyncOpenAI(api_key=api_key)
    
    test_data = load_test_data()
    few_shot_examples = load_few_shot_examples() if mode == "few_shot" else []
    
    for config in MODEL_CONFIGS:
        try:
            result = evaluate_model(config, test_data, few_shot_examples, client, mode)
            if result:
                save_results(result, config, mode)
        except Exception as e:
            print(f"ERROR evaluating model {config['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("GENERATING SUMMARY")
    print("="*80)
    create_summary()
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
