#!/usr/bin/env python3
"""
Evaluate Qwen3.5-9B base and finetuned models.
"""

import os
import sys
import json
import re
import gc
import torch
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODELS_DIR = "/home/user/GLP_project/GLP_project_ad/code/finetune/models"
TEST_DATA_PATH = "/home/user/GLP_project/GLP_project_ad/code/finetune/training_data/train_test/test.jsonl"
RESULTS_DIR = "/home/user/GLP_project/GLP_project_ad/code/finetune/testing_results"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "confusion_matrices"), exist_ok=True)

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

Respond with a JSON object containing:
1. "primary_reason": The main discontinuation category
2. "confidence": Your confidence level (high/medium/low)
3. "supporting_evidence": Key phrases from the note supporting your classification
4. "secondary_factors": Any additional contributing factors (if applicable)"""


def load_test_data():
    test_data = []
    with open(TEST_DATA_PATH, 'r') as f:
        for line in f:
            test_data.append(json.loads(line.strip()))
    print(f"Loaded {len(test_data)} test samples")
    return test_data


def load_model(model_path, model_type):
    print(f"\n{'='*80}")
    print(f"Loading {model_type} model: {model_path}")
    print(f"{'='*80}")
    
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=4096,
            load_in_4bit=False,
        )
        
        tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")
        
        print(f"Model loaded successfully: {model_path}")
        return model, tokenizer
        
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def unload_model(model, tokenizer):
    print(f"Unloading model and clearing GPU memory...")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def generate_prompt(tokenizer, clinical_note):
    messages = [
        {"role": "user", "content": INSTRUCTION + "\n\nClinical Note:\n" + clinical_note}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    return prompt


def generate_response(model, tokenizer, prompt, max_new_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            use_cache=True,
        )
    
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response.strip()


def parse_output(raw_output):
    try:
        cleaned = re.sub(r'```json\s*', '', raw_output)
        cleaned = re.sub(r'```\s*', '', cleaned)
        cleaned = cleaned.strip()
        
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            primary_reason = result.get("primary_reason", "").upper().strip()
            
            for cat in CATEGORIES:
                if cat in primary_reason or primary_reason in cat:
                    return cat, result.get("confidence", "medium")
            
            return primary_reason if primary_reason in CATEGORIES else None, result.get("confidence", "medium")
        else:
            return None, "low"
            
    except (json.JSONDecodeError, AttributeError) as e:
        return None, "low"


def compute_metrics(y_true, y_pred, y_scores=None, model_name=""):
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score,
        confusion_matrix,
        classification_report
    )
    
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
    
    results["precision_macro"] = float(precision_macro)
    results["recall_macro"] = float(recall_macro)
    results["f1_macro"] = float(f1_macro)
    results["precision_weighted"] = float(precision_weighted)
    results["recall_weighted"] = float(recall_weighted)
    results["f1_weighted"] = float(f1_weighted)
    
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
    
    if y_scores is not None and len(y_scores) > 0:
        try:
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true_valid, classes=CATEGORIES)
            
            y_scores_arr = np.array(y_scores)
            if y_scores_arr.shape[1] == len(CATEGORIES):
                results["roc_auc_macro"] = roc_auc_score(
                    y_true_bin, y_scores_arr, multi_class='ovr', average='macro'
                )
            else:
                results["roc_auc_macro"] = None
        except Exception as e:
            print(f"  ROC-AUC error: {e}")
            results["roc_auc_macro"] = None
    else:
        results["roc_auc_macro"] = None
    
    results["classification_report"] = classification_report(
        y_true_valid, y_pred_valid, labels=CATEGORIES, zero_division=0
    )
    
    return results


def evaluate_model(model_name, model_path, model_type, test_data):
    print(f"\n{'#'*80}")
    print(f"# Evaluating {model_name} ({model_type})")
    print(f"# Path: {model_path}")
    print(f"{'#'*80}")
    
    filename = f"{model_name}_{model_type}_results.json"
    filepath = os.path.join(RESULTS_DIR, filename)
    
    if os.path.exists(filepath):
        print(f"  Results already exist at {filepath}, skipping...")
        with open(filepath, 'r') as f:
            return json.load(f)
    
    model, tokenizer = load_model(model_path, model_type)
    if model is None:
        return None
    
    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    except:
        pass
    
    y_true = []
    y_pred = []
    y_scores = []
    predictions_detail = []
    
    total_samples = len(test_data)
    
    for idx, sample in enumerate(test_data):
        note_id = sample.get("note_id", f"sample_{idx}")
        clinical_note = sample.get("clinical_note", "")
        true_label = sample.get("primary_reason", "")
        
        if idx % 20 == 0:
            print(f"  Processing sample {idx+1}/{total_samples}...")
        
        try:
            prompt = generate_prompt(tokenizer, clinical_note)
            raw_output = generate_response(model, tokenizer, prompt)
            
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
                "confidence": score,
                "raw_output": raw_output[:1000],
            })
            
        except Exception as e:
            print(f"  Error processing sample {idx}: {str(e)[:100]}")
            y_true.append(true_label)
            y_pred.append(None)
            y_scores.append(np.zeros(len(CATEGORIES)))
            predictions_detail.append({
                "note_id": note_id,
                "true_label": true_label,
                "predicted_label": None,
                "error": str(e)
            })
    
    y_scores_array = np.array(y_scores) if y_scores else None
    metrics = compute_metrics(y_true, y_pred, y_scores_array, model_name=model_name)
    
    unload_model(model, tokenizer)
    
    if metrics is None:
        return None
    
    results = {
        "model_name": model_name,
        "model_type": model_type,
        "metrics": {k: v for k, v in metrics.items() if k not in ["confusion_matrix", "classification_report", "per_class"]},
        "per_class_metrics": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report"],
        "predictions": predictions_detail,
    }
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")
    
    cm = results.get("confusion_matrix")
    if cm:
        cm_df = pd.DataFrame(cm, columns=CATEGORIES, index=CATEGORIES)
        cm_filepath = os.path.join(RESULTS_DIR, "confusion_matrices", f"{model_name}_{model_type}_confusion_matrix.csv")
        cm_df.to_csv(cm_filepath)
        print(f"Confusion matrix saved to: {cm_filepath}")
    
    return results


def main():
    print("="*80)
    print("Qwen3.5-9B Model Evaluation - Base and Finetuned")
    print("="*80)
    print(f"\nTest dataset: {TEST_DATA_PATH}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Categories: {CATEGORIES}")
    
    test_data = load_test_data()
    
    base_model_name = "Qwen3.5-9B"
    base_model_path = "unsloth/Qwen3.5-9B"
    finetuned_model_path = os.path.join(MODELS_DIR, "qwen3.5-9b-finetune-fp16")
    
    print("\n" + "="*80)
    print("STEP 1: Evaluating BASE model")
    print("="*80)
    base_results = evaluate_model(base_model_name, base_model_path, "base", test_data)
    
    print("\n" + "="*80)
    print("STEP 2: Evaluating FINETUNED model")
    print("="*80)
    finetuned_results = evaluate_model(f"{base_model_name}-finetuned", finetuned_model_path, "finetuned", test_data)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    if base_results:
        print(f"\nBase Model Results:")
        print(f"  Accuracy: {base_results['metrics'].get('accuracy', 'N/A')}")
        print(f"  F1 Macro: {base_results['metrics'].get('f1_macro', 'N/A')}")
        print(f"  ROC-AUC: {base_results['metrics'].get('roc_auc_macro', 'N/A')}")
    
    if finetuned_results:
        print(f"\nFinetuned Model Results:")
        print(f"  Accuracy: {finetuned_results['metrics'].get('accuracy', 'N/A')}")
        print(f"  F1 Macro: {finetuned_results['metrics'].get('f1_macro', 'N/A')}")
        print(f"  ROC-AUC: {finetuned_results['metrics'].get('roc_auc_macro', 'N/A')}")
    
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()