#!/usr/bin/env python3
"""
Comprehensive evaluation script for GLP-1 discontinuation classification models.
Evaluates both base and fine-tuned models on the test dataset.
Computes: Accuracy, Precision, Recall, F1-Score, ROC-AUC
"""

import os
import sys
import json
import re
import gc
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Paths
MODELS_DIR = "/home/user/GLP_project/GLP_project_ad/code/finetune/models"
TEST_DATA_PATH = "/home/user/GLP_project/GLP_project_ad/code/finetune/training_data/train_test/test.jsonl"
RESULTS_DIR = "/home/user/GLP_project/GLP_project_ad/code/finetune/testing_results"

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "confusion_matrices"), exist_ok=True)

# Classification categories
CATEGORIES = [
    "ADVERSE_EFFECTS_TOLERABILITY",
    "ACCESS_COST_SYSTEM_BARRIERS",
    "EFFICACY_EXPECTATION_MISMATCH",
    "PATIENT_PREFERENCE_BELIEFS",
    "PLANNED_GOAL_BASED_DISCONTINUATION",
    "CLINICAL_DECISION_MEDICAL_APPROPRIATENESS",
    "OTHER_UNKNOWN"
]

# Instruction template
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

# Model configurations
MODEL_CONFIGS = [
    {
        "name": "qwen3-4b",
        "base_model_name": "unsloth/Qwen3-4B-Instruct-2507",
        "finetuned_path": os.path.join(MODELS_DIR, "qwen3-4b-finetune-fp16"),
        "format": "sharegpt",
        "chat_template": "qwen3-instruct",
        "use_fast_model": False,
    },
    {
        "name": "qwen3-14b",
        "base_model_name": "unsloth/Qwen3-14B",
        "finetuned_path": os.path.join(MODELS_DIR, "qwen3-14b-finetune-fp16"),
        "format": "sharegpt",
        "chat_template": "qwen3-instruct",
        "use_fast_model": False,
    },
    {
        "name": "qwen3.5-0.8b",
        "base_model_name": "unsloth/Qwen3.5-0.8B",
        "finetuned_path": os.path.join(MODELS_DIR, "qwen3.5-0-8b-finetune-fp16"),
        "format": "sharegpt",
        "chat_template": "qwen3-instruct",
        "use_fast_model": False,
    },
    {
        "name": "qwen3.5-2b",
        "base_model_name": "unsloth/Qwen3.5-2B",
        "finetuned_path": os.path.join(MODELS_DIR, "qwen3.5-2b-finetune-fp16"),
        "format": "sharegpt",
        "chat_template": "qwen3-instruct",
        "use_fast_model": False,
    },
    {
        "name": "qwen3.5-9b",
        "base_model_name": "unsloth/Qwen3.5-9B",
        "finetuned_path": os.path.join(MODELS_DIR, "qwen3.5-9b-finetune-fp16"),
        "format": "sharegpt",
        "chat_template": "qwen3-instruct",
        "use_fast_model": False,
    },
    {
        "name": "qwen3.5-27b",
        "base_model_name": "unsloth/Qwen3.5-27B",
        "finetuned_path": os.path.join(MODELS_DIR, "qwen3.5-27b-finetune-fp16"),
        "format": "sharegpt",
        "chat_template": "qwen3-instruct",
        "use_fast_model": False,
    },
    {
        "name": "llama3-1-8b",
        "base_model_name": "unsloth/Llama-3.1-8B",
        "finetuned_path": os.path.join(MODELS_DIR, "llama3-1-8b-finetune-fp16"),
        "format": "alpaca",
        "chat_template": "llama-3.1",
        "use_fast_model": False,
    },
    {
        "name": "llama3.2-1b",
        "base_model_name": "unsloth/Llama-3.2-1B-Instruct",
        "finetuned_path": os.path.join(MODELS_DIR, "llama3.2-1b-finetune"),
        "format": "sharegpt",
        "chat_template": "llama-3.1",
        "use_fast_model": False,
    },
    {
        "name": "llama3.2-3b",
        "base_model_name": "unsloth/Llama-3.2-3B-Instruct",
        "finetuned_path": os.path.join(MODELS_DIR, "llama3-2-3b-finetune-fp16"),
        "format": "sharegpt",
        "chat_template": "llama-3.1",
        "use_fast_model": False,
    },
    {
        "name": "phi4",
        "base_model_name": "unsloth/Phi-4",
        "finetuned_path": os.path.join(MODELS_DIR, "phi4-finetune-fp16"),
        "format": "sharegpt",
        "chat_template": "phi-4",
        "use_fast_model": False,
    },
    {
        "name": "gemma3-4b",
        "base_model_name": "unsloth/gemma-3-4b-it",
        "finetuned_path": os.path.join(MODELS_DIR, "gemma3-4b-finetune-fp16"),
        "format": "sharegpt",
        "chat_template": "gemma-3",
        "use_fast_model": True,
    },
    {
        "name": "gemma4-e2b",
        "base_model_name": "unsloth/gemma-4-E2B-it",
        "finetuned_path": os.path.join(MODELS_DIR, "gemma4-e2b-finetune-fp16"),
        "format": "sharegpt",
        "chat_template": "gemma-4",
        "use_fast_model": True,
    },
]

# Keyword mapping for base model evaluation
KEYWORD_MAP = {
    "ADVERSE_EFFECTS_TOLERABILITY": [
        "side effect", "nausea", "vomiting", "gi ", "intolerable", "adverse",
        "injection site", "pancreatitis", "gallbladder", "diarrhea", "constipation",
        "abdominal pain", "tolerability", "poorly tolerated", "unable to tolerate",
        "severe nausea", "persistent vomiting", "gi symptoms"
    ],
    "ACCESS_COST_SYSTEM_BARRIERS": [
        "insurance", "cost", "prior authorization", "coverage", "expensive",
        "afford", "financial", "formulary", "denied", "not covered",
        "out of pocket", "copay", "supply shortage", "unavailable",
        "pharmacy issue", "medication access", "cannot afford", "too expensive"
    ],
    "EFFICACY_EXPECTATION_MISMATCH": [
        "not working", "plateau", "ineffective", "no improvement", "weight gain",
        "insufficient", "didn't work", "stopped working", "no longer effective",
        "weight loss plateau", "minimal effect", "lack of efficacy",
        "disappointing results", "not meeting expectations", "poor response"
    ],
    "PATIENT_PREFERENCE_BELIEFS": [
        "preference", "natural", "fear", "concern", "unsafe", "don't want",
        "patient declined", "patient refused", "patient choice", "personal belief",
        "stigma", "misinformation", "safety concern", "worried about",
        "prefer lifestyle", "natural approach", "doesn't want medication"
    ],
    "PLANNED_GOAL_BASED_DISCONTINUATION": [
        "goal", "target reached", "planned", "taper", "achieved",
        "reached goal", "goal weight", "planned discontinuation", "time-limited",
        "original plan", "as planned", "successful", "met target",
        "achieved weight loss", "glycemic goal reached", "planned stop"
    ],
    "CLINICAL_DECISION_MEDICAL_APPROPRIATENESS": [
        "contraindication", "interaction", "pregnancy", "switch", "medical decision",
        "provider decision", "clinical judgment", "drug interaction", "comorbidity",
        "renal", "hepatic", "cardiac", "thyroid", "medullary thyroid",
        "men", "men2", "pancreatitis history", "guideline", "deprescribe",
        "switching therapy", "change medication", "medical reason"
    ],
    "OTHER_UNKNOWN": [
        "unknown", "unclear", "lost", "ambiguous", "no reason",
        "lost to follow-up", "reason unclear", "unclear reason", "not documented",
        "patient stopped", "discontinued without explanation", "no clear reason",
        "ambiguous documentation"
    ]
}


def load_test_data():
    test_data = []
    with open(TEST_DATA_PATH, 'r') as f:
        for line in f:
            test_data.append(json.loads(line.strip()))
    print(f"Loaded {len(test_data)} test samples")
    return test_data


def load_model(config, is_base=True):
    model_name = config["base_model_name"] if is_base else config["finetuned_path"]
    use_fast_model = config["use_fast_model"]
    
    print(f"\n{'='*80}")
    print(f"Loading {'BASE' if is_base else 'FINETUNED'} model: {config['name']}")
    print(f"Model path: {model_name}")
    print(f"{'='*80}")
    
    try:
        if use_fast_model:
            from unsloth import FastModel
            model, tokenizer = FastModel.from_pretrained(
                model_name=model_name,
                max_seq_length=16384,
                load_in_4bit=False,
            )
        else:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=16384,
                load_in_4bit=False,
            )
        
        if not is_base and config["format"] == "sharegpt":
            from unsloth.chat_templates import get_chat_template
            tokenizer = get_chat_template(tokenizer, chat_template=config["chat_template"])
        
        print(f"Model loaded successfully: {config['name']} ({'base' if is_base else 'finetuned'})")
        return model, tokenizer
        
    except Exception as e:
        print(f"ERROR loading model {config['name']}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def unload_model(model, tokenizer):
    print(f"Unloading model and clearing GPU memory...")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory cleared")


def generate_prompt(config, tokenizer, clinical_note, is_base=True):
    if config["format"] == "alpaca":
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{INSTRUCTION}

### Input:
{clinical_note}

### Response:
"""
    else:
        from unsloth.chat_templates import get_chat_template
        tokenizer_with_template = get_chat_template(tokenizer, chat_template=config["chat_template"])
        
        messages = [
            {"role": "user", "content": INSTRUCTION + "\n\nClinical Note:\n" + clinical_note}
        ]
        
        prompt = tokenizer_with_template.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        if config["chat_template"] == "gemma-3":
            prompt = prompt.removeprefix('<bos>')
    
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


def parse_output_finetuned(raw_output):
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
        print(f"  Warning: JSON parsing failed: {str(e)[:100]}")
        return None, "low"


def parse_output_base(raw_output, clinical_note=""):
    text = (raw_output + " " + clinical_note).lower()
    
    scores = defaultdict(float)
    
    for category, keywords in KEYWORD_MAP.items():
        for keyword in keywords:
            if keyword.lower() in text:
                weight = len(keyword.split())
                scores[category] += weight
    
    if not scores:
        return "OTHER_UNKNOWN", 0.0
    
    max_score = max(scores.values())
    normalized_scores = {cat: score / max_score for cat, score in scores.items()}
    
    best_category = max(scores, key=scores.get)
    confidence = normalized_scores.get(best_category, 0.0)
    
    return best_category, confidence


def compute_metrics(y_true, y_pred, y_scores=None, model_name=""):
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
            print(f"  Warning: ROC-AUC computation failed: {str(e)[:100]}")
            results["roc_auc_macro"] = None
    else:
        results["roc_auc_macro"] = None
    
    results["classification_report"] = classification_report(
        y_true_valid, y_pred_valid, labels=CATEGORIES, zero_division=0
    )
    
    return results


def evaluate_model(config, test_data, is_base=True):
    model_type = "base" if is_base else "finetuned"
    print(f"\n{'#'*80}")
    print(f"# Evaluating {config['name']} ({model_type})")
    print(f"{'#'*80}")
    
    model, tokenizer = load_model(config, is_base=is_base)
    if model is None:
        return None
    
    if not is_base:
        try:
            if config["use_fast_model"]:
                from unsloth import FastModel
                FastModel.for_inference(model)
            else:
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
        
        if idx % 10 == 0:
            print(f"  Processing sample {idx+1}/{total_samples}...")
        
        try:
            prompt = generate_prompt(config, tokenizer, clinical_note, is_base=is_base)
            raw_output = generate_response(model, tokenizer, prompt)
            
            if is_base:
                pred_label, confidence = parse_output_base(raw_output, clinical_note)
            else:
                pred_label, confidence = parse_output_finetuned(raw_output)
            
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
    metrics = compute_metrics(y_true, y_pred, y_scores_array, model_name=config["name"])
    
    unload_model(model, tokenizer)
    
    if metrics is None:
        return None
    
    results = {
        "model_name": config["name"],
        "model_type": model_type,
        "metrics": {k: v for k, v in metrics.items() if k not in ["confusion_matrix", "classification_report", "per_class"]},
        "per_class_metrics": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report"],
        "predictions": predictions_detail,
    }
    
    return results


def save_results(results, config, is_base=True):
    model_type = "base" if is_base else "finetuned"
    filename = f"{config['name']}_{model_type}_results.json"
    filepath = os.path.join(RESULTS_DIR, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")
    
    cm = results.get("confusion_matrix")
    if cm:
        cm_df = pd.DataFrame(cm, columns=CATEGORIES, index=CATEGORIES)
        cm_filepath = os.path.join(RESULTS_DIR, "confusion_matrices", f"{config['name']}_{model_type}_confusion_matrix.csv")
        cm_df.to_csv(cm_filepath)
        print(f"Confusion matrix saved to: {cm_filepath}")


def create_summary(all_results):
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
    print("="*80)
    print("GLP-1 Discontinuation Classification - Model Evaluation")
    print("="*80)
    print(f"\nTest dataset: {TEST_DATA_PATH}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Number of models: {len(MODEL_CONFIGS)}")
    print(f"Categories: {len(CATEGORIES)}")
    
    test_data = load_test_data()
    
    all_results = []
    
    print("\n" + "="*80)
    print("PHASE 1: Evaluating BASE MODELS")
    print("="*80)
    
    for config in MODEL_CONFIGS:
        try:
            result = evaluate_model(config, test_data, is_base=True)
            if result:
                save_results(result, config, is_base=True)
                all_results.append(result)
            else:
                print(f"WARNING: Evaluation failed for base model {config['name']}")
        except Exception as e:
            print(f"ERROR evaluating base model {config['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("PHASE 2: Evaluating FINETUNED MODELS")
    print("="*80)
    
    for config in MODEL_CONFIGS:
        try:
            result = evaluate_model(config, test_data, is_base=False)
            if result:
                save_results(result, config, is_base=False)
                all_results.append(result)
            else:
                print(f"WARNING: Evaluation failed for finetuned model {config['name']}")
        except Exception as e:
            print(f"ERROR evaluating finetuned model {config['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("GENERATING SUMMARY")
    print("="*80)
    create_summary(all_results)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
