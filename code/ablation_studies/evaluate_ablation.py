#!/usr/bin/env python3
"""
Phase 4: Evaluation of Ablation Study Models
Evaluates all fine-tuned ModernBERT-large and Qwen3.5-9B models
on the fixed 320-note test set. Uses vLLM for decoder models.
"""

import os
import sys
import json
import re
import gc
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

warnings_path = Path(__file__).parent / ".." / ".." / "code" / "ablation_studies"
sys.path.insert(0, str(warnings_path))
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import warnings
warnings.filterwarnings("ignore")

DATASETS_DIR = Path("/home/user/GLP_project/GLP_project_ad/code/ablation_studies/datasets")
MODELS_DIR = Path("/home/user/GLP_project/GLP_project_ad/code/ablation_studies/models")
RESULTS_DIR = Path("/home/user/GLP_project/GLP_project_ad/code/ablation_studies/results")
TEST_DATA_PATH = "/home/user/GLP_project/GLP_project_ad/code/ablation_studies/datasets/test_gpt54_200.jsonl"

CATEGORIES = [
    "ADVERSE_EFFECTS_TOLERABILITY",
    "ACCESS_COST_SYSTEM_BARRIERS",
    "EFFICACY_EXPECTATION_MISMATCH",
    "PATIENT_PREFERENCE_BELIEFS",
    "PLANNED_GOAL_BASED_DISCONTINUATION",
    "CLINICAL_DECISION_MEDICAL_APPROPRIATENESS",
    "OTHER_UNKNOWN"
]

THRESHOLDS = [1.0, 0.95, 0.90, 0.85, 0.80]

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

KEYWORD_MAP = {
    "ADVERSE_EFFECTS_TOLERABILITY": ["side effect", "nausea", "vomiting", "gi ", "intolerable", "adverse", "injection site", "pancreatitis", "gallbladder", "diarrhea", "constipation", "abdominal pain", "tolerability", "poorly tolerated", "unable to tolerate", "severe nausea", "persistent vomiting", "gi symptoms"],
    "ACCESS_COST_SYSTEM_BARRIERS": ["insurance", "cost", "prior authorization", "coverage", "expensive", "afford", "financial", "formulary", "denied", "not covered", "out of pocket", "copay", "supply shortage", "unavailable", "pharmacy issue", "medication access", "cannot afford", "too expensive"],
    "EFFICACY_EXPECTATION_MISMATCH": ["not working", "plateau", "ineffective", "no improvement", "weight gain", "insufficient", "didn't work", "stopped working", "no longer effective", "weight loss plateau", "minimal effect", "lack of efficacy", "disappointing results", "not meeting expectations", "poor response"],
    "PATIENT_PREFERENCE_BELIEFS": ["preference", "natural", "fear", "concern", "unsafe", "don't want", "patient declined", "patient refused", "patient choice", "personal belief", "stigma", "misinformation", "safety concern", "worried about", "prefer lifestyle", "natural approach", "doesn't want medication"],
    "PLANNED_GOAL_BASED_DISCONTINUATION": ["goal", "target reached", "planned", "taper", "achieved", "reached goal", "goal weight", "planned discontinuation", "time-limited", "original plan", "as planned", "successful", "met target", "achieved weight loss", "glycemic goal reached", "planned stop"],
    "CLINICAL_DECISION_MEDICAL_APPROPRIATENESS": ["contraindication", "interaction", "pregnancy", "switch", "medical decision", "provider decision", "clinical judgment", "drug interaction", "comorbidity", "renal", "hepatic", "cardiac", "thyroid", "medullary thyroid", "men", "men2", "pancreatitis history", "guideline", "deprescribe", "switching therapy", "change medication", "medical reason"],
    "OTHER_UNKNOWN": ["unknown", "unclear", "lost", "ambiguous", "no reason", "lost to follow-up", "reason unclear", "unclear reason", "not documented", "patient stopped", "discontinued without explanation", "no clear reason", "ambiguous documentation"]
}


def load_test_data():
    test_data = []
    with open(TEST_DATA_PATH) as f:
        for line in f:
            test_data.append(json.loads(line.strip()))
    print(f"Loaded {len(test_data)} test samples")
    return test_data


def compute_metrics(y_true, y_pred):
    valid_mask = [p is not None for p in y_pred]
    y_true_valid = [t for t, v in zip(y_true, valid_mask) if v]
    y_pred_valid = [p for p, v in zip(y_pred, valid_mask) if v]

    if not y_true_valid:
        return None

    accuracy = accuracy_score(y_true_valid, y_pred_valid)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_valid, y_pred_valid, average='macro', labels=CATEGORIES, zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_valid, y_pred_valid, average='weighted', labels=CATEGORIES, zero_division=0
    )
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        y_true_valid, y_pred_valid, labels=CATEGORIES, zero_division=0
    )

    per_class = {}
    for i, cat in enumerate(CATEGORIES):
        per_class[cat] = {
            "precision": float(precision_per[i]),
            "recall": float(recall_per[i]),
            "f1": float(f1_per[i]),
            "support": int(support_per[i])
        }

    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=CATEGORIES)

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_true_valid, y_pred_valid, labels=CATEGORIES, zero_division=0),
    }


def evaluate_modernbert(test_data, threshold):
    label = str(threshold).replace(".", "_")
    model_path = MODELS_DIR / "modernbert_large" / f"threshold_{label}"
    if not model_path.exists():
        return None

    print(f"\n{'#'*70}")
    print(f"Evaluating ModernBERT-large @ threshold={threshold}")
    print(f"{'#'*70}")

    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding
    )
    from datasets import Dataset

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path), trust_remote_code=True
    )
    model = model.to("cuda")
    model.eval()

    test_texts = [d.get("clinical_note", "") for d in test_data]
    test_labels = [CATEGORIES.index(d.get("primary_reason", "OTHER_UNKNOWN")) if d.get("primary_reason") in CATEGORIES else CATEGORIES.index("OTHER_UNKNOWN") for d in test_data]

    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=False, max_length=512)

    test_dataset = test_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.remove_columns(["text"])
    test_dataset = test_dataset.rename_column("label", "labels")

    training_args = TrainingArguments(
        output_dir=str(model_path / "eval"),
        per_device_eval_batch_size=16,
        fp16=True,
        report_to="none",
    )

    data_collator = DataCollatorWithPadding(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=lambda p: {
            "accuracy": accuracy_score(p.label_ids, p.predictions.argmax(-1)),
            "f1_macro": precision_recall_fscore_support(p.label_ids, p.predictions.argmax(-1), average='macro', zero_division=0)[2],
        },
    )

    preds_output = trainer.predict(test_dataset)
    preds = preds_output.predictions.argmax(-1)
    y_true = [CATEGORIES[l] for l in test_labels]
    y_pred = [CATEGORIES[p] for p in preds]

    metrics = compute_metrics(y_true, y_pred)

    del model, trainer
    torch.cuda.empty_cache()

    return metrics


def evaluate_qwen_vllm(test_data, threshold):
    label = str(threshold).replace(".", "_")
    model_path = MODELS_DIR / "qwen3.5-9b" / f"threshold_{label}"
    if not model_path.exists():
        print(f"  Model not found: {model_path}, skipping")
        return None

    print(f"\n{'#'*70}")
    print(f"Evaluating Qwen3.5-9B @ threshold={threshold} with vLLM")
    print(f"{'#'*70}")

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=16384,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)

    tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")

    y_true = []
    y_pred = []

    for idx, sample in enumerate(test_data):
        if idx % 40 == 0:
            print(f"  Processing {idx+1}/{len(test_data)}...")

        note = sample.get("clinical_note", "")
        true_label = sample.get("primary_reason", "")

        messages = [{"role": "user", "content": INSTRUCTION + "\n\nClinical Note:\n" + note}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text=prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                use_cache=True,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        pred_label = parse_qwen_response(response)
        y_true.append(true_label)
        y_pred.append(pred_label)

    metrics = compute_metrics(y_true, y_pred)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return metrics


def parse_qwen_response(raw_output):
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
                    return cat
            return primary_reason if primary_reason in CATEGORIES else None
        return None
    except (json.JSONDecodeError, AttributeError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["modernbert", "qwen", "both"], default="both")
    parser.add_argument("--threshold", type=float, choices=THRESHOLDS, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 4: Ablation Study Evaluation")
    print("=" * 70)

    test_data = load_test_data()

    thresholds_to_run = [args.threshold] if args.threshold else THRESHOLDS

    all_results = {}

    if args.model in ["modernbert", "both"]:
        for threshold in thresholds_to_run:
            metrics = evaluate_modernbert(test_data, threshold)
            if metrics:
                all_results[f"modernbert_{threshold}"] = {
                    "threshold": threshold,
                    "model": "modernbert-large",
                    "metrics": {k: v for k, v in metrics.items() if k not in ["confusion_matrix", "classification_report", "per_class"]},
                    "per_class": metrics.get("per_class", {}),
                }

    if args.model in ["qwen", "both"]:
        for threshold in thresholds_to_run:
            metrics = evaluate_qwen_vllm(test_data, threshold)
            if metrics:
                all_results[f"qwen_{threshold}"] = {
                    "threshold": threshold,
                    "model": "qwen3.5-9b",
                    "metrics": {k: v for k, v in metrics.items() if k not in ["confusion_matrix", "classification_report", "per_class"]},
                    "per_class": metrics.get("per_class", {}),
                }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    eval_results_path = RESULTS_DIR / "evaluation_results.json"
    with open(eval_results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nEvaluation results saved to: {eval_results_path}")

    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    print(f"{'Model':<25} {'Threshold':<10} {'Accuracy':<10} {'Macro-F1':<10}")
    for key, val in all_results.items():
        m = val["metrics"]
        print(f"{val['model']:<25} {val['threshold']:<10} {m.get('accuracy', 0):<10.4f} {m.get('f1_macro', 0):<10.4f}")
    print("\nPhase 4 complete.")


if __name__ == "__main__":
    main()