#!/usr/bin/env python3
"""
Phase 3a: ModernBERT-large Training for Ablation Study
Fine-tunes thomas-sounack/BioClinical-ModernBERT-large on each threshold dataset.
All hyperparameters held constant across runs.
"""

import os
import json
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset

DATASETS_DIR = Path("/home/user/GLP_project/GLP_project_ad/code/ablation_studies/datasets")
MODELS_DIR = Path("/home/user/GLP_project/GLP_project_ad/code/ablation_studies/models/modernbert_large")
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

MODEL_NAME = "thomas-sounack/BioClinical-ModernBERT-large"
THRESHOLDS = [1.0, 0.95, 0.90, 0.85, 0.80]

TRAIN_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "max_length": 512,
    "warmup_steps": 5,
    "logging_steps": 50,
}


def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro', labels=list(range(len(CATEGORIES))), zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', labels=list(range(len(CATEGORIES))), zero_division=0
    )
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(CATEGORIES))), zero_division=0
    )
    per_class = {}
    for i, cat in enumerate(CATEGORIES):
        per_class[cat] = {
            "precision": float(precision_per[i]),
            "recall": float(recall_per[i]),
            "f1": float(f1_per[i]),
            "support": int(support_per[i])
        }
    cm = confusion_matrix(labels, preds, labels=list(range(len(CATEGORIES))))
    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(labels, preds, labels=list(range(len(CATEGORIES))), target_names=CATEGORIES, zero_division=0)
    }


def train_at_threshold(threshold, train_data, test_data):
    label = str(threshold).replace(".", "_")
    output_path = MODELS_DIR / f"threshold_{label}"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Training ModernBERT-large @ threshold={threshold}")
    print(f"Train: {len(train_data)} samples | Output: {output_path}")
    print(f"{'='*70}")

    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CATEGORIES),
        id2label={i: cat for i, cat in enumerate(CATEGORIES)},
        label2id={cat: i for i, cat in enumerate(CATEGORIES)}
    )

    train_texts = [d.get("clinical_note", "") for d in train_data]
    train_labels = [CATEGORIES.index(d.get("primary_reason", "OTHER_UNKNOWN")) if d.get("primary_reason") in CATEGORIES else CATEGORIES.index("OTHER_UNKNOWN") for d in train_data]
    test_texts = [d.get("clinical_note", "") for d in test_data]
    test_labels = [CATEGORIES.index(d.get("primary_reason", "OTHER_UNKNOWN")) if d.get("primary_reason") in CATEGORIES else CATEGORIES.index("OTHER_UNKNOWN") for d in test_data]

    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=False, max_length=TRAIN_CONFIG["max_length"])

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    train_dataset = train_dataset.remove_columns(["text"])
    test_dataset = test_dataset.remove_columns(["text"])
    train_dataset = train_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=TRAIN_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAIN_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAIN_CONFIG["per_device_eval_batch_size"],
        learning_rate=TRAIN_CONFIG["learning_rate"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        logging_steps=TRAIN_CONFIG["logging_steps"],
        save_total_limit=2,
        fp16=True,
        report_to="none",
        warmup_steps=TRAIN_CONFIG["warmup_steps"],
        seed=3407,
    )

    data_collator = DataCollatorWithPadding(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating...")
    metrics = trainer.evaluate()
    elapsed = time.time() - start_time

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    results = {
        "threshold": threshold,
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "training_time_seconds": round(elapsed, 1),
        "metrics": {k: v for k, v in metrics.items() if k not in ["confusion_matrix", "classification_report", "per_class"]},
        "per_class_metrics": metrics.get("eval_per_class", {}),
        "confusion_matrix": metrics.get("eval_confusion_matrix", []),
        "classification_report": metrics.get("eval_classification_report", ""),
    }

    results_path = RESULTS_DIR / f"modernbert_large_threshold_{label}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Accuracy: {metrics['eval_accuracy']:.4f}")
    print(f"  Macro-F1: {metrics['eval_f1_macro']:.4f}")
    print(f"  Time: {elapsed:.1f}s | Results: {results_path}")

    del model, trainer
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, choices=THRESHOLDS, default=None,
                        help="Run a specific threshold. If not set, runs all thresholds.")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 3a: ModernBERT-large Training Sweep")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Config: {TRAIN_CONFIG}")

    test_data = load_jsonl(TEST_DATA_PATH)
    print(f"Test samples: {len(test_data)}\n")

    thresholds_to_run = [args.threshold] if args.threshold else THRESHOLDS
    all_results = {}

    for threshold in thresholds_to_run:
        label = str(threshold).replace(".", "_")
        train_path = DATASETS_DIR / f"train_{label}.jsonl"
        if not train_path.exists():
            print(f"WARNING: {train_path} not found, skipping threshold {threshold}")
            continue
        train_data = load_jsonl(train_path)
        results = train_at_threshold(threshold, train_data, test_data)
        all_results[threshold] = results

    summary_path = RESULTS_DIR / "modernbert_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to: {summary_path}")

    print("\n" + "=" * 70)
    print("Summary: ModernBERT-large Macro-F1 per Threshold")
    print("=" * 70)
    print(f"{'Threshold':<12} {'Train N':<10} {'Macro-F1':<10} {'Accuracy':<10} {'Time (s)':<10}")
    for threshold, res in all_results.items():
        m = res["metrics"]
        print(f"{threshold:<12} {res['train_samples']:<10} {m.get('eval_f1_macro', 0):<10.4f} {m.get('eval_accuracy', 0):<10.4f} {res['training_time_seconds']:<10}")
    print("\nPhase 3a complete.")


if __name__ == "__main__":
    main()