#!/usr/bin/env python3
"""
Fine-tune BioClinical ModernBERT models for GLP-1 discontinuation classification.
Supports both base and large variants.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configuration
TRAIN_DATA_PATH = "/home/user/GLP_project/GLP_project_ad/code/finetune/training_data/train_test/train.jsonl"
TEST_DATA_PATH = "/home/user/GLP_project/GLP_project_ad/code/finetune/training_data/train_test/test.jsonl"
OUTPUT_DIR = "/home/user/GLP_project/GLP_project_ad/code/encoding_models/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CATEGORIES = [
    "ADVERSE_EFFECTS_TOLERABILITY",
    "ACCESS_COST_SYSTEM_BARRIERS",
    "EFFICACY_EXPECTATION_MISMATCH",
    "PATIENT_PREFERENCE_BELIEFS",
    "PLANNED_GOAL_BASED_DISCONTINUATION",
    "CLINICAL_DECISION_MEDICAL_APPROPRIATENESS",
    "OTHER_UNKNOWN"
]

MODEL_CONFIGS = {
    "base": "thomas-sounack/BioClinical-ModernBERT-base",
    "large": "thomas-sounack/BioClinical-ModernBERT-large",
}


def load_data(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            d = json.loads(line.strip())
            data.append(d)
    return data


def prepare_dataset(data, tokenizer, max_length=512):
    texts = [d.get("clinical_note", "") for d in data]
    labels = [CATEGORIES.index(d.get("primary_reason", "OTHER_UNKNOWN")) if d.get("primary_reason") in CATEGORIES else CATEGORIES.index("OTHER_UNKNOWN") for d in data]
    
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    encodings["labels"] = torch.tensor(labels)
    return encodings


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


def train_model(model_size):
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding
    )
    from datasets import Dataset
    
    print(f"\n{'='*80}")
    print(f"Fine-tuning BioClinical-ModernBERT-{model_size}")
    print(f"{'='*80}")
    
    model_name = MODEL_CONFIGS[model_size]
    output_path = os.path.join(OUTPUT_DIR, f"modernbert-{model_size}-finetuned")
    
    # Load tokenizer and model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(CATEGORIES),
        id2label={i: cat for i, cat in enumerate(CATEGORIES)},
        label2id={cat: i for i, cat in enumerate(CATEGORIES)}
    )
    
    # Load data
    print("Loading training data...")
    train_data = load_data(TRAIN_DATA_PATH)
    test_data = load_data(TEST_DATA_PATH)
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    
    # Prepare datasets
    train_texts = [d.get("clinical_note", "") for d in train_data]
    train_labels = [CATEGORIES.index(d.get("primary_reason", "OTHER_UNKNOWN")) if d.get("primary_reason") in CATEGORIES else CATEGORIES.index("OTHER_UNKNOWN") for d in train_data]
    
    test_texts = [d.get("clinical_note", "") for d in test_data]
    test_labels = [CATEGORIES.index(d.get("primary_reason", "OTHER_UNKNOWN")) if d.get("primary_reason") in CATEGORIES else CATEGORIES.index("OTHER_UNKNOWN") for d in test_data]
    
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=False, max_length=512)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    train_dataset = train_dataset.remove_columns(["text"])
    test_dataset = test_dataset.remove_columns(["text"])
    train_dataset = train_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        fp16=True,
        report_to="none",
    )
    
    # Trainer
    data_collator = DataCollatorWithPadding(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Evaluate
    print("Evaluating...")
    metrics = trainer.evaluate()
    
    # Save model
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Model saved to: {output_path}")
    
    # Save results
    results = {
        "model_name": f"modernbert-{model_size}",
        "model_type": "finetuned",
        "metrics": {k: v for k, v in metrics.items() if k not in ["confusion_matrix", "classification_report", "per_class"]},
        "per_class_metrics": metrics.get("eval_per_class", {}),
        "confusion_matrix": metrics.get("eval_confusion_matrix", []),
        "classification_report": metrics.get("eval_classification_report", ""),
    }
    
    results_path = os.path.join(OUTPUT_DIR, f"modernbert-{model_size}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BioClinical ModernBERT models")
    parser.add_argument("--model", type=str, choices=["base", "large", "both"], default="both",
                        help="Model size to fine-tune: base, large, or both (default: both)")
    args = parser.parse_args()
    
    import torch
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    
    if args.model == "both":
        for size in ["base", "large"]:
            train_model(size)
    else:
        train_model(args.model)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
