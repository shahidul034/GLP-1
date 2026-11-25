#!/usr/bin/env python3
"""
Evaluate base (pre-trained) BioClinical ModernBERT models without fine-tuning.
Uses zero-shot classification via embedding similarity.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configuration
TEST_DATA_PATH = "/home/user/GLP_project/GLP_project_ad/code/finetune/training_data/train_test/test.jsonl"
RESULTS_DIR = "/home/user/GLP_project/GLP_project_ad/code/encoding_models/results"
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

# Category descriptions for zero-shot classification
CATEGORY_DESCRIPTIONS = {
    "ADVERSE_EFFECTS_TOLERABILITY": "patient experienced side effects, adverse reactions, GI intolerance, nausea, vomiting, diarrhea from medication",
    "ACCESS_COST_SYSTEM_BARRIERS": "patient unable to access medication due to cost, insurance denial, prior authorization, supply shortage, backorder",
    "EFFICACY_EXPECTATION_MISMATCH": "medication not working as expected, weight plateau, insufficient results, stopped working",
    "PATIENT_PREFERENCE_BELIEFS": "patient preference, safety fears, misinformation, stigma, wants natural approach, concerned about dependency",
    "PLANNED_GOAL_BASED_DISCONTINUATION": "patient achieved goal weight, planned discontinuation, target reached, goal met",
    "CLINICAL_DECISION_MEDICAL_APPROPRIATENESS": "clinician decision, contraindication, medical appropriateness, guideline-based, drug interaction, pregnancy",
    "OTHER_UNKNOWN": "reason unclear, unknown, lost to follow-up, ambiguous documentation, patient stopped without clear reason"
}

MODEL_CONFIGS = {
    "base": "thomas-sounack/BioClinical-ModernBERT-base",
    "large": "thomas-sounack/BioClinical-ModernBERT-large",
}


def load_test_data():
    test_data = []
    with open(TEST_DATA_PATH, 'r') as f:
        for line in f:
            test_data.append(json.loads(line.strip()))
    print(f"Loaded {len(test_data)} test samples")
    return test_data


def get_embeddings(model, tokenizer, texts, batch_size=32):
    """Get [CLS] embeddings for a list of texts."""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Get [CLS] token embeddings (first token)
            embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)


def cosine_similarity(a, b):
    """Compute cosine similarity between vectors."""
    norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)
    return (a @ b.T) / (norm_a * norm_b.T)


def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics."""
    results = {}
    
    results["accuracy"] = accuracy_score(y_true, y_pred)
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', labels=list(range(len(CATEGORIES))), zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', labels=list(range(len(CATEGORIES))), zero_division=0
    )
    
    results["precision_macro"] = precision_macro
    results["recall_macro"] = recall_macro
    results["f1_macro"] = f1_macro
    results["precision_weighted"] = precision_weighted
    results["recall_weighted"] = recall_weighted
    results["f1_weighted"] = f1_weighted
    
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CATEGORIES))), zero_division=0
    )
    
    per_class = {}
    for i, cat in enumerate(CATEGORIES):
        per_class[cat] = {
            "precision": float(precision_per[i]),
            "recall": float(recall_per[i]),
            "f1": float(f1_per[i]),
            "support": int(support_per[i])
        }
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CATEGORIES))))
    
    results["per_class"] = per_class
    results["confusion_matrix"] = cm.tolist()
    results["classification_report"] = classification_report(
        y_true, y_pred, labels=list(range(len(CATEGORIES))), target_names=CATEGORIES, zero_division=0
    )
    
    return results


def evaluate_base_model(model_size):
    """Evaluate base model using zero-shot embedding similarity."""
    print(f"\n{'='*80}")
    print(f"Evaluating BioClinical-ModernBERT-{model_size} (base, zero-shot)")
    print(f"{'='*80}")
    
    model_name = MODEL_CONFIGS[model_size]
    
    # Load model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to("cuda")
    model.eval()
    
    # Load test data
    test_data = load_test_data()
    
    # Get category description embeddings
    print("Computing category description embeddings...")
    category_texts = [CATEGORY_DESCRIPTIONS[cat] for cat in CATEGORIES]
    category_embeddings = get_embeddings(model, tokenizer, category_texts)
    
    # Get test note embeddings
    print("Computing test note embeddings...")
    test_texts = [d.get("clinical_note", "") for d in test_data]
    test_embeddings = get_embeddings(model, tokenizer, test_texts)
    
    # Compute similarity and predict
    print("Computing similarities and making predictions...")
    similarities = cosine_similarity(test_embeddings, category_embeddings)
    predictions = np.argmax(similarities, axis=1)
    
    # Get true labels
    y_true = [CATEGORIES.index(d.get("primary_reason", "OTHER_UNKNOWN")) if d.get("primary_reason") in CATEGORIES else CATEGORIES.index("OTHER_UNKNOWN") for d in test_data]
    y_pred = predictions.tolist()
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(y_true, y_pred)
    
    # Save results
    results = {
        "model_name": f"modernbert-{model_size}",
        "model_type": "base_zero_shot",
        "metrics": {k: v for k, v in metrics.items() if k not in ["confusion_matrix", "classification_report", "per_class"]},
        "per_class_metrics": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report"],
    }
    
    results_path = os.path.join(RESULTS_DIR, f"modernbert-{model_size}_base_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    # Print summary
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"F1-macro: {metrics['f1_macro']:.4f}")
    print(f"F1-weighted: {metrics['f1_weighted']:.4f}")
    
    return metrics


def main():
    print("="*80)
    print("BioClinical ModernBERT Base Model Evaluation (Zero-Shot)")
    print("="*80)
    
    for size in ["base", "large"]:
        try:
            evaluate_base_model(size)
        except Exception as e:
            print(f"ERROR evaluating {size}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
