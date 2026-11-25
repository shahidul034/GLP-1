#!/usr/bin/env python3
"""
evaluate_gender_swap.py

Run all gender-swap evaluation conditions.

Conditions:
  A: Original paper model on original test (baseline, from existing results)
  B: Original paper model on gender-swapped test
  C: Gender-swapped model on gender-swapped test
  D: Augmented model on original test + gender-swapped test

Usage:
    CUDA_VISIBLE_DEVICES=7 python evaluate_gender_swap.py
"""

import os, sys, json, re, gc, torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

BASE_DIR = "/home/mshahidul/GLP_project/GLP_project_ad/experiments/gender_swap_stress_test"
RESULTS_DIR = f"{BASE_DIR}/results"
DATA_DIR = f"{BASE_DIR}/data"
os.makedirs(f"{RESULTS_DIR}/confusion_matrices", exist_ok=True)

ORIGINAL_MODEL_PATH = "/home/mshahidul/GLP_project/GLP_project_ad/code/finetune/models/llama3.2-3b-finetune"
ORIGINAL_TEST_PATH = "/home/mshahidul/GLP_project/GLP_project_ad/experiments/dataset/test.jsonl"
SWAPPED_TEST_PATH = f"{DATA_DIR}/test_gender_swapped.jsonl"
BASELINE_PATH = "/home/mshahidul/GLP_project/GLP_project_ad/code/finetune/testing_results/llama3.2-3b_finetuned_results.json"

SWAPPED_MODEL_PATH = f"{BASE_DIR}/models/llama3.2-3b_gender_swapped"
AUGMENTED_MODEL_PATH = f"{BASE_DIR}/models/llama3.2-3b_gender_augmented"

CATEGORIES = [
    "ADVERSE_EFFECTS_TOLERABILITY", "ACCESS_COST_SYSTEM_BARRIERS",
    "EFFICACY_EXPECTATION_MISMATCH", "PATIENT_PREFERENCE_BELIEFS",
    "PLANNED_GOAL_BASED_DISCONTINUATION", "CLINICAL_DECISION_MEDICAL_APPROPRIATENESS",
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


def load_test_data(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def load_model(model_path):
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path, max_seq_length=4096, dtype=None, load_in_4bit=False
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_prompt(tokenizer, clinical_note):
    messages = [{"role": "user", "content": INSTRUCTION + "\n\nClinical Note:\n" + clinical_note}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.1,
                                 top_p=0.9, do_sample=True, use_cache=True)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def parse_output(raw_output):
    try:
        cleaned = re.sub(r'```json\s*', '', raw_output)
        cleaned = re.sub(r'```\s*', '', cleaned).strip()
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            pred = result.get("primary_reason", "").upper().strip()
            for cat in CATEGORIES:
                if cat in pred or pred in cat:
                    return cat
            return pred if pred in CATEGORIES else None
    except:
        pass
    return None


def compute_metrics(y_true, y_pred, categories=CATEGORIES):
    valid = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    if not valid:
        return None
    yt, yp = zip(*valid)
    acc = accuracy_score(yt, yp)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(yt, yp, labels=categories, zero_division=0, average='macro')
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(yt, yp, labels=categories, zero_division=0, average='weighted')
    cm = confusion_matrix(yt, yp, labels=categories).tolist()
    p_p, r_p, f1_p, s_p = precision_recall_fscore_support(yt, yp, labels=categories, zero_division=0)
    per_class = {}
    for i, cat in enumerate(categories):
        per_class[cat] = {"precision": float(p_p[i]), "recall": float(r_p[i]), "f1": float(f1_p[i]), "support": int(s_p[i])}
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_w),
        "per_class": per_class,
        "confusion_matrix": cm,
        "classification_report": classification_report(yt, yp, labels=categories, zero_division=0),
        "valid_samples": len(valid),
        "total_samples": len(y_true),
    }


def save_results(results, result_path, confusion_path):
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    cm = results["confusion_matrix"]
    import pandas as pd
    pd.DataFrame(cm, columns=CATEGORIES, index=CATEGORIES).to_csv(confusion_path)
    print(f"  Saved: {result_path}", flush=True)


def evaluate_condition(name, model_path, test_path, add_gender_subgroups=False):
    result_file = f"{RESULTS_DIR}/{name}.json"
    if os.path.exists(result_file):
        print(f"  SKIP {name} — results exist", flush=True)
        return json.load(open(result_file))

    print(f"  Loading model from {model_path}...", flush=True)
    model, tokenizer = load_model(model_path)
    test_data = load_test_data(test_path)
    print(f"  Evaluating {len(test_data)} test samples...", flush=True)

    y_true, y_pred = [], []
    for idx, sample in enumerate(test_data):
        if idx % 50 == 0:
            print(f"  [{idx}/{len(test_data)}]", flush=True)
        try:
            prompt = generate_prompt(tokenizer, sample["clinical_note"])
            raw = generate_response(model, tokenizer, prompt)
            pred = parse_output(raw)
            y_true.append(sample["primary_reason"])
            y_pred.append(pred)
        except Exception as e:
            print(f"  Error at {idx}: {str(e)[:100]}", flush=True)
            y_true.append(sample["primary_reason"])
            y_pred.append(None)

    del model, tokenizer; gc.collect(); torch.cuda.empty_cache()

    metrics = compute_metrics(y_true, y_pred)
    if metrics is None:
        print("  No valid predictions!", flush=True)
        return None

    results = {"condition": name, **metrics}

    # Gender subgroup analysis
    if add_gender_subgroups:
        from collections import defaultdict
        gender_groups = defaultdict(lambda: {"y_true": [], "y_pred": []})
        for idx, sample in enumerate(test_data):
            g = sample.get("_detected_gender", "unknown")
            gender_groups[g]["y_true"].append(y_true[idx])
            gender_groups[g]["y_pred"].append(y_pred[idx])
        subgroups = {}
        for g, data in gender_groups.items():
            sub_metrics = compute_metrics(data["y_true"], data["y_pred"])
            if sub_metrics:
                subgroups[g] = sub_metrics
        results["gender_subgroups"] = subgroups

    save_results(results, result_file, f"{RESULTS_DIR}/confusion_matrices/{name}_confusion_matrix.csv")
    print(f"  Done: acc={metrics['accuracy']:.4f}, macro-F1={metrics['f1_macro']:.4f}, w-F1={metrics['f1_weighted']:.4f} ({metrics['valid_samples']}/{metrics['total_samples']})", flush=True)
    return results


def run_all():
    print("="*60, flush=True)
    print("Gender-Swap Stress Test Evaluation", flush=True)
    print("="*60, flush=True)

    # Condition A: Baseline (use existing results)
    print("\n--- Condition A: Original baseline ---", flush=True)
    if os.path.exists(BASELINE_PATH):
        baseline = json.load(open(BASELINE_PATH))
        a_results = {
            "condition": "condition_A_original",
            "accuracy": baseline["metrics"]["accuracy"],
            "f1_macro": baseline["metrics"]["f1_macro"],
            "f1_weighted": baseline["metrics"]["f1_weighted"],
            "per_class": baseline["per_class_metrics"],
            "confusion_matrix": baseline["confusion_matrix"],
            "classification_report": "",
            "valid_samples": 320,
            "total_samples": 320,
        }
        save_results(a_results, f"{RESULTS_DIR}/condition_A_original.json",
                     f"{RESULTS_DIR}/confusion_matrices/condition_A_original_confusion_matrix.csv")
        print(f"  Baseline: acc={a_results['accuracy']:.4f}, macro-F1={a_results['f1_macro']:.4f}", flush=True)
    else:
        print(f"  WARN: No baseline results at {BASELINE_PATH}", flush=True)

    # Condition B: Original model on gender-swapped test
    print("\n--- Condition B: Original model → Gender-swapped test ---", flush=True)
    if os.path.exists(ORIGINAL_MODEL_PATH) and os.path.exists(SWAPPED_TEST_PATH):
        evaluate_condition("condition_B_gender_swapped_inference", ORIGINAL_MODEL_PATH, SWAPPED_TEST_PATH)
    else:
        print("  SKIP: model or test data not found", flush=True)

    # Condition C: Gender-swapped model on gender-swapped test
    print("\n--- Condition C: Gender-swapped model → Gender-swapped test ---", flush=True)
    if os.path.exists(SWAPPED_MODEL_PATH) and os.path.exists(SWAPPED_TEST_PATH):
        evaluate_condition("condition_C_gender_swapped_retrain", SWAPPED_MODEL_PATH, SWAPPED_TEST_PATH)
    else:
        print("  SKIP: model or test data not found", flush=True)

    # Condition D: Augmented model on original + swapped test
    print("\n--- Condition D: Augmented model → Original test ---", flush=True)
    if os.path.exists(AUGMENTED_MODEL_PATH) and os.path.exists(ORIGINAL_TEST_PATH):
        evaluate_condition("condition_D_augmented_on_original", AUGMENTED_MODEL_PATH, ORIGINAL_TEST_PATH)
    else:
        print("  SKIP: model or test data not found", flush=True)

    print("\n--- Condition D: Augmented model → Gender-swapped test ---", flush=True)
    if os.path.exists(AUGMENTED_MODEL_PATH) and os.path.exists(SWAPPED_TEST_PATH):
        evaluate_condition("condition_D_augmented_on_swapped", AUGMENTED_MODEL_PATH, SWAPPED_TEST_PATH)
    else:
        print("  SKIP: model or test data not found", flush=True)

    print("\nAll evaluations complete!", flush=True)


if __name__ == "__main__":
    run_all()
