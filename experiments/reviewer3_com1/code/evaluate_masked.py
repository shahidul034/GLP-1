#!/usr/bin/env python3
import os, sys, json, re, gc, torch
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

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

MODEL_PATH = "/home/mshahidul/GLP_project/GLP_project_ad/code/finetune/models/llama3.2-3b-finetune"
RESULTS_DIR = "/home/mshahidul/GLP_project/GLP_project_ad/experiments/mask_stress_test/results"
os.makedirs(f"{RESULTS_DIR}/confusion_matrices", exist_ok=True)

def load_test_data(path):
    return [json.loads(line) for line in open(path)]

def load_model():
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH, max_seq_length=4096, dtype=None, load_in_4bit=False
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

def evaluate(test_data, save_name):
    result_file = f"{RESULTS_DIR}/{save_name}.json"
    if os.path.exists(result_file):
        print(f"  SKIP {save_name} — results exist")
        return json.load(open(result_file))

    print(f"  Loading model...")
    model, tokenizer = load_model()

    y_true, y_pred, details = [], [], []
    for idx, sample in enumerate(test_data):
        if idx % 50 == 0:
            print(f"  [{idx}/{len(test_data)}]")
        try:
            prompt = generate_prompt(tokenizer, sample["clinical_note"])
            raw = generate_response(model, tokenizer, prompt)
            pred = parse_output(raw)
            y_true.append(sample["primary_reason"])
            y_pred.append(pred)
            details.append({"true": sample["primary_reason"], "pred": pred})
        except Exception as e:
            print(f"  Error at {idx}: {str(e)[:100]}")
            y_true.append(sample["primary_reason"])
            y_pred.append(None)
            details.append({"true": sample["primary_reason"], "pred": None, "error": str(e)[:200]})

    del model, tokenizer; gc.collect(); torch.cuda.empty_cache()

    valid = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    if not valid:
        print("  No valid predictions!")
        return None

    yt, yp = zip(*valid)
    acc = accuracy_score(yt, yp)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(yt, yp, labels=CATEGORIES, zero_division=0, average='macro')
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(yt, yp, labels=CATEGORIES, zero_division=0, average='weighted')
    cm = confusion_matrix(yt, yp, labels=CATEGORIES).tolist()
    p_p, r_p, f1_p, s_p = precision_recall_fscore_support(yt, yp, labels=CATEGORIES, zero_division=0)
    per_class = {}
    for i, cat in enumerate(CATEGORIES):
        per_class[cat] = {"precision": float(p_p[i]), "recall": float(r_p[i]), "f1": float(f1_p[i]), "support": int(s_p[i])}

    results = {
        "condition": save_name,
        "accuracy": float(acc), "f1_macro": float(f1_macro), "f1_weighted": float(f1_w),
        "per_class": per_class, "confusion_matrix": cm,
        "classification_report": classification_report(yt, yp, labels=CATEGORIES, zero_division=0),
        "valid_samples": len(valid), "total_samples": len(y_true),
    }

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    pd = __import__('pandas')
    pd.DataFrame(cm, columns=CATEGORIES, index=CATEGORIES).to_csv(f"{RESULTS_DIR}/confusion_matrices/{save_name}_confusion_matrix.csv")

    print(f"  Done: acc={acc:.4f}, macro-F1={f1_macro:.4f}, w-F1={f1_w:.4f} ({len(valid)}/{len(y_true)})")
    return results

if __name__ == "__main__":
    data_dir = "/home/mshahidul/GLP_project/GLP_project_ad/experiments/mask_stress_test/data"

    print("=== Condition B: Original train → Masked test ===")
    test_masked = load_test_data(f"{data_dir}/test_masked.jsonl")
    res_b = evaluate(test_masked, "condition_B_masked_inference")

    print("\n=== Baseline (Condition A) — copy from existing ===")
    baseline_path = "/home/mshahidul/GLP_project/GLP_project_ad/code/finetune/testing_results/llama3.2-3b_finetuned_results.json"
    baseline = json.load(open(baseline_path))
    with open(f"{RESULTS_DIR}/condition_A_baseline.json", 'w') as f:
        json.dump(baseline, f, indent=2)
    print(f"  Baseline: acc={baseline['metrics']['accuracy']:.4f}, macro-F1={baseline['metrics']['f1_macro']:.4f}")
