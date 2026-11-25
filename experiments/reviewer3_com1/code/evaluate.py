#!/usr/bin/env python3
import os, sys, json, re, gc, torch
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

sys.path.insert(0, '/home/mshahidul/GLP_project/GLP_project_ad/experiments/branch_held_out')

CATEGORIES = [
    "AdverseEffects", "AccessCost", "EfficacyMismatch", "PatientPreference",
    "PlannedGoal", "ClinicalDecision", "OtherUnknown"
]

INSTRUCTION = """You are a clinical NLP specialist. Analyze the following clinical note and classify the primary reason for GLP-1 receptor agonist discontinuation.

Categories:
- AdverseEffects: GI side effects, systemic symptoms, serious adverse events, injection site reactions
- AccessCost: Insurance issues, high costs, prior authorization, supply shortages
- EfficacyMismatch: Weight plateau, insufficient results, medication not working as expected
- PatientPreference: Safety fears, misinformation, stigma, preference for natural approaches
- PlannedGoal: Reached goal weight, planned tapering, underlying condition improved
- ClinicalDecision: Contraindications, drug interactions, switching therapy, pregnancy
- OtherUnknown: Reason unclear, lost to follow-up, ambiguous documentation

Respond with a JSON object containing:
1. "primary_reason": The main discontinuation category
2. "confidence": Your confidence level (high/medium/low)
3. "supporting_evidence": Key phrases from the note supporting your classification
4. "secondary_factors": Any additional contributing factors (if applicable)"""

RESULTS_DIR = "/home/mshahidul/GLP_project/GLP_project_ad/experiments/branch_held_out/results"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/confusion_matrices", exist_ok=True)

def load_test_data(path):
    data = [json.loads(line) for line in open(path)]
    print(f"Loaded {len(data)} test samples")
    return data

QWEN_BASE = "unsloth/Qwen3.5-9B"
LLAMA_BASE = "unsloth/Llama-3.2-3B-Instruct"

def load_model(model_path, is_qwen=True):
    from unsloth import FastLanguageModel
    if is_qwen:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path, max_seq_length=4096, load_in_4bit=False
        )
    else:
        from unsloth.chat_templates import get_chat_template
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path, max_seq_length=4096, dtype=None, load_in_4bit=False
        )
        tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def generate_prompt(tokenizer, clinical_note, is_qwen=True):
    messages = [{"role": "user", "content": INSTRUCTION + "\n\nClinical Note:\n" + clinical_note}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def sanitize_note(text):
    text = re.sub(r'https?://\S+', '[URL]', text)
    text = re.sub(r'www\.\S+', '[URL]', text)
    return text

def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    prompt = sanitize_note(prompt)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=4096, truncation=True).to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.1,
                                 top_p=0.9, do_sample=True, use_cache=True)
    gen = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(gen, skip_special_tokens=True)
    return response.strip()

def parse_output(raw_output):
    try:
        cleaned = re.sub(r'```json\s*', '', raw_output)
        cleaned = re.sub(r'```\s*', '', cleaned).strip()
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            pred = result.get("primary_reason", "").strip()
            for cat in CATEGORIES:
                if cat.lower() in pred.lower() or pred.lower() in cat.lower():
                    return cat, result.get("confidence", "medium")
            return pred if pred in CATEGORIES else None, result.get("confidence", "medium")
    except:
        pass
    return None, "low"

def evaluate(config):
    name = config["name"]
    model_path = config["model_path"]
    test_path = config["test_path"]
    is_qwen = config.get("is_qwen", True)
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")
    
    result_file = f"{RESULTS_DIR}/{name}_results.json"
    if os.path.exists(result_file):
        print(f"  Results already exist, skipping")
        return
    
    model, tokenizer = load_model(model_path, is_qwen)
    test_data = load_test_data(test_path)
    
    y_true, y_pred, details = [], [], []
    
    for idx, sample in enumerate(test_data):
        note = sample["clinical_note"]
        true_label = sample["primary_reason"]
        
        if idx % 50 == 0:
            print(f"  [{idx}/{len(test_data)}]")
        
        try:
            prompt = generate_prompt(tokenizer, note, is_qwen)
            raw = generate_response(model, tokenizer, prompt)
            pred_label, _ = parse_output(raw)
            y_true.append(true_label)
            y_pred.append(pred_label)
            details.append({"true": true_label, "pred": pred_label, "raw": raw[:300]})
        except Exception as e:
            print(f"  Error at {idx}: {str(e)[:80]}")
            y_true.append(true_label)
            y_pred.append(None)
            details.append({"true": true_label, "pred": None, "error": str(e)[:200]})
    
    valid = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    if not valid:
        print("  No valid predictions!")
        return
    
    yt, yp = zip(*valid)
    acc = accuracy_score(yt, yp)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(yt, yp, labels=CATEGORIES, zero_division=0, average='macro')
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(yt, yp, labels=CATEGORIES, zero_division=0, average='weighted')
    cm = confusion_matrix(yt, yp, labels=CATEGORIES).tolist()
    per_class = {}
    p_per, r_per, f1_per, s_per = precision_recall_fscore_support(yt, yp, labels=CATEGORIES, zero_division=0)
    for i, cat in enumerate(CATEGORIES):
        per_class[cat] = {"precision": float(p_per[i]), "recall": float(r_per[i]), "f1": float(f1_per[i]), "support": int(s_per[i])}
    
    results = {
        "model": name,
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "per_class": per_class,
        "confusion_matrix": cm,
        "classification_report": classification_report(yt, yp, labels=CATEGORIES, zero_division=0),
        "valid_samples": len(valid),
        "total_samples": len(y_true),
    }
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {result_file}")
    
    import pandas as pd
    cm_df = pd.DataFrame(cm, columns=CATEGORIES, index=CATEGORIES)
    cm_df.to_csv(f"{RESULTS_DIR}/confusion_matrices/{name}_confusion_matrix.csv")
    
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Done: acc={acc:.4f}, macro-F1={f1_macro:.4f}, weighted-F1={f1_weighted:.4f}")

if __name__ == "__main__":
    base = "/home/mshahidul/GLP_project/GLP_project_ad/experiments/branch_held_out"
    data = f"{base}/data"
    models = f"{base}/models"
    
    configs = [
        {"name": "qwen3.5-9b_from_pooled_on_gpt54", "model_path": f"{models}/qwen3.5-9b_from_pooled",
         "test_path": f"{data}/test_gpt54_raw.jsonl", "is_qwen": True},
        {"name": "qwen3.5-9b_from_gpt54_on_pooled", "model_path": f"{models}/qwen3.5-9b_from_gpt54",
         "test_path": f"{data}/test_pooled_raw.jsonl", "is_qwen": True},
        {"name": "llama3.2-3b_from_pooled_on_gpt54", "model_path": f"{models}/llama3.2-3b_from_pooled",
         "test_path": f"{data}/test_gpt54_raw.jsonl", "is_qwen": False},
        {"name": "llama3.2-3b_from_gpt54_on_pooled", "model_path": f"{models}/llama3.2-3b_from_gpt54",
         "test_path": f"{data}/test_pooled_raw.jsonl", "is_qwen": False},
    ]
    
    for cfg in configs:
        if not os.path.exists(cfg["model_path"]):
            print(f"Skipping {cfg['name']} -- model not found at {cfg['model_path']}")
            continue
        evaluate(cfg)
