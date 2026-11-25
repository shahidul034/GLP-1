#!/usr/bin/env python3
import os, sys, json, re, gc, torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from collections import Counter

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
os.makedirs(f"{RESULTS_DIR}/confusion_matrices", exist_ok=True)
QVEN_BASE = "unsloth/Qwen3.5-9B"

def load_test_data(path):
    return [json.loads(line) for line in open(path)]

def evaluate_model(model, tokenizer, test_data, model_name, save_name):
    result_file = f"{RESULTS_DIR}/{save_name}_results.json"
    if os.path.exists(result_file):
        print(f"Skipping {save_name} - results exist")
        return

    y_true, y_pred, details = [], [], []
    
    for idx, sample in enumerate(test_data):
        note = sample["clinical_note"]
        true_label = sample["primary_reason"]
        
        if idx % 50 == 0:
            print(f"  [{idx}/{len(test_data)}]")
        
        try:
            messages = [{"role": "user", "content": INSTRUCTION + "\n\nClinical Note:\n" + note}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", max_length=4096, truncation=True).to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1,
                                         top_p=0.9, do_sample=True, use_cache=True)
            gen = outputs[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(gen, skip_special_tokens=True)
            
            pred_label = None
            try:
                cleaned = re.sub(r'```json\s*', '', response)
                cleaned = re.sub(r'```\s*', '', cleaned).strip()
                json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    pred = result.get("primary_reason", "").strip()
                    for cat in CATEGORIES:
                        if cat.lower() in pred.lower() or pred.lower() in cat.lower():
                            pred_label = cat
                            break
                    if pred_label is None and pred in CATEGORIES:
                        pred_label = pred
            except:
                pass
            
            y_true.append(true_label)
            y_pred.append(pred_label)
            details.append({"true": true_label, "pred": pred_label})
        except Exception as e:
            print(f"  Error {idx}: {str(e)[:100]}")
            y_true.append(true_label)
            y_pred.append(None)
            details.append({"true": true_label, "pred": None, "error": str(e)[:200]})
    
    valid = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    if not valid:
        print(f"  No valid predictions for {save_name}!")
        return
    
    yt, yp = zip(*valid)
    acc = accuracy_score(yt, yp)
    f1_macro = precision_recall_fscore_support(yt, yp, labels=CATEGORIES, zero_division=0, average='macro')[2]
    f1_weighted = precision_recall_fscore_support(yt, yp, labels=CATEGORIES, zero_division=0, average='weighted')[2]
    cm = confusion_matrix(yt, yp, labels=CATEGORIES).tolist()
    p_per, r_per, f1_per, s_per = precision_recall_fscore_support(yt, yp, labels=CATEGORIES, zero_division=0)
    per_class = {cat: {"f1": float(f1_per[i]), "support": int(s_per[i])} for i, cat in enumerate(CATEGORIES)}
    
    results = {
        "model": model_name, "accuracy": float(acc), "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted), "per_class": per_class, "confusion_matrix": cm,
        "classification_report": classification_report(yt, yp, labels=CATEGORIES, zero_division=0),
        "valid_samples": len(valid), "total_samples": len(y_true),
    }
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    import pandas as pd
    pd.DataFrame(cm, columns=CATEGORIES, index=CATEGORIES).to_csv(f"{RESULTS_DIR}/confusion_matrices/{save_name}_confusion_matrix.csv")
    
    print(f"  Done: {model_name}: acc={acc:.4f}, macro-F1={f1_macro:.4f}, weighted-F1={f1_weighted:.4f} ({len(valid)}/{len(y_true)})")

if __name__ == "__main__":
    base = "/home/mshahidul/GLP_project/GLP_project_ad/experiments/branch_held_out"
    data = f"{base}/data"
    models = f"{base}/models"
    
    from transformers import AutoConfig
    from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    
    # Load config and strip vision to avoid multimodal init
    print("Loading config (text-only mode)...")
    config = AutoConfig.from_pretrained(f"{models}/qwen3.5-9b_from_pooled", trust_remote_code=True)
    config.vision_config = None
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(QVEN_BASE, trust_remote_code=True)
    
    # Eval 1: finetuned from pooled, test on gpt54
    print("\n=== qwen3.5-9b_from_pooled_on_gpt54 ===")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        f"{models}/qwen3.5-9b_from_pooled",
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    test_data = load_test_data(f"{data}/test_gpt54_raw.jsonl")
    evaluate_model(model, tokenizer, test_data, "qwen3.5-9b_from_pooled_on_gpt54", "qwen3.5-9b_from_pooled_on_gpt54")
    del model; gc.collect(); torch.cuda.empty_cache()
    
    # Eval 2: finetuned from gpt54, test on pooled
    print("\n=== qwen3.5-9b_from_gpt54_on_pooled ===")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        f"{models}/qwen3.5-9b_from_gpt54",
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    test_data = load_test_data(f"{data}/test_pooled_raw.jsonl")
    evaluate_model(model, tokenizer, test_data, "qwen3.5-9b_from_gpt54_on_pooled", "qwen3.5-9b_from_gpt54_on_pooled")
    del model; gc.collect(); torch.cuda.empty_cache()
    
    print("\nAll Qwen evaluations complete!")
