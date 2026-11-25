#!/usr/bin/env python3
"""Evaluate Qwen3.5-9B finetuned models using vLLM for inference."""
import os, json, re, gc
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

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

def load_test_data(path):
    return [json.loads(line) for line in open(path)]

def format_prompt(clinical_note):
    return INSTRUCTION + "\n\nClinical Note:\n" + clinical_note

def parse_response(response):
    try:
        cleaned = re.sub(r'```json\s*', '', response)
        cleaned = re.sub(r'```\s*', '', cleaned).strip()
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            pred = result.get("primary_reason", "").strip()
            for cat in CATEGORIES:
                if cat.lower() in pred.lower() or pred.lower() in cat.lower():
                    return cat
            if pred in CATEGORIES:
                return pred
    except:
        pass
    return None

def evaluate_with_vllm(model_path, test_data, model_name, save_name, sampling_params=None):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    result_file = f"{RESULTS_DIR}/{save_name}_results.json"
    if os.path.exists(result_file):
        print(f"Skipping {save_name} - results exist")
        return

    if sampling_params is None:
        sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=256)

    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen3.5-9B", trust_remote_code=True)
    chat_template = tokenizer.get_chat_template()
    
    print(f"Loading model {model_path} with vLLM...")
    llm = LLM(model=model_path, max_model_len=8192, gpu_memory_utilization=0.9, trust_remote_code=True)
    
    # Format prompts
    prompts = []
    for sample in test_data:
        messages = [{"role": "user", "content": format_prompt(sample["clinical_note"])}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    
    print(f"Generating on {len(prompts)} samples...")
    outputs = llm.generate(prompts, sampling_params)
    
    y_true, y_pred = [], []
    for idx, (sample, output) in enumerate(zip(test_data, outputs)):
        response = output.outputs[0].text
        pred_label = parse_response(response)
        y_true.append(sample["primary_reason"])
        y_pred.append(pred_label)
        if idx % 100 == 0:
            print(f"  [{idx}/{len(test_data)}]")
    
    valid = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    if not valid:
        print(f"  No valid predictions!")
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
    
    print(f"  Done: acc={acc:.4f}, macro-F1={f1_macro:.4f}, weighted-F1={f1_weighted:.4f} ({len(valid)}/{len(y_true)})")
    
    del llm
    gc.collect()

if __name__ == "__main__":
    base = "/home/mshahidul/GLP_project/GLP_project_ad/experiments/branch_held_out"
    models = f"{base}/models"
    data = f"{base}/data"
    
    sp = __import__('vllm', fromlist=['SamplingParams']).SamplingParams(temperature=0.1, top_p=0.9, max_tokens=256)
    
    # Eval 1
    evaluate_with_vllm(
        f"{models}/qwen3.5-9b_from_pooled",
        load_test_data(f"{data}/test_gpt54_raw.jsonl"),
        "qwen3.5-9b_from_pooled_on_gpt54", "qwen3.5-9b_from_pooled_on_gpt54",
        sp
    )
    
    # Eval 2
    evaluate_with_vllm(
        f"{models}/qwen3.5-9b_from_gpt54",
        load_test_data(f"{data}/test_pooled_raw.jsonl"),
        "qwen3.5-9b_from_gpt54_on_pooled", "qwen3.5-9b_from_gpt54_on_pooled",
        sp
    )
    
    print("\nAll Qwen evaluations complete!")
