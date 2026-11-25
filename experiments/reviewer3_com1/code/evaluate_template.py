import os, sys, json, re, gc, torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

PROJECT = '/home/mshahidul/GLP_project/GLP_project_ad/experiments'
BASE = f'{PROJECT}/reviewer3_com1'
RESULTS_DIR = f'{BASE}/results'
os.makedirs(f'{RESULTS_DIR}/confusion_matrices', exist_ok=True)

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

MODEL_PATH = f'{BASE}/models/llama3.2-3b_template_held_out'
TEST_PATH = f'{BASE}/data/test_template_held_out.jsonl'
RESULTS_FILE = f'{RESULTS_DIR}/llama3.2-3b_template_held_out_results.json'

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

if __name__ == "__main__":
    if os.path.exists(RESULTS_FILE):
        print(f'Results already exist at {RESULTS_FILE}')
        sys.exit(0)

    print('Loading model...', flush=True)
    model, tokenizer = load_model()
    print('Model loaded.', flush=True)

    test_data = [json.loads(line) for line in open(TEST_PATH)]
    print(f'Test samples: {len(test_data)}', flush=True)

    y_true, y_pred, details = [], [], []
    for idx, sample in enumerate(test_data):
        if idx % 50 == 0:
            print(f'  [{idx}/{len(test_data)}]', flush=True)
        try:
            prompt = generate_prompt(tokenizer, sample['clinical_note'])
            raw = generate_response(model, tokenizer, prompt)
            pred = parse_output(raw)
            y_true.append(sample['primary_reason'])
            y_pred.append(pred)
            details.append({'true': sample['primary_reason'], 'pred': pred})
        except Exception as e:
            print(f'  Error at {idx}: {str(e)[:100]}', flush=True)
            y_true.append(sample['primary_reason'])
            y_pred.append(None)
            details.append({'true': sample['primary_reason'], 'pred': None, 'error': str(e)[:200]})

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    valid = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    if not valid:
        print('No valid predictions!')
        sys.exit(1)

    yt, yp = zip(*valid)
    acc = accuracy_score(yt, yp)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(yt, yp, labels=CATEGORIES, zero_division=0, average='macro')
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(yt, yp, labels=CATEGORIES, zero_division=0, average='weighted')
    cm = confusion_matrix(yt, yp, labels=CATEGORIES).tolist()
    p_p, r_p, f1_p, s_p = precision_recall_fscore_support(yt, yp, labels=CATEGORIES, zero_division=0)
    per_class = {}
    for i, cat in enumerate(CATEGORIES):
        per_class[cat] = {'precision': float(p_p[i]), 'recall': float(r_p[i]), 'f1': float(f1_p[i]), 'support': int(s_p[i])}

    results = {
        'condition': 'template_held_out',
        'accuracy': float(acc), 'f1_macro': float(f1_macro), 'f1_weighted': float(f1_w),
        'per_class': per_class, 'confusion_matrix': cm,
        'classification_report': classification_report(yt, yp, labels=CATEGORIES, zero_division=0),
        'valid_samples': len(valid), 'total_samples': len(y_true),
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    import pandas as pd
    cm_df = pd.DataFrame(cm, columns=CATEGORIES, index=CATEGORIES)
    cm_df.to_csv(f'{RESULTS_DIR}/confusion_matrices/llama3.2-3b_template_held_out_confusion_matrix.csv')

    print(f'\nTemplate Held-Out: acc={acc:.4f}, macro-F1={f1_macro:.4f}, w-F1={f1_w:.4f} ({len(valid)}/{len(y_true)})', flush=True)
