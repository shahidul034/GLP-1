#!/usr/bin/env python3
import os, sys, json, re, gc, torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

CATEGORIES = [
    'ADVERSE_EFFECTS_TOLERABILITY', 'ACCESS_COST_SYSTEM_BARRIERS',
    'EFFICACY_EXPECTATION_MISMATCH', 'PATIENT_PREFERENCE_BELIEFS',
    'PLANNED_GOAL_BASED_DISCONTINUATION', 'CLINICAL_DECISION_MEDICAL_APPROPRIATENESS',
    'OTHER_UNKNOWN'
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

MODEL_PATH = '/home/mshahidul/GLP_project/GLP_project_ad/experiments/mask_stress_test/models/llama3.2-3b_masked'
TEST_PATH = '/home/mshahidul/GLP_project/GLP_project_ad/experiments/mask_stress_test/data/test_masked.jsonl'
RESULT_FILE = '/home/mshahidul/GLP_project/GLP_project_ad/experiments/mask_stress_test/results/condition_C_masked_retrain.json'

print('Loading model...', flush=True)
model, tokenizer = FastLanguageModel.from_pretrained(MODEL_PATH, max_seq_length=4096, dtype=None, load_in_4bit=False)
tokenizer = get_chat_template(tokenizer, chat_template='llama-3.1')
FastLanguageModel.for_inference(model)
print('Model loaded.', flush=True)

test_data = [json.loads(line) for line in open(TEST_PATH)]
print(f'Test samples: {len(test_data)}', flush=True)

y_true, y_pred = [], []
for idx, sample in enumerate(test_data):
    if idx % 50 == 0:
        print(f'  [{idx}/{len(test_data)}]', flush=True)
    try:
        messages = [{'role': 'user', 'content': INSTRUCTION + '\n\nClinical Note:\n' + sample['clinical_note']}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, temperature=0.1, top_p=0.9, do_sample=True, use_cache=True)
        response = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        
        pred = None
        cleaned = re.sub(r'```json\s*', '', response)
        cleaned = re.sub(r'```\s*', '', cleaned).strip()
        jm = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if jm:
            result = json.loads(jm.group())
            p = result.get('primary_reason', '').upper().strip()
            for cat in CATEGORIES:
                if cat in p or p in cat:
                    pred = cat; break
            if pred is None and p in CATEGORIES:
                pred = p
        y_true.append(sample['primary_reason'])
        y_pred.append(pred)
    except Exception as e:
        print(f'  Error {idx}: {str(e)[:80]}', flush=True)
        y_true.append(sample['primary_reason'])
        y_pred.append(None)

del model, tokenizer; gc.collect(); torch.cuda.empty_cache()

valid = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
if not valid:
    print('No valid predictions!'); sys.exit(1)

yt, yp = zip(*valid)
acc = accuracy_score(yt, yp)
f1_macro = precision_recall_fscore_support(yt, yp, labels=CATEGORIES, zero_division=0, average='macro')[2]
f1_w = precision_recall_fscore_support(yt, yp, labels=CATEGORIES, zero_division=0, average='weighted')[2]
cm = confusion_matrix(yt, yp, labels=CATEGORIES).tolist()
p_p, r_p, f1_p, s_p = precision_recall_fscore_support(yt, yp, labels=CATEGORIES, zero_division=0)
per_class = {cat: {'f1': float(f1_p[i]), 'support': int(s_p[i])} for i, cat in enumerate(CATEGORIES)}

results = {
    'condition': 'C_masked_retrain',
    'accuracy': float(acc), 'f1_macro': float(f1_macro), 'f1_weighted': float(f1_w),
    'per_class': per_class, 'confusion_matrix': cm,
    'classification_report': classification_report(yt, yp, labels=CATEGORIES, zero_division=0),
    'valid_samples': len(valid), 'total_samples': len(y_true),
}

with open(RESULT_FILE, 'w') as f:
    json.dump(results, f, indent=2)

print(f'\nCondition C: acc={acc:.4f}, macro-F1={f1_macro:.4f}, w-F1={f1_w:.4f} ({len(valid)}/{len(y_true)})', flush=True)
