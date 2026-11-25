import json
import os

base = '/home/mshahidul/GLP_project/GLP_project_ad/data/data_structure'
out = '/home/mshahidul/GLP_project/GLP_project_ad/experiments/branch_held_out/data'

label_map = {
    'ADVERSE_EFFECTS_TOLERABILITY': 'AdverseEffects',
    'ADVERSE_EFFECTS_OR_TOLERABILITY': 'AdverseEffects',
    'ACCESS_COST_SYSTEM_BARRIERS': 'AccessCost',
    'EFFICACY_EXPECTATION_MISMATCH': 'EfficacyMismatch',
    'PATIENT_PREFERENCE_BELIEFS': 'PatientPreference',
    'PLANNED_GOAL_BASED_DISCONTINUATION': 'PlannedGoal',
    'CLINICAL_DECISION_MEDICAL_APPROPRIATENESS': 'ClinicalDecision',
    'OTHER_UNKNOWN': 'OtherUnknown',
}

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

def to_sharegpt(clinical_note, label):
    return {
        "conversations": [
            {"from": "human", "value": INSTRUCTION + "\n\nClinical Note:\n" + clinical_note},
            {"from": "gpt", "value": json.dumps({"primary_reason": label, "confidence": "high", "supporting_evidence": "", "secondary_factors": []})}
        ]
    }

# Load pooled accepted notes (1,635)
pooled = []
with open(f'{base}/3rd/oss.jsonl') as f:
    for line in f:
        d = json.loads(line)
        if d.get('overall_accept') == True:
            pooled.append({
                'clinical_note': d['clinical_note'],
                'primary_reason': label_map.get(d['primary_reason'], d['primary_reason'])
            })
print(f'Pooled accepted: {len(pooled)}')

# Load GPT-5.4 shakib accepted (692)
gpt54_shakib = []
with open(f'{base}/3rd/gpt5.4_shakib.jsonl') as f:
    for idx, line in enumerate(f):
        d = json.loads(line)
        if d.get('overall_accept') == True:
            # Get clinical_note from 2nd (matching by line index since both are ordered same)
            gpt54_shakib.append({'index': idx, 'primary_reason': label_map.get(d['primary_reason'], d['primary_reason'])})

# Load clinical notes from 2nd/shakib
with open(f'{base}/2nd/gpt5.4_shakib.jsonl') as f:
    for idx, line in enumerate(f):
        d = json.loads(line)
        for item in gpt54_shakib:
            if item['index'] == idx:
                item['clinical_note'] = d['clinical_note']
                break

gpt54_shakib = [item for item in gpt54_shakib if 'clinical_note' in item]
print(f'GPT-5.4 shakib accepted: {len(gpt54_shakib)}')

# Load GPT-5.4 adiba (72)
gpt54_adiba = []
with open(f'{base}/3rd/gpt5.4_adiba.jsonl') as f:
    for line in f:
        d = json.loads(line)
        gpt54_adiba.append({
            'clinical_note': d['clinical_note'],
            'primary_reason': label_map.get(d['primary_reason'], d['primary_reason'])
        })
print(f'GPT-5.4 adiba: {len(gpt54_adiba)}')

# Combine GPT-5.4
gpt54 = gpt54_shakib + gpt54_adiba
print(f'GPT-5.4 total: {len(gpt54)}')

# Write train/test files
# Experiment A: train on pooled, test on GPT-5.4
train_a = [to_sharegpt(d['clinical_note'], d['primary_reason']) for d in pooled]
test_a = [to_sharegpt(d['clinical_note'], d['primary_reason']) for d in gpt54]

# Experiment B: train on GPT-5.4, test on pooled
train_b = [to_sharegpt(d['clinical_note'], d['primary_reason']) for d in gpt54]
test_b = [to_sharegpt(d['clinical_note'], d['primary_reason']) for d in pooled]

for fname, data in [('train_pooled.jsonl', train_a), ('test_gpt54.jsonl', test_a),
                     ('train_gpt54.jsonl', train_b), ('test_pooled.jsonl', test_b)]:
    with open(os.path.join(out, fname), 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f'Wrote {fname}: {len(data)} examples')

# Also save raw test data for evaluation
for fname, data in [('test_gpt54_raw.jsonl', gpt54), ('test_pooled_raw.jsonl', pooled)]:
    with open(os.path.join(out, fname), 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f'Wrote {fname}: {len(data)} examples')
