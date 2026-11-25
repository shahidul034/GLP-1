import json, re, os, collections
import numpy as np
from sklearn.cluster import KMeans

PROJECT = '/home/mshahidul/GLP_project/GLP_project_ad/experiments'
DATA_OUT = f'{PROJECT}/reviewer3_com1/data'
os.makedirs(DATA_OUT, exist_ok=True)

SECTION_HEADERS = [
    'HPI', 'SUBJECTIVE', 'OBJECTIVE', 'ASSESSMENT', 'PLAN', 'VITALS',
    'PHYSICAL EXAM', 'EXAM', 'MEDICATIONS', 'ALLERGIES', 'ROS',
    'REVIEW OF SYSTEMS', 'LABS', 'IMAGING', 'DISPOSITION',
    'FOLLOW UP', 'DATE', 'PROVIDER', 'LOCATION', 'AUTHOR',
    'CC', 'CHIEF COMPLAINT', 'PMH', 'PSH', 'FAMILY HISTORY', 'SOCIAL HISTORY',
    'CONSTITUTIONAL', 'HEENT', 'CV', 'RESP', 'LUNGS', 'GI', 'NEURO', 'PSYCH',
    'MUSCULOSKELETAL', 'SKIN', 'GENERAL', 'FEET', 'EXTREMITIES',
    'ACTIVE PROBLEMS', 'MEDICATION LIST', 'CURRENT MEDICATIONS',
    'ASSESSMENT AND PLAN', 'ASSESSMENT & PLAN', 'ASSESSMENT/PLAN',
    'PAST MEDICAL HISTORY', 'PAST SURGICAL HISTORY',
    'DX', 'DIAGNOSIS', 'IMPRESSION', 'RECOMMENDATIONS',
    'BMP', 'CBC', 'A1C', 'LIPID PANEL', 'THYROID', 'LIVER FUNCTION',
    'URINALYSIS', 'MICROALBUMIN', 'VITAMIN', 'HR', 'BP', 'TEMP', 'O2',
    'WT', 'HT', 'BMI', 'WEIGHT', 'HEIGHT',
    'DATE OF SERVICE', 'DATE OF VISIT', 'VISIT DATE',
    'TITLE', 'PATIENT', 'PRIMARY CARE', 'MEDICATION', 'MEDS', 'LAB',
    'PLAN OF CARE', 'TREATMENT PLAN', 'GOALS', 'PATIENT EDUCATION',
    'POINT OF CARE', 'SCREENING', 'IMMUNIZATIONS', 'PREVENTIVE CARE',
    'HISTORY OF PRESENT ILLNESS',
    'ABD', 'ABDOMEN', 'CAMPUS', 'CARDIAC', 'CLINIC', 'COUNSELING',
    'DOB', 'ENCOUNTER TYPE', 'IDENTIFICATION', 'INTERVAL HISTORY',
    'MENTAL STATUS', 'MSE', 'NECK', 'OTHER', 'PROBLEM LIST',
    'PTA', 'PULM', 'SERVICE', 'SEX', 'SIGNATURE', 'SSN', 'TIME',
    'VA ID', 'IDENTIFYING DATA',
    'LOCATION OF CARE', 'RECENT DATA', 'RECENT LABS',
    'HEALTH MAINTENANCE', 'HEALTH MAINT',
    'DM / MEDICATION MANAGEMENT',
    'SUBJECTIVE / INTERVAL HISTORY',
    'SUBJ HX', 'SUBJECTIVE/HPI',
    'HPI / INTERVAL HISTORY',
    'INTERVAL HX',
    'PATIENT EDUCATION / DISCUSSION',
    'PATIENT UNDERSTANDING',
    'IDENTIFICATION/CC',
    'WEIGHT TREND', 'MONITORING',
    'DISCHARGE/DISPOSITION', 'FALL RISK ASSESSMENT',
    'CURRENT DIABETES MEDS',
    'HTN/HLD', 'LIFESTYLE', 'MISC', 'ACTIVE MEDS',
    'ACTIVE PROBLEM LIST', 'LABORATORY DATA',
    'LABS/DIAGNOSTICS',
    'IDENTIFICATION / HISTORY', 'VITAL SIGNS',
    'DATE/TIME',
]

def get_section_vector(note):
    vec = {}
    for h in SECTION_HEADERS:
        if re.search(r'^' + re.escape(h) + r'\s*:', note, re.MULTILINE | re.IGNORECASE):
            vec[h] = 1
    return vec

data = [json.loads(line) for line in open(f'{PROJECT}/dataset/train.jsonl')]
notes = [d['clinical_note'] for d in data]
labels = [d['primary_reason'] for d in data]
note_ids = [d.get('note_id', f'train_{i}') for i, d in enumerate(data)]

X = np.array([[get_section_vector(n).get(h, 0) for h in SECTION_HEADERS] for n in notes])
print(f'Feature matrix: {X.shape}')

N_CLUSTERS = 8
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
clusters = km.fit_predict(X)

# Find best held-out clusters: must contain all 7 labels
# Try holding out 2 clusters
all_labels_set = set(labels)
best_held = None
best_train = None

for c1 in range(N_CLUSTERS):
    for c2 in range(c1 + 1, N_CLUSTERS):
        held_mask = (clusters == c1) | (clusters == c2)
        held_labels = set(np.array(labels)[held_mask])
        if held_labels == all_labels_set:
            best_held = (c1, c2)
            print(f'Held-out clusters: {c1}, {c2} (all 7 labels present in test)')
            break
    if best_held:
        break

if not best_held:
    # Fallback: hold out single clusters until all labels found
    for c in range(N_CLUSTERS):
        held_mask = clusters == c
        held_labels = set(np.array(labels)[held_mask])
        if held_labels == all_labels_set:
            best_held = (c,)
            print(f'Held-out single cluster: {c} (all 7 labels present)')
            break

if not best_held:
    # Pick 2 clusters with most label diversity, add safety samples from other clusters
    held_candidates = sorted(
        [(c, len(set(np.array(labels)[clusters == c]))) for c in range(N_CLUSTERS)],
        key=lambda x: -x[1]
    )
    best_held = (held_candidates[0][0], held_candidates[1][0])
    print(f'Using fallback held-out clusters: {best_held}')

held_clusters = set(best_held)
train_mask = np.array([c not in held_clusters for c in clusters])
held_mask = ~train_mask

# Check test label coverage
test_labels = set(np.array(labels)[held_mask])
missing = all_labels_set - test_labels
if missing:
    print(f'WARNING: Test missing labels: {missing}')
    # Move one sample per missing label from train to test
    train_indices = np.where(train_mask)[0]
    for label in missing:
        for idx in train_indices:
            if labels[idx] == label:
                held_mask[idx] = True
                train_mask[idx] = False
                print(f'  Moved {label} sample {idx} to test set')
                break

print(f'Train: {train_mask.sum()} notes, Test: {held_mask.sum()} notes')
print(f'Train labels: {sorted(set(np.array(labels)[train_mask]))}')
print(f'Test labels: {sorted(set(np.array(labels)[held_mask]))}')

# Save cluster assignments
cluster_info = []
for i in range(len(data)):
    cluster_info.append({
        'note_id': note_ids[i],
        'primary_reason': labels[i],
        'cluster': int(clusters[i]),
        'split': 'test' if held_mask[i] else 'train'
    })
with open(f'{DATA_OUT}/template_clusters.json', 'w') as f:
    json.dump(cluster_info, f, indent=2)
print(f'Saved cluster assignments to template_clusters.json')

# Print per-cluster label distribution
for c in range(N_CLUSTERS):
    mask = clusters == c
    dist = collections.Counter(np.array(labels)[mask])
    split = 'HELD' if c in held_clusters else 'train'
    print(f'  Cluster {c} [{split}]: {mask.sum():4d} notes, labels={dict(dist.most_common())}')

# Save raw test data for evaluation
test_raw = []
for i in np.where(held_mask)[0]:
    test_raw.append({'clinical_note': notes[i], 'primary_reason': labels[i], 'note_id': note_ids[i]})
with open(f'{DATA_OUT}/test_template_held_out.jsonl', 'w') as f:
    for item in test_raw:
        f.write(json.dumps(item) + '\n')
print(f'Saved test_template_held_out.jsonl ({len(test_raw)} notes)')

# Save raw train data
train_raw = []
for i in np.where(train_mask)[0]:
    train_raw.append({'clinical_note': notes[i], 'primary_reason': labels[i], 'note_id': note_ids[i]})

# Convert train to ShareGPT format for fine-tuning
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

train_sharegpt = []
for item in train_raw:
    train_sharegpt.append({
        "conversations": [
            {"from": "human", "value": INSTRUCTION + "\n\nClinical Note:\n" + item['clinical_note']},
            {"from": "gpt", "value": json.dumps({"primary_reason": item['primary_reason'], "confidence": "high", "supporting_evidence": "", "secondary_factors": []})}
        ]
    })

with open(f'{DATA_OUT}/train_template_held_out.jsonl', 'w') as f:
    for item in train_sharegpt:
        f.write(json.dumps(item) + '\n')
print(f'Saved train_template_held_out.jsonl ({len(train_sharegpt)} notes)')

print('\nTemplate-held-out split ready.')
