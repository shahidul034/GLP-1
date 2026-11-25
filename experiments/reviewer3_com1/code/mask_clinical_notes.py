import re
import json

# ---- Section headers ----
SECTION_HEADERS = [
    'HPI', 'SUBJECTIVE', 'OBJECTIVE', 'ASSESSMENT', 'PLAN', 'VITALS',
    'PHYSICAL EXAM', 'EXAM', 'MEDICATIONS', 'ALLERGIES', 'ROS',
    'REVIEW OF SYSTEMS', 'LABS', 'IMAGING', 'PROCEDURE', 'DISPOSITION',
    'FOLLOW UP', 'DATE', 'PROVIDER', 'LOCATION', 'AUTHOR',
    'CC', 'CHIEF COMPLAINT', 'PMH', 'PSH', 'FAMILY HISTORY', 'SOCIAL HISTORY',
    'CONSTITUTIONAL', 'HEENT', 'CV', 'RESP', 'LUNGS', 'GI', 'NEURO', 'PSYCH',
    'MUSCULOSKELETAL', 'SKIN', 'GENERAL', 'FEET', 'EXTREMITIES',
    'ACTIVE PROBLEMS', 'MEDICATION LIST', 'CURRENT MEDICATIONS',
    'ASSESSMENT AND PLAN', 'ASSESSMENT & PLAN', 'ASSESSMENT/PLAN',
    'PAST MEDICAL HISTORY', 'PAST SURGICAL HISTORY', 'SOCIAL HISTORY',
    'FAMILY HISTORY', 'REVIEW OF SYSTEMS', 'PHYSICAL EXAMINATION',
    'DX', 'DIAGNOSIS', 'IMPRESSION', 'RECOMMENDATIONS',
    'BMP', 'CBC', 'A1C', 'LIPID PANEL', 'THYROID', 'LIVER FUNCTION',
    'URINALYSIS', 'MICROALBUMIN', 'VITAMIN', 'HR', 'BP', 'TEMP', 'O2',
    'WT', 'HT', 'BMI', 'WEIGHT', 'HEIGHT',
    'DATE OF SERVICE', 'DATE OF VISIT', 'VISIT DATE',
    'TITLE', 'PATIENT', 'PRIMARY CARE', 'MEDICATION', 'MEDS', 'LAB',
    'PLAN OF CARE', 'TREATMENT PLAN', 'GOALS', 'PATIENT EDUCATION',
    'POINT OF CARE', 'SCREENING', 'IMMUNIZATIONS', 'PREVENTIVE CARE',
    'CHIEF COMPLAINT', 'PRESENTING PROBLEM', 'PRESENTING COMPLAINT',
    'HISTORY OF PRESENT ILLNESS', 'HISTORY PRESENT ILLNESS',
]

# ---- Medication names ----
MEDICATION_NAMES = [
    'semaglutide', 'Ozempic', 'Wegovy', 'liraglutide', 'Saxenda', 'Victoza',
    'dulaglutide', 'Trulicity', 'tirzepatide', 'Mounjaro', 'Zepbound',
    'exenatide', 'Byetta', 'Bydureon', 'orforglipron', 'retatrutide',
    'metformin', 'GLP-1', 'GLP-1RA', 'GLP-1 receptor agonist',
    'GLP1', 'GLP-1RAs', 'anti-obesity medication', 'anti-obesity',
    'weight-loss injection', 'weight loss injection', 'weight-loss drug',
    'glargine', 'Lantus', 'Tresiba', 'degludec', 'insulin',
    'pioglitazone', 'ACTOS', 'empagliflozin', 'Jardiance',
    'dapagliflozin', 'Farxiga', 'canagliflozin', 'Invokana',
    'SGLT2', 'SGLT2i', 'SGLT-2', 'sulfonylurea', 'glipizide',
    'Glyburide', 'glimepiride', 'Amaryl', 'repaglinide', 'Prandin',
    'acarbose', 'Precose', 'pramlintide', 'Symlin',
    'liraglutide', 'Saxenda', 'Victoza',
]

def mask_section_headers(text):
    """Replace section headers like 'HPI:' or 'ASSESSMENT:' with '[SECTION]'."""
    for header in sorted(SECTION_HEADERS, key=len, reverse=True):
        pattern = re.compile(r'^' + re.escape(header) + r'\s*:', re.MULTILINE | re.IGNORECASE)
        text = pattern.sub('[SECTION]', text)
        pattern = re.compile(r'^' + re.escape(header) + r'\s*\n', re.MULTILINE | re.IGNORECASE)
        text = pattern.sub('[SECTION]\n', text)
    return text

def mask_medications(text):
    """Replace GLP-1 and related medication names with '[MEDICATION]'."""
    for med in sorted(MEDICATION_NAMES, key=len, reverse=True):
        text = re.sub(re.escape(med), '[MEDICATION]', text, flags=re.IGNORECASE)
    return text

def mask_label_phrases(text):
    """Replace strongly label-associated phrases with '[REASON_PHRASE]'."""
    # All phrases in one pass to avoid order dependence
    phrases = [
        # AdverseEffects
        'GI intolerance', 'GI upset', 'gastrointestinal',
        'abdominal pain', 'injection site reaction', 'injection site',
        'nausea', 'vomiting', 'diarrhea', 'pancreatitis', 'constipation',
        'side effect',
        # AccessCost
        'prior authorization', 'not covered',
        'insurance denied', 'insurance denial',
        'insurance', 'copay', 'deductible', 'shortage', 'backorder',
        'formulary', 'out-of-pocket',
        # EfficacyMismatch
        'insufficient response', 'no weight loss', 'weight plateau',
        'A1c not improved',
        'ineffective', 'plateau',
        'not working', 'not losing',
        # PatientPreference
        'patient preference', 'personal preference',
        'fear of injections', 'needle phobia',
        'does not want', 'do not want',
        'concerned about', 'decided to stop',
        'natural approach', 'declined',
        # ClinicalDecision
        'clinical decision', 'provider recommended',
        'contraindication', 'contraindicated',
        'renal function', 'pregnancy',
        # PlannedGoal
        'goal achieved', 'goal reached', 'target weight',
        'weight loss goal', 'course complete',
        'planned course', 'bridge therapy',
        'maintenance transition',
    ]
    for phrase in sorted(phrases, key=len, reverse=True):
        text = re.sub(re.escape(phrase), '[REASON_PHRASE]', text, flags=re.IGNORECASE)
    return text

def mask_clinical_note(text):
    text = mask_section_headers(text)
    text = mask_medications(text)
    text = mask_label_phrases(text)
    return text

def mask_dataset(input_path, output_path):
    data = [json.loads(line) for line in open(input_path)]
    with open(output_path, 'w') as f:
        for d in data:
            d = dict(d)
            if 'clinical_note' in d:
                d['clinical_note'] = mask_clinical_note(d['clinical_note'])
            elif 'conversations' in d:
                for msg in d['conversations']:
                    if msg.get('from') == 'human':
                        msg['value'] = mask_clinical_note(msg['value'])
            f.write(json.dumps(d) + '\n')
    print(f"Masked {len(data)} notes → {output_path}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3:
        mask_dataset(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python mask_clinical_notes.py <input.jsonl> <output.jsonl>")
