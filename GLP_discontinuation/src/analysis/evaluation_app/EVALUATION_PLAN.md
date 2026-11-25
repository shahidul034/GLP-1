# GLP-1 Primary Reason Category — Human Evaluation Plan

## Overview

A Gradio-based web application for human evaluation of synthetic clinical notes. Evaluators review each note and determine whether the model-assigned **primary reason category** for GLP-1 RA discontinuation is correct.

## Data Source

- **File**: `/home/user/GLP_project/GLP_project_ad/code/finetune/training_data/train_test/test.jsonl`
- **Total records**: 320 clinical notes
- **7 categories** with the following distribution:

| Category | Count |
|---|---|
| CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | 83 |
| OTHER_UNKNOWN | 47 |
| ADVERSE_EFFECTS_TOLERABILITY | 43 |
| EFFICACY_EXPECTATION_MISMATCH | 42 |
| ACCESS_COST_SYSTEM_BARRIERS | 41 |
| PATIENT_PREFERENCE_BELIEFS | 37 |
| PLANNED_GOAL_BASED_DISCONTINUATION | 27 |

## Category Definitions

### CLINICAL_DECISION_MEDICAL_APPROPRIATENESS
Discontinuation driven by clinical judgment — the provider determines the GLP-1 RA is no longer medically appropriate. Includes: guideline-based deprescribing, contraindications (pancreatitis history, gastroparesis, pregnancy, cirrhosis/sarcopenia), drug-drug interactions, dose reduction for hypoglycemia risk, shifting to preferred agent (e.g., SGLT2i for renoprotection), pre-surgical holds, or patient clinical trajectory making ongoing GLP-1 therapy inadvisable.

### ADVERSE_EFFECTS_TOLERABILITY
Discontinuation primarily due to side effects or intolerable adverse reactions — GI symptoms (nausea, vomiting, diarrhea, constipation, bloating), injection site reactions, fatigue/dizziness/orthostasis attributed to the medication, gallbladder disease, appetite suppression beyond therapeutic intent, or any adverse effect leading to stop/dose-reduce.

### EFFICACY_EXPECTATION_MISMATCH
Discontinuation or clinical concern because the medication is perceived or observed as no longer providing adequate benefit — weight loss plateau, A1c not improving or worsening despite adherence, patient perception that "it stopped working," waning response, or mismatch between expected and actual outcomes.

### ACCESS_COST_SYSTEM_BARRIERS
Discontinuation due to external/systemic barriers — insurance denial, prior authorization failure, formulary restrictions, medication shortage/backorder, cost/affordability, pharmacy supply issues, or VA/non-VA coverage changes preventing continued access.

### PATIENT_PREFERENCE_BELIEFS
Discontinuation driven by patient choice not primarily due to side effects or medical appropriateness — social stigma, family influence, misinformation/fear from media, desire to not be medication-dependent, preference for "natural" approaches, injection fatigue, or psychosocial distress about long-term pharmacotherapy.

### PLANNED_GOAL_BASED_DISCONTINUATION
Discontinuation that was anticipated from the start — time-limited trial with pre-defined endpoints met, goal weight/A1c achieved and maintained, planned taper/step-off after reaching targets, pre-conception/pregnancy planning, or structured deprescribing protocol where the original treatment plan included stopping.

### OTHER_UNKNOWN
Discontinuation where the reason cannot be determined from the clinical note — unclear documentation, patient unable to explain, ambiguous chart entries, conflicting information, or insufficient detail to assign any of the above categories.

## Evaluation Instructions

1. Read the full clinical note carefully
2. Determine the **primary reason** for GLP-1 RA discontinuation or change
3. Select the category that best represents the primary reason
4. Rate your confidence in the assignment (High / Medium / Low)
5. If multiple reasons are present, select the one that appears to be the **dominant/primary** driver
6. Add optional comments for ambiguous or difficult cases
7. Click Submit to save and advance to the next note

## App Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Note order | Sequential (file order) | Consistent across evaluators |
| Model category visibility | Hidden (blind evaluation) | Avoids anchoring bias |
| Authentication | Username only | Trusted internal team environment |
| Data storage | Local JSON files | No database dependency |

## App Flow

### 1. Login
- Enter username to log in
- New usernames auto-register
- Returning users resume where they left off

### 2. Instructions Panel
- Displays all 7 category definitions with detailed descriptions
- Shows evaluation guidelines and confidence scale
- Available throughout the session via a tab

### 3. Evaluation Interface
- Clinical note displayed in a scrollable text area
- 7 radio buttons for category selection (definitions shown inline)
- Confidence dropdown (High / Medium / Low)
- Optional free-text comment box
- Submit button — saves result and loads next note
- Progress bar showing completion (X / 320)

### 4. Resume Logic
- On login, the app checks which notes the user has already evaluated
- Automatically skips to the next un-evaluated note
- User can close the browser and return later to continue

### 5. Completion Screen
- Displayed when all 320 notes have been evaluated
- Shows summary statistics

## Data Storage

All files stored in: `/home/user/GLP_project/GLP_project_ad/code/interface/data_evaluation/`

### users.json
```json
["evaluator1", "evaluator2", "evaluator3"]
```

### user_progress.json
```json
{
  "evaluator1": {
    "current_index": 45,
    "completed_indices": [0, 1, 2, 3, 4],
    "last_active": "2026-05-17T10:30:00"
  }
}
```

### evaluations.json
```json
[
  {
    "username": "evaluator1",
    "note_id": "850",
    "note_index": 0,
    "model_category": "OTHER_UNKNOWN",
    "evaluator_category": "ADVERSE_EFFECTS_TOLERABILITY",
    "confidence": "Medium",
    "comment": "Ambiguous — mentions both GI issues and access problems",
    "timestamp": "2026-05-17T10:30:00"
  }
]
```

## Implementation

### File Structure
```
code/interface/data_evaluation/
├── evaluate_app.py        # Main Gradio application
├── users.json             # User store (auto-created)
├── evaluations.json       # All evaluation results (auto-created)
├── user_progress.json     # Resume state per user (auto-created)
└── EVALUATION_PLAN.md     # This file
```

### Dependencies
- `gradio` — Web interface framework
- Standard library: `json`, `datetime`, `os`, `hashlib`

### Launch Command
```bash
cd /home/user/GLP_project/GLP_project_ad/code/interface/data_evaluation
python evaluate_app.py
```

The app will start on `http://localhost:7860` by default.

## Post-Evaluation Analysis

After evaluation is complete, the `evaluations.json` file can be analyzed to compute:
- **Inter-annotator agreement** (Cohen's kappa, Fleiss' kappa) between evaluators
- **Model vs. human agreement** — how often the model category matches the evaluator category
- **Per-category accuracy** — which categories the model gets right vs. wrong
- **Confidence distributions** — which categories evaluators are most/least confident about
- **Disagreement patterns** — which category pairs are most commonly confused
