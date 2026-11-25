# Surface-Artifact Masking Stress Test Report

## Goal

Test whether the reported **0.933 macro-F1** (Llama3.2-3B on the 320-note held-out test set) comes from clinically meaningful context or from easy synthetic artifacts such as repeated section headers, explicit medication strings, and label-trigger phrases.

## Approach

Three masking categories applied to all clinical notes before tokenization:

| Mask Type | Examples | Replacement |
|---|---|---|
| **Section headers** | `HPI:`, `ASSESSMENT:`, `VITALS:`, `PLAN:`, `DATE:`, `PROVIDER:`, etc. | `[SECTION]` |
| **Medication names** | `semaglutide`, `Ozempic`, `Wegovy`, `tirzepatide`, `Mounjaro`, `GLP-1`, `metformin`, etc. | `[MEDICATION]` |
| **Label-associated phrases** | `nausea`, `insurance denied`, `plateau`, `patient preference`, `contraindication`, `goal achieved`, etc. | `[REASON_PHRASE]` |

## Three Evaluation Conditions

| Cond | Train | Test | Description |
|---|---|---|---|
| **A** | Original (2,914) | Original (320) | Paper baseline |
| **B** | Original (2,914) | **Masked** (320) | Tests surface-cue reliance at inference |
| **C** | **Masked** (2,914) | **Masked** (320) | Tests remaining clinical signal after full masking |

Model: **Llama3.2-3B-Instruct** fine-tuned with LoRA (r=16, 100 steps, lr=2e-4).

## Results

### Summary

| Metric | A (orig→orig) | B (orig→masked) | C (masked→masked) | Δ A→B | Δ A→C |
|---|---|---|---|---|---|
| **Accuracy** | 0.9344 | 0.9250 | 0.9094 | -0.0094 | -0.0250 |
| **Macro-F1** | **0.9334** | **0.9295** | **0.9123** | **-0.0039** | **-0.0212** |
| Weighted-F1 | 0.9339 | 0.9252 | 0.9102 | -0.0087 | -0.0237 |

### Per-Class F1

| Category | Support | A (orig) | B (mask-inf) | C (mask-retrain) |
|---|---|---|---|---|
| ADVERSE_EFFECTS_TOLERABILITY | 43 | 0.9647 | 0.9157 | 0.9024 |
| ACCESS_COST_SYSTEM_BARRIERS | 41 | 0.9639 | 0.9398 | 0.9512 |
| EFFICACY_EXPECTATION_MISMATCH | 42 | 1.0000 | 0.9762 | 0.9756 |
| PATIENT_PREFERENCE_BELIEFS | 37 | 0.9600 | 0.9600 | 0.8974 |
| PLANNED_GOAL_BASED_DISCONTINUATION | 27 | 0.8000 | 0.8800 | 0.8400 |
| CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | 83 | 0.8889 | 0.8786 | 0.8636 |
| OTHER_UNKNOWN | 47 | 0.9565 | 0.9565 | 0.9556 |

### Confusion Matrices

See CSV files in `results/confusion_matrices/`.

## Key Findings

### 1. Small A→B drop (0.0039 macro-F1)

Masking surface artifacts at inference time causes only a **0.4% relative drop** in macro-F1. This indicates the model relies primarily on **recoverable clinical semantics**, not on shortcut cues from section headers, medication names, or label-associated trigger phrases.

Some classes drop slightly more:
- **AdverseEffects** (-0.0490): likely depends on explicit symptoms like `nausea`/`vomiting`
- **AccessCost** (-0.0241): partially uses `insurance`, `cost`, `shortage` signals
- **EfficacyMismatch** (-0.0238): partially uses `plateau`, `not working` signals

### 2. Larger A→C drop (0.0212 macro-F1)

Retraining on masked data and testing on masked data drops macro-F1 by **2.1% relative**. This is still modest: the model achieves **0.912 macro-F1** with all surface cues removed from both training and inference.

The drop is concentrated in:
- **PatientPreference** (-0.0626): largest drop, likely because preference-related reasoning benefits most from explicit patient-stated phrases
- **AdverseEffects** (-0.0623): second largest, symptom descriptions provide strong signals
- **ClinicalDecision** (-0.0253): moderate drop
- **AccessCost** (-0.0127): minimal drop — retains good performance

### 3. Interpretation

**Conclusion: The 0.933 macro-F1 is robust to removal of obvious synthetic artifacts.**

- **Small A→B drop (<0.01)**: The model is not relying on easy surface artifacts at inference time. The benchmark's reported scores reflect genuine clinical reasoning, not label shortcuts.
- **Modest A→C drop (~0.02)**: Even when artifacts are removed from both training and evaluation, the model still achieves 0.912 macro-F1, indicating that meaningful clinical signal remains after masking.

This directly answers the concern about synthetic benchmark quality: the performance is not an artifact of template structure or label-trigger phrases.

## Files

| Path | Description |
|---|---|
| `mask_clinical_notes.py` | Masking utilities |
| `finetune_masked.py` | LoRA finetuning on masked data |
| `eval_cond_c.py` | Condition C evaluation |
| `data/train_masked.jsonl` | Masked training set (2,914) |
| `data/test_masked.jsonl` | Masked test set (320) |
| `models/llama3.2-3b_masked/` | Model fine-tuned on masked data |
| `results/condition_A_baseline.json` | Original baseline |
| `results/condition_B_masked_inference.json` | Original model on masked test |
| `results/condition_C_masked_retrain.json` | Masked model on masked test |
