# Label-Independence Validation Report

---
## Experiment Overview

This report presents three experiments assessing whether the primary discontinuation reason can be inferred from the clinical note text alone, without exposing annotators or models to the generation target label, generator metadata, file paths, model predictions, or original LLM-judge scores.

### Experiments
1. **Baseline (label-conditioned)**: Re-analysis of existing annotation where annotators saw a pre-filled label and verified it.
2. **Label-blind human re-annotation**: Three annotators assigned labels from scratch seeing only note text and category definitions.
3. **Label-blind LLM re-judging**: `google/gemma-4-31B-it` classified the same blinded notes via a locally hosted vLLM endpoint, without label exposure.

---
## 1. Baseline: Existing Label-Conditioned Annotation

**Method:** The original annotation presented annotators with a pre-populated label (`assigned_label`) and asked whether they agreed. The `human_corrected_label` field records their final judgment. This is a label-conditioned annotation, not blind.

### Inter-Annotator Agreement
- **Fleiss' kappa**: 0.8291127484366294 (Almost Perfect)
- **Unanimous agreement**: 252/320 (78.8%)

#### Pairwise Agreement
| Pair | Agreement Rate | Cohen's κ |
| --- | --- | --- |
| Anu-Parth | 0.8063 | 0.7703 |
| Anu-Adiba | 0.8000 | 0.7626 |
| Parth-Adiba | 0.9625 | 0.9552 |

### Consensus vs Intended Label
- **Accuracy**: 0.965625
- **Cohen's κ**: 0.9589886985902365 (Almost Perfect)

**Caveat:** This baseline reflects annotation where the intended label was visible. It overestimates label recoverability from text alone.

---
## 2. Label-Blind Human Re-Annotation

Blind annotation results not available. Run the annotation UI and `compute_human_metrics.py` first.

---
## 3. Label-Blind LLM Re-Judging

**Method:** We ran `google/gemma-4-31B-it` (hosted on vLLM at `http://localhost:8091/v1`, GPU 4 / H200) on the same 320 blinded notes. The model received only the task description, seven category definitions, and the clinical note text. It was asked to return a JSON object with `primary_reason`, `confidence`, `supporting_evidence`, `secondary_factors`, and `ambiguity_note`. No intended label or generation metadata was provided.

### LLM vs Intended Labels
- **Accuracy**: 0.7781
- **Cohen's κ**: 0.7382 (Substantial)
- **Macro F1**: 0.7780
- **Weighted F1**: 0.7785
- **Valid predictions**: 311/320

### LLM Per-Class Metrics (vs Intended)
| Category | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| ADVERSE_EFFECTS_TOLERABILITY | 0.8039 | 0.9535 | 0.8723 | 43 |
| ACCESS_COST_SYSTEM_BARRIERS | 0.8298 | 0.9750 | 0.8966 | 40 |
| EFFICACY_EXPECTATION_MISMATCH | 0.8000 | 0.5714 | 0.6667 | 42 |
| PATIENT_PREFERENCE_BELIEFS | 0.9333 | 0.7778 | 0.8485 | 36 |
| PLANNED_GOAL_BASED_DISCONTINUATION | 0.6061 | 0.9091 | 0.7273 | 22 |
| CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | 0.9483 | 0.6627 | 0.7801 | 83 |
| OTHER_UNKNOWN | 0.5645 | 0.7778 | 0.6542 | 45 |

#### Confusion Matrix: Intended × LLM
| Intended \ LLM Prediction | ADVERSE_EFFECTS_TOLERABILITY | ACCESS_COST_SYSTEM_BARRIERS | EFFICACY_EXPECTATION_MISMATCH | PATIENT_PREFERENCE_BELIEFS | PLANNED_GOAL_BASED_DISCONTINUATION | CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | OTHER_UNKNOWN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **ADVERSE_EFFECTS_TOLERABILITY** | **41** | 0 | 0 | 0 | 0 | 1 | 1 |
| **ACCESS_COST_SYSTEM_BARRIERS** | 1 | **39** | 0 | 0 | 0 | 0 | 0 |
| **EFFICACY_EXPECTATION_MISMATCH** | 0 | 0 | **24** | 0 | 0 | 0 | 18 |
| **PATIENT_PREFERENCE_BELIEFS** | 0 | 0 | 0 | **28** | 8 | 0 | 0 |
| **PLANNED_GOAL_BASED_DISCONTINUATION** | 0 | 0 | 0 | 0 | **20** | 2 | 0 |
| **CLINICAL_DECISION_MEDICAL_APPROPRIATENESS** | 6 | 4 | 5 | 0 | 5 | **55** | 8 |
| **OTHER_UNKNOWN** | 3 | 4 | 1 | 2 | 0 | 0 | **35** |

### Low-Confidence Predictions
- **Threshold**: < 0.6
- **Count**: 9/320 (2.8%)
| Blind ID | Predicted | Confidence | Intended | Ambiguity Note |
| --- | --- | --- | --- | --- |
| B-066 | PARSE_ERROR | 0.00 | PLANNED_GOAL_BASED_DISCONTINUATION | Failed to parse: Please provide the full text of the clinical note. You have pro |
| B-134 | PARSE_ERROR | 0.00 | OTHER_UNKNOWN | Failed to parse: Please provide the clinical note you would like me to analyze.  |
| B-188 | PARSE_ERROR | 0.00 | ACCESS_COST_SYSTEM_BARRIERS | Failed to parse: Please provide the clinical note text you would like me to anal |
| B-197 | PARSE_ERROR | 0.00 | OTHER_UNKNOWN | Failed to parse: Please provide the clinical note content. You have provided the |
| B-203 | PARSE_ERROR | 0.00 | PLANNED_GOAL_BASED_DISCONTINUATION | Failed to parse: Please provide the clinical note text. You have provided the in |
| B-251 | PARSE_ERROR | 0.00 | PLANNED_GOAL_BASED_DISCONTINUATION | Failed to parse: Please provide the clinical note text you would like me to anal |
| B-297 | PARSE_ERROR | 0.00 | PATIENT_PREFERENCE_BELIEFS | Failed to parse: Please provide the clinical note content. You have provided the |
| B-307 | PARSE_ERROR | 0.00 | PLANNED_GOAL_BASED_DISCONTINUATION | Failed to parse: Please provide the clinical note you would like me to analyze.  |
| B-310 | PARSE_ERROR | 0.00 | PLANNED_GOAL_BASED_DISCONTINUATION | Failed to parse: Please provide the clinical note you would like me to analyze.  |

### Parse and API Errors
- **Count**: 9/320


---
## 4. Metadata Removal Audit

See [metadata_audit_report.md](../results/metadata_audit_report.md) for full details.

### Audit Summary
## Summary
- Note IDs with label hints: 0/320
- Category phrases in text: 306/320
- Prompt remnants: 1/320
- Direct label statements: 0/320
- Categories with distinctive boilerplate: 0/7

*Note: Finding a category-related phrase in note text does not necessarily mean the label is leaked — it may reflect clinically appropriate terminology. Each finding should be reviewed manually to assess whether it reveals the intended label.*

---
## 5. Discussion


### Representative Ambiguous Cases

Cases where annotators disagreed or where blind consensus differed from the intended label illustrate the boundary of the taxonomy. Common patterns include:

- **Adverse effects vs. clinical decision**: A note may describe both nausea (adverse effect) and a clinician's decision to stop due to renal function decline (clinical decision). The dominant reason may be interpreted differently.
- **Patient preference vs. planned goal**: A patient who achieved weight loss and chooses to stop could be classified under either category depending on whether the clinician's plan or the patient's initiative is viewed as primary.
- **Efficacy mismatch vs. other/unknown**: When the note is ambiguous about why the medication stopped, annotators may differ on whether the text supports a specific label or belongs in the catch-all category.

*Detailed disagreement cases will be added after annotation review.*

## 6. Limitations

Even if blind agreement is high, the following limitations remain:

1. **Target-label-conditioned generation:** All notes were generated from target labels. This experiment supports label coherence and text-level recoverability, but it does not fully reproduce naturally occurring label uncertainty in real clinical documentation.
2. **Synthetic note artifacts:** The notes were generated by language models and may contain rhetorical patterns, section structures, or clinical framing that make the primary reason more salient than in real notes.
3. **Controlled taxonomy:** The seven categories are predefined and annotators are forced to choose one. Real clinical notes may have genuinely multi-factorial discontinuations that do not fit neatly into a single category.
4. **Annotator expertise:** All three annotators have biomedical research experience and were familiar with the taxonomy from the baseline annotation. This may not generalize to naive annotators or real clinical coders.

## 7. Suggested Rebuttal Framing

> The label-blind audit showed that many labels are recoverable from the note text, but some classes remain ambiguous under independent review. We will report these disagreements and revise the benchmark framing to emphasize that the dataset is a controlled primary-reason benchmark rather than a natural estimate of real-world label uncertainty.

---
*Report generated: 2026-07-14 21:39*
