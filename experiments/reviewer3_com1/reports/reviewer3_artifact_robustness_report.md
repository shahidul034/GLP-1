# Reviewer 3: Artifact-Robustness Evaluation Report

## 1. Reviewer Concern

Reviewer 3 raised concerns that the synthetic train/test splits in the GLP-1 discontinuation benchmark may share generator and template artifacts. Within-category cosine similarities range from 0.934 to 0.975, and both train and test data originate from the same generation and curation pipeline. To address this, we ran three artifact-robustness evaluations:

1. **Generator-Held-Out** — train on one generation branch, test on the other.
2. **Surface-Artifact Masking** — mask section headers, medication names, and label-associated trigger phrases.
3. **Template-Held-Out** — cluster notes by section-header template pattern, hold out entire template clusters during training.

All experiments use Llama3.2-3B-Instruct fine-tuned with LoRA (rank 16, 100 steps, lr=2e-4).

---

## 2. Experiment 1: Generator-Held-Out Evaluation

### Setup

| Direction | Train Branch | Train Count | Test Branch | Test Count |
|-----------|-------------|-------------|-------------|------------|
| Pooled → GPT-5.4 | Pooled (multi-model) | 1,635 | GPT-5.4 | 764 |
| GPT-5.4 → Pooled | GPT-5.4 | 764 | Pooled (multi-model) | 1,635 |

### Results

| Metric | Pooled → GPT-5.4 | GPT-5.4 → Pooled |
|--------|:----------------:|:-----------------:|
| Accuracy | **0.9279** | **0.7180** |
| Macro F1 | **0.8396** | **0.7017** |
| Weighted F1 | **0.9403** | **0.7001** |
| Valid / Total | 749 / 764 | 1,635 / 1,635 |

### Per-Class F1 (Pooled → GPT-5.4)

| Class | F1 | Support |
|-------|:--:|:-------:|
| ADVERSE_EFFECTS_TOLERABILITY | 0.9351 | 120 |
| ACCESS_COST_SYSTEM_BARRIERS | 1.0000 | 82 |
| EFFICACY_EXPECTATION_MISMATCH | 0.9235 | 176 |
| PATIENT_PREFERENCE_BELIEFS | 0.9717 | 109 |
| PLANNED_GOAL_BASED_DISCONTINUATION | 0.5000 | 3 |
| CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | 0.5714 | 31 |
| OTHER_UNKNOWN | 0.9754 | 228 |

### Per-Class F1 (GPT-5.4 → Pooled)

| Class | F1 | Support |
|-------|:--:|:-------:|
| ADVERSE_EFFECTS_TOLERABILITY | 0.6985 | 195 |
| ACCESS_COST_SYSTEM_BARRIERS | 0.8832 | 185 |
| EFFICACY_EXPECTATION_MISMATCH | 0.7273 | 136 |
| PATIENT_PREFERENCE_BELIEFS | 0.7893 | 154 |
| PLANNED_GOAL_BASED_DISCONTINUATION | 0.3535 | 155 |
| CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | 0.6805 | 623 |
| OTHER_UNKNOWN | 0.7797 | 187 |

### Interpretation

Cross-generator transfer is possible but **asymmetric**. Training on the more diverse pooled (multi-model) branch transfers well to GPT-5.4 notes (0.9279 accuracy, 0.8396 macro-F1). Training only on GPT-5.4 transfers less well to pooled notes (0.7180 accuracy, 0.7017 macro-F1). The asymmetry is explained by:
- **Training data volume**: pooled has 2.1× more training samples.
- **Branch diversity**: pooled contains multi-model outputs; GPT-5.4 is single-model and lower-diversity.
- **Low-support classes**: PlannedGoal and ClinicalDecision suffer when GPT-5.4 has very few examples of these categories.

These results show the model is not simply memorizing a single generator's style. Cross-generator generalization exists, though branch composition and class balance matter.

---

## 3. Experiment 2: Surface-Artifact Masking Stress Test

### Setup

| Condition | Train Data | Test Data | Samples |
|-----------|-----------|-----------|:-------:|
| A (baseline) | Original | Original | 2,914 / 320 |
| B (masked inference) | Original | Masked | 2,914 / 320 |
| C (masked retrain) | Masked | Masked | 2,914 / 320 |

Three artifact types were masked: **section headers** (HPI, Assessment, Plan, etc.), **medication names** (semaglutide, Ozempic, tirzepatide, etc.), and **label-associated trigger phrases** (nausea, insurance denied, plateau, goal achieved, etc.).

### Results

| Metric | A (baseline) | B (masked inference) | C (masked retrain) | Δ A→B | Δ A→C |
|--------|:----------:|:-------------------:|:------------------:|:----:|:----:|
| Accuracy | 0.9344 | 0.9250 | 0.9094 | **−0.0094** | **−0.0250** |
| Macro F1 | 0.9334 | 0.9295 | 0.9123 | **−0.0039** | **−0.0212** |
| Weighted F1 | 0.9339 | 0.9252 | 0.9102 | −0.0087 | −0.0237 |

### Per-Class F1 (Condition B — Masked Inference)

| Class | F1 | Support |
|-------|:--:|:-------:|
| ADVERSE_EFFECTS_TOLERABILITY | 0.9157 | 43 |
| ACCESS_COST_SYSTEM_BARRIERS | 0.9398 | 41 |
| EFFICACY_EXPECTATION_MISMATCH | 0.9762 | 42 |
| PATIENT_PREFERENCE_BELIEFS | 0.9600 | 37 |
| PLANNED_GOAL_BASED_DISCONTINUATION | 0.8800 | 27 |
| CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | 0.8786 | 83 |
| OTHER_UNKNOWN | 0.9565 | 47 |

### Per-Class F1 (Condition C — Masked Retrain)

| Class | F1 | Support |
|-------|:--:|:-------:|
| ADVERSE_EFFECTS_TOLERABILITY | 0.9024 | 43 |
| ACCESS_COST_SYSTEM_BARRIERS | 0.9512 | 41 |
| EFFICACY_EXPECTATION_MISMATCH | 0.9756 | 42 |
| PATIENT_PREFERENCE_BELIEFS | 0.8974 | 37 |
| PLANNED_GOAL_BASED_DISCONTINUATION | 0.8400 | 27 |
| CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | 0.8636 | 83 |
| OTHER_UNKNOWN | 0.9556 | 47 |

### Interpretation

The original-to-masked (A→B) macro-F1 drop is only **−0.0039**, suggesting that the model does not rely heavily on surface-level section headers, medication strings, or trigger phrases. Even when trained and evaluated entirely on masked notes (Condition C), macro-F1 remains **0.9123** — a 2.1% relative drop from baseline. These results indicate that the model's performance captures genuine clinical semantics rather than synthetic text shortcuts.

---

## 4. Experiment 3: Seed-Held-Out Evaluation

**Not performed due to metadata constraints.** The public dataset release includes only three fields per note (`clinical_note`, `primary_reason`, `note_id`). There is no seed ID, scenario ID, prompt template ID, or generation-metadata field that would allow grouping notes by their source seed or scenario.

As a partial proxy, the generator-held-out evaluation (Experiment 1) separates notes by generation pipeline (pooled multi-model vs. GPT-5.4), which captures the primary seed/pipeline variation present in the data. We recommend that future synthetic datasets include seed-origin metadata to enable direct seed-held-out evaluation.

---

## 5. Experiment 4: Template-Held-Out Evaluation

### Setup

Notes in the training set (2,914) were clustered by section-header template signature using k-means (k=8) on a 140-dimensional binary feature vector (section header presence/absence). Two clusters were held out for testing, leaving 2,102 training notes and 812 held-out test notes.

| Split | Clusters | Notes | Label Coverage |
|-------|----------|:-----:|:--------------:|
| Train | 6 clusters | 2,102 | All 7 labels |
| Test (held-out) | 2 clusters | 812 | All 7 labels |

### Results

| Metric | Template Held-Out |
|--------|:-----------------:|
| Accuracy | **0.9495** |
| Macro F1 | **0.9446** |
| Weighted F1 | **0.9495** |
| Valid / Total | 812 / 812 |

### Per-Class F1

| Class | F1 | Support |
|-------|:--:|:-------:|
| ADVERSE_EFFECTS_TOLERABILITY | 0.8936 | 22 |
| ACCESS_COST_SYSTEM_BARRIERS | 0.9231 | 48 |
| EFFICACY_EXPECTATION_MISMATCH | 0.9686 | 159 |
| PATIENT_PREFERENCE_BELIEFS | 0.9805 | 127 |
| PLANNED_GOAL_BASED_DISCONTINUATION | 0.9407 | 135 |
| CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | 0.9155 | 219 |
| OTHER_UNKNOWN | 0.9901 | 102 |

### Interpretation

The model achieves 0.9446 macro-F1 when evaluated on notes with unseen template structures, demonstrating that performance is not driven by memorization of specific section-header patterns. The test set contains all 7 labels with diverse class representation, and the model maintains strong per-class F1 across all categories (range: 0.8936–0.9901). This further supports that the model's classification ability generalizes beyond the specific note formats seen during training.

---

## 6. Summary of All Experiments

| Experiment | Test Condition | Accuracy | Macro F1 | Δ from Baseline |
|-----------|----------------|:--------:|:--------:|:---------------:|
| *Original benchmark (baseline)* | In-distribution test | 0.9344 | 0.9334 | — |
| Generator-Held-Out (Pooled→GPT) | Cross-generator | 0.9279 | 0.8396 | – |
| Generator-Held-Out (GPT→Pooled) | Cross-generator | 0.7180 | 0.7017 | – |
| Masking (original→masked test) | Surface artifacts removed | 0.9250 | 0.9295 | −0.0039 |
| Masking (masked train→test) | Full masking cycle | 0.9094 | 0.9123 | −0.0212 |
| Template-Held-Out | Unseen template clusters | 0.9495 | 0.9446 | —* |

*Note: Template-held-out uses a different train/test split and is not directly comparable to the original baseline.

---

## 7. Interpretation and Limitations

### What These Results Show

1. **Cross-generator generalization is real but asymmetric.** Pooled-branch training transfers well to GPT-5.4 notes. The weaker reverse direction confirms that the GPT-5.4 branch alone provides less diverse training signal.

2. **Surface artifacts are not the main driver of performance.** Masking section headers, medication names, and trigger phrases causes only a −0.0039 macro-F1 drop (conditioned on original training). The model clearly learns clinical semantics, not just pattern-matching shortcuts.

3. **Template structure is not memorized.** The model performs strongly (0.9446 macro-F1) on notes from entirely held-out template clusters, showing generalization across note format variations.

### Limitations

- **These are stress tests, not real-world validation.** All experiments use synthetic notes. Performance on real clinical notes may differ due to noise, ambiguity, and documentation variability not captured in synthetic data.
- **The generator-held-out asymmetry** (0.8396 vs. 0.7017 macro-F1) is a meaningful limitation. The benchmark's published results use a pooled train/test split, which masks the branch-composition effect visible in cross-branch evaluation.
- **Low-support classes** (PlannedGoal, ClinicalDecision in certain splits) remain challenging, particularly when training data is limited for that category.
- **Seed-held-out evaluation was not feasible** due to lack of metadata. We recommend future work include seed/scenario origin information in published datasets.

---

## 8. Confusion Matrices

Confusion matrices for all experiments are available as CSV files:

| Experiment | File |
|-----------|------|
| Generator-Held-Out (Pooled→GPT) | `results/confusion_matrices/llama3.2-3b_from_pooled_on_gpt54_confusion_matrix.csv` |
| Generator-Held-Out (GPT→Pooled) | `results/confusion_matrices/llama3.2-3b_from_gpt54_on_pooled_confusion_matrix.csv` |
| Masking (Condition B) | `results/confusion_matrices/condition_B_masked_inference_confusion_matrix.csv` |
| Template-Held-Out | `results/confusion_matrices/llama3.2-3b_template_held_out_confusion_matrix.csv` |

---

## 9. Suggested Rebuttal Wording

We agree that generator and template artifacts are a central risk for synthetic benchmarks. To test this directly, we added three artifact-robustness evaluations using Llama3.2-3B-Instruct fine-tuned with LoRA.

**Generator-held-out:** Training on the pooled multi-model branch and testing only on GPT-5.4 notes reached 0.9279 accuracy, 0.8396 macro-F1, and 0.9403 weighted F1. Reversing the direction (GPT-5.4 → pooled) reached 0.7180 accuracy and 0.7017 macro-F1. Cross-generator transfer is possible but asymmetric, with better transfer from the more diverse pooled branch.

**Surface-artifact masking:** We masked section headers, medication names, and strongly label-associated phrases. The original model evaluated on masked test text dropped only 0.0039 macro-F1 (0.9334 → 0.9295). A model trained and tested entirely on masked notes still achieved 0.9123 macro-F1 — a 2.1% relative drop. This suggests performance reflects clinical semantics, not surface pattern matching.

**Template-held-out:** We clustered notes by section-header template pattern and held out entire template clusters during training. The model achieved 0.9495 accuracy and 0.9446 macro-F1 on unseen template structures, confirming that performance generalizes beyond specific note formats.

**Limitations acknowledged:** The cross-branch asymmetry confirms that branch composition and class balance remain important. Seed-held-out evaluation was not feasible because the public release lacks seed/scenario metadata. We recommend that future datasets include origin metadata. These analyses are stress tests intended to characterize the benchmark's robustness to known synthetic artifacts, not as substitutes for larger real-note validation.

---

## 10. Files

| File | Description |
|------|-------------|
| `code/prepare_template_held_out.py` | Template clustering and split creation |
| `code/finetune_llama.py` | LoRA fine-tuning (copied from branch_held_out) |
| `code/evaluate_template.py` | Template-held-out evaluation |
| `data/train_template_held_out.jsonl` | Training data (in-cluster notes, 2,102) |
| `data/test_template_held_out.jsonl` | Test data (held-out clusters, 812) |
| `data/template_clusters.json` | Per-note cluster assignments |
| `models/llama3.2-3b_template_held_out/` | Fine-tuned model |
| `results/llama3.2-3b_template_held_out_results.json` | Full evaluation results |
| `results/confusion_matrices/` | Confusion matrices (CSV) |

Existing completed reports referenced:

| Report | Path |
|--------|------|
| Generator-Held-Out | `experiments/branch_held_out/branch_held_out_report.md` |
| Masking Stress Test | `experiments/mask_stress_test/mask_stress_test_report.md` |
