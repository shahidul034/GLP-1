# Annotator Evaluation Results
*Dataset:* 320 items annotated by 3 human evaluators (1st, 2nd, 3rd)
*Label categories:* 7
## 1. Methodology & Metric Definitions
### Cohen's Kappa ($\kappa$)
Measures inter-rater agreement for two raters, correcting for chance agreement:
$$\kappa = \frac{p_o - p_e}{1 - p_e}$$
where $p_o$ is observed agreement and $p_e$ is expected agreement by chance.
Interpretation (Landis & Koch, 1977): $<0.20$ Slight, $0.20-0.40$ Fair, $0.40-0.60$ Moderate, $0.60-0.80$ Substantial, $0.80-1.00$ Almost Perfect.
### Fleiss' Kappa ($\kappa$)
Generalization of Cohen's Kappa for more than two raters (Davies & Fleiss, 1982). Computed as the average pairwise observed and expected agreement across all rater pairs.
### Accuracy
Simple proportion of exact label matches between model predictions and human labels:
$$\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}$$
### Disagreement Resolution Protocol
When the 1st and 2nd annotators disagree on a label, the 3rd annotator serves as the **master annotator**. The 3rd annotator reviews each disagreement case manually and resolves it by hand, providing the final adjudicated label. This ensures that all conflicting annotations are settled by an experienced expert rather than relying solely on majority vote.

### Majority Vote Consensus (Gold Standard)
For each item, the consensus label is determined by majority vote across the three annotators. In case of a 3-way tie (all three annotators disagree), the 3rd annotator's label is used as the tiebreaker, consistent with their role as master annotator.
## 2. Inter-Annotator Agreement (IAA) Results
### Pairwise Raw Agreement
| Pair | Agreement Count | Agreement Rate |
| --- | --- | --- |
| 1st-2nd | 258/320 | 0.8063 |
| 1st-3rd | 256/320 | 0.8000 |
| 2nd-3rd | 308/320 | 0.9625 |

### Cohen's Kappa: 1st vs 2nd
- **$\kappa$ = 0.7703** (Substantial)
- This measures the agreement between 1st and 2nd beyond what would be expected by chance.
### Fleiss' Kappa: All Three Raters
- **$\kappa$ = 0.8291** (Almost Perfect)
- This measures the global agreement across all three annotators simultaneously.
### Human Disagreement Summary
- Items where all 3 annotators agree: 252/320 (78.8%)
- Items with any disagreement: 68/320 (21.2%)
  - 2-vs-1 majority disagreements: 66
  - 3-way disagreements (all different): 2
## 3. Model vs. Human Alignment Metrics
### Per-Annotator Results
| Annotator | Accuracy | Cohen's $\kappa$ | Interpretation |
| --- | --- | --- | --- |
| 1st | 0.7906 (79.1%) | 0.7523 | Substantial |
| 2nd | 0.9531 (95.3%) | 0.9441 | Almost Perfect |
| 3rd | 0.9656 (96.6%) | 0.9590 | Almost Perfect |

### Model vs. Consensus (Gold Standard)
- **Accuracy:** 0.9656 (96.6%)
- **Cohen's $\kappa$:** 0.9590 (Almost Perfect)
### Confusion Matrices

**Model vs 1st**
| True \ Pred | ACCESS_COST_SYSTEM_BARRIERS | ADVERSE_EFFECTS_TOLERABILITY | CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | EFFICACY_EXPECTATION_MISMATCH | OTHER_UNKNOWN | PATIENT_PREFERENCE_BELIEFS | PLANNED_GOAL_BASED_DISCONTINUATION |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **ACCESS_COST_SYSTEM_BARRIERS** | 39 | 0 | **4** | 0 | **2** | 0 | 0 |
| **ADVERSE_EFFECTS_TOLERABILITY** | 0 | 34 | **6** | 0 | **1** | 0 | 0 |
| **CLINICAL_DECISION_MEDICAL_APPROPRIATENESS** | 0 | 0 | 62 | 0 | 0 | **1** | 0 |
| **EFFICACY_EXPECTATION_MISMATCH** | 0 | 0 | **2** | 23 | 0 | 0 | 0 |
| **OTHER_UNKNOWN** | **2** | **9** | **5** | **19** | 43 | **2** | **5** |
| **PATIENT_PREFERENCE_BELIEFS** | 0 | 0 | 0 | 0 | **1** | 30 | 0 |
| **PLANNED_GOAL_BASED_DISCONTINUATION** | 0 | 0 | **4** | 0 | 0 | **4** | 22 |

**Model vs 2nd**
| True \ Pred | ACCESS_COST_SYSTEM_BARRIERS | ADVERSE_EFFECTS_TOLERABILITY | CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | EFFICACY_EXPECTATION_MISMATCH | OTHER_UNKNOWN | PATIENT_PREFERENCE_BELIEFS | PLANNED_GOAL_BASED_DISCONTINUATION |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **ACCESS_COST_SYSTEM_BARRIERS** | 40 | 0 | **2** | 0 | 0 | 0 | 0 |
| **ADVERSE_EFFECTS_TOLERABILITY** | 0 | 43 | **1** | 0 | 0 | 0 | 0 |
| **CLINICAL_DECISION_MEDICAL_APPROPRIATENESS** | 0 | 0 | 78 | 0 | **1** | 0 | **1** |
| **EFFICACY_EXPECTATION_MISMATCH** | 0 | 0 | **1** | 42 | **1** | 0 | 0 |
| **OTHER_UNKNOWN** | **1** | 0 | 0 | 0 | 45 | **1** | **5** |
| **PATIENT_PREFERENCE_BELIEFS** | 0 | 0 | 0 | 0 | 0 | 36 | 0 |
| **PLANNED_GOAL_BASED_DISCONTINUATION** | 0 | 0 | **1** | 0 | 0 | 0 | 21 |

**Model vs 3rd**
| True \ Pred | ACCESS_COST_SYSTEM_BARRIERS | ADVERSE_EFFECTS_TOLERABILITY | CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | EFFICACY_EXPECTATION_MISMATCH | OTHER_UNKNOWN | PATIENT_PREFERENCE_BELIEFS | PLANNED_GOAL_BASED_DISCONTINUATION |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **ACCESS_COST_SYSTEM_BARRIERS** | 39 | 0 | 0 | 0 | 0 | 0 | 0 |
| **ADVERSE_EFFECTS_TOLERABILITY** | 0 | 43 | **1** | 0 | 0 | 0 | 0 |
| **CLINICAL_DECISION_MEDICAL_APPROPRIATENESS** | 0 | 0 | 82 | 0 | 0 | 0 | **1** |
| **EFFICACY_EXPECTATION_MISMATCH** | 0 | 0 | 0 | 42 | 0 | 0 | 0 |
| **OTHER_UNKNOWN** | **1** | 0 | 0 | 0 | 47 | **1** | **4** |
| **PATIENT_PREFERENCE_BELIEFS** | **1** | 0 | 0 | 0 | 0 | 34 | 0 |
| **PLANNED_GOAL_BASED_DISCONTINUATION** | 0 | 0 | 0 | 0 | 0 | **2** | 22 |

**Model vs Consensus (Gold Standard)**
| True \ Pred | ACCESS_COST_SYSTEM_BARRIERS | ADVERSE_EFFECTS_TOLERABILITY | CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | EFFICACY_EXPECTATION_MISMATCH | OTHER_UNKNOWN | PATIENT_PREFERENCE_BELIEFS | PLANNED_GOAL_BASED_DISCONTINUATION |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **ACCESS_COST_SYSTEM_BARRIERS** | 40 | 0 | **1** | 0 | 0 | 0 | 0 |
| **ADVERSE_EFFECTS_TOLERABILITY** | 0 | 43 | **1** | 0 | 0 | 0 | 0 |
| **CLINICAL_DECISION_MEDICAL_APPROPRIATENESS** | 0 | 0 | 81 | 0 | 0 | 0 | **1** |
| **EFFICACY_EXPECTATION_MISMATCH** | 0 | 0 | 0 | 42 | 0 | 0 | 0 |
| **OTHER_UNKNOWN** | **1** | 0 | 0 | 0 | 47 | **1** | **5** |
| **PATIENT_PREFERENCE_BELIEFS** | 0 | 0 | 0 | 0 | 0 | 35 | 0 |
| **PLANNED_GOAL_BASED_DISCONTINUATION** | 0 | 0 | 0 | 0 | 0 | **1** | 21 |

## 4. Discussion & Error Analysis
### Model Confusion Patterns (Top 10 per comparison)

**Model vs 1st — Most Common Misclassifications:**
| True Label | Predicted Label | Count |
| --- | --- | --- |
| OTHER_UNKNOWN | EFFICACY_EXPECTATION_MISMATCH | 19 |
| OTHER_UNKNOWN | ADVERSE_EFFECTS_TOLERABILITY | 9 |
| ADVERSE_EFFECTS_TOLERABILITY | CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | 6 |
| OTHER_UNKNOWN | CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | 5 |
| OTHER_UNKNOWN | PLANNED_GOAL_BASED_DISCONTINUATION | 5 |
| ACCESS_COST_SYSTEM_BARRIERS | CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | 4 |
| PLANNED_GOAL_BASED_DISCONTINUATION | CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | 4 |
| PLANNED_GOAL_BASED_DISCONTINUATION | PATIENT_PREFERENCE_BELIEFS | 4 |
| ACCESS_COST_SYSTEM_BARRIERS | OTHER_UNKNOWN | 2 |
| EFFICACY_EXPECTATION_MISMATCH | CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | 2 |


**Model vs 2nd — Most Common Misclassifications:**
| True Label | Predicted Label | Count |
| --- | --- | --- |
| OTHER_UNKNOWN | PLANNED_GOAL_BASED_DISCONTINUATION | 5 |
| ACCESS_COST_SYSTEM_BARRIERS | CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | 2 |
| ADVERSE_EFFECTS_TOLERABILITY | CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | 1 |
| CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | OTHER_UNKNOWN | 1 |
| CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | PLANNED_GOAL_BASED_DISCONTINUATION | 1 |
| EFFICACY_EXPECTATION_MISMATCH | CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | 1 |
| EFFICACY_EXPECTATION_MISMATCH | OTHER_UNKNOWN | 1 |
| OTHER_UNKNOWN | ACCESS_COST_SYSTEM_BARRIERS | 1 |
| OTHER_UNKNOWN | PATIENT_PREFERENCE_BELIEFS | 1 |
| PLANNED_GOAL_BASED_DISCONTINUATION | CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | 1 |


**Model vs 3rd — Most Common Misclassifications:**
| True Label | Predicted Label | Count |
| --- | --- | --- |
| OTHER_UNKNOWN | PLANNED_GOAL_BASED_DISCONTINUATION | 4 |
| PLANNED_GOAL_BASED_DISCONTINUATION | PATIENT_PREFERENCE_BELIEFS | 2 |
| ADVERSE_EFFECTS_TOLERABILITY | CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | 1 |
| CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | PLANNED_GOAL_BASED_DISCONTINUATION | 1 |
| OTHER_UNKNOWN | ACCESS_COST_SYSTEM_BARRIERS | 1 |
| OTHER_UNKNOWN | PATIENT_PREFERENCE_BELIEFS | 1 |
| PATIENT_PREFERENCE_BELIEFS | ACCESS_COST_SYSTEM_BARRIERS | 1 |


**Model vs Consensus — Most Common Misclassifications:**
| True Label | Predicted Label | Count |
| --- | --- | --- |
| OTHER_UNKNOWN | PLANNED_GOAL_BASED_DISCONTINUATION | 5 |
| ACCESS_COST_SYSTEM_BARRIERS | CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | 1 |
| ADVERSE_EFFECTS_TOLERABILITY | CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | 1 |
| CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | PLANNED_GOAL_BASED_DISCONTINUATION | 1 |
| OTHER_UNKNOWN | ACCESS_COST_SYSTEM_BARRIERS | 1 |
| OTHER_UNKNOWN | PATIENT_PREFERENCE_BELIEFS | 1 |
| PLANNED_GOAL_BASED_DISCONTINUATION | PATIENT_PREFERENCE_BELIEFS | 1 |

### Key Observations
1. **Overall Model Performance:** The model correctly matches the consensus gold standard on 309/320 items (96.6%). The Cohen's $\kappa$ of 0.9590 (Almost Perfect) indicates the model's agreement beyond chance.
2. **Human Agreement Level:** The Fleiss' $\kappa$ of 0.8291 (Almost Perfect) across all three annotators suggests almost perfect agreement, indicating a well-defined annotation schema.
3. **1st vs 2nd Agreement:** Cohen's $\kappa$ = 0.7703 (Substantial). Raw agreement rate: 258/320 (80.6%).
4. **Label Distribution Skew:** The most frequent label is `CLINICAL_DECISION_MEDICAL_APPROPRIATENESS` (82 items, 25.6%). This class imbalance may inflate accuracy scores; Cohen's $\kappa$ provides a more reliable measure by correcting for chance.
5. **Areas of Human Disagreement:** 68 items (21.2%) had at least one disagreement among the three annotators. Of these, 2 items had all three annotators assigning different labels, indicating genuinely ambiguous cases.
6. **Model Error Patterns:** The most common model misclassifications tend to involve:
   - True `OTHER_UNKNOWN` → Predicted `PLANNED_GOAL_BASED_DISCONTINUATION` (5 instances)
   - True `ACCESS_COST_SYSTEM_BARRIERS` → Predicted `CLINICAL_DECISION_MEDICAL_APPROPRIATENESS` (1 instances)
   - True `ADVERSE_EFFECTS_TOLERABILITY` → Predicted `CLINICAL_DECISION_MEDICAL_APPROPRIATENESS` (1 instances)

7. **Model vs Individual Annotators:** The model's $\kappa$ scores range from 0.7523 to 0.9590 across individual annotators. The model aligns best with 3rd ($\kappa$ = 0.9590).
### Recommendations
- **Annotation Guidelines:** Review the label pairs that show the most human disagreement to refine annotation guidelines.
- **Model Improvement:** Focus model training on the most common confusion pairs identified above.
- **Ambiguous Cases:** Items with 3-way human disagreement may benefit from expert review or additional context.
