# Exploratory Subgroup Evaluation Report

## Overview

This report evaluates model performance across proxy demographic subgroups (age quartile, implied sex)
to characterize subgroup robustness in the GLP-1 discontinuation classification task.

**Models Evaluated:**
- Qwen3.5-9B (fine-tuned) - Decoder LLM
- Llama3.2-3B (fine-tuned) - Decoder LLM  
- BioClinical ModernBERT-large (fine-tuned) - Encoder LLM

**Test Set Size:** 320 samples

**Sparsity Threshold:** n < 10 → F1 score flagged as unreliable

---

## 1. Feature Extraction Summary

### Age Extraction
- Used regex patterns targeting forms such as `(\d+)y/?o`, `(\d+)year-old`, and `age (\d+)`
- Age quartiles were computed only for notes with extractable ages
- Notes without an extracted age were assigned to an `Unknown` group

### Pronoun-Based Implied Sex
- Male pronouns: he, him, his, he'd, he's, himself
- Female pronouns: she, her, hers, she'd, she's, herself
- Assigned to majority pronoun sex; "Unknown" if tied or zero

---

## 2. Subgroup Metrics

### qwen3.5-9b

#### Age Quartile Performance
| Quartile | Accuracy | Macro-F1 | Support | Reliable |
|----------|----------|----------|---------|----------|
| Q1 | 0.9841 | 0.9830 | 63 | Yes |
| Q2 | 1.0000 | 1.0000 | 60 | Yes |
| Q3 | 0.9851 | 0.8389 | 67 | Yes |
| Q4 | 0.9667 | 0.6877 | 30 | Yes |
| Unknown | 0.8400 | 0.7431 | 100 | Yes |

#### Implied Sex Performance
| Sex | Accuracy | Macro-F1 | Support | Reliable |
|-----|----------|----------|---------|----------|
| Female | 0.9286 | 0.7755 | 28 | Yes |
| Male | 0.9709 | 0.9714 | 275 | Yes |
| Unknown | 0.4706 | 0.2929 | 17 | Yes |

### llama3.2-3b

#### Age Quartile Performance
| Quartile | Accuracy | Macro-F1 | Support | Reliable |
|----------|----------|----------|---------|----------|
| Q1 | 0.9524 | 0.9480 | 63 | Yes |
| Q2 | 0.9833 | 0.9637 | 60 | Yes |
| Q3 | 1.0000 | 0.8571 | 67 | Yes |
| Q4 | 1.0000 | 0.7143 | 30 | Yes |
| Unknown | 0.8300 | 0.7833 | 100 | Yes |

#### Implied Sex Performance
| Sex | Accuracy | Macro-F1 | Support | Reliable |
|-----|----------|----------|---------|----------|
| Female | 0.8929 | 0.7604 | 28 | Yes |
| Male | 0.9709 | 0.9696 | 275 | Yes |
| Unknown | 0.4118 | 0.2208 | 17 | Yes |

### modernbert-large

#### Age Quartile Performance
| Quartile | Accuracy | Macro-F1 | Support | Reliable |
|----------|----------|----------|---------|----------|
| Q1 | 0.9841 | 0.9852 | 63 | Yes |
| Q2 | 1.0000 | 1.0000 | 60 | Yes |
| Q3 | 0.9851 | 0.8438 | 67 | Yes |
| Q4 | 1.0000 | 0.7143 | 30 | Yes |
| Unknown | 0.9400 | 0.9045 | 100 | Yes |

#### Implied Sex Performance
| Sex | Accuracy | Macro-F1 | Support | Reliable |
|-----|----------|----------|---------|----------|
| Female | 1.0000 | 0.8571 | 28 | Yes |
| Male | 0.9891 | 0.9882 | 275 | Yes |
| Unknown | 0.7059 | 0.4444 | 17 | Yes |

---

## 3. Statistical Significance Testing

Bootstrap resampling (1,000 iterations) was used to compute 95% confidence intervals
for the F1 score difference (ΔF1) between demographic groups.

### Interpretation
- **p-value < 0.05**: Statistically significant performance disparity between groups
- **95% CI does not include 0**: Indicates significant difference in F1 scores
- **Mean ΔF1**: Average difference in F1 (positive = majority group performs better)

### qwen3.5-9b

#### Implied Sex: Male vs Female**
- Majority: Male (n=275)
- Minority: Female (n=28)
- Mean ΔF1: 0.2567 ± 0.0905
- 95% CI: [0.1151, 0.4348]
- p-value: 0.0000
- Statistically Significant: Yes

#### Age: Q1 vs Unknown**
- Majority: Q1 (n=63)
- Minority: Unknown (n=100)
- Mean ΔF1: 0.2536 ± 0.0799
- 95% CI: [0.0918, 0.3963]
- p-value: 0.0030
- Statistically Significant: Yes

---

### llama3.2-3b

#### Implied Sex: Male vs Female**
- Majority: Male (n=275)
- Minority: Female (n=28)
- Mean ΔF1: 0.2629 ± 0.0927
- 95% CI: [0.1191, 0.4553]
- p-value: 0.0000
- Statistically Significant: Yes

#### Age: Q1 vs Unknown**
- Majority: Q1 (n=63)
- Minority: Unknown (n=100)
- Mean ΔF1: 0.1897 ± 0.0867
- 95% CI: [0.0314, 0.3552]
- p-value: 0.0120
- Statistically Significant: Yes

---

### modernbert-large

#### Implied Sex: Male vs Female**
- Majority: Male (n=275)
- Minority: Female (n=28)
- Mean ΔF1: 0.1785 ± 0.0754
- 95% CI: [0.1157, 0.2883]
- p-value: 0.0000
- Statistically Significant: Yes

#### Age: Q1 vs Unknown**
- Majority: Q1 (n=63)
- Minority: Unknown (n=100)
- Mean ΔF1: 0.1294 ± 0.0913
- 95% CI: [-0.0195, 0.3173]
- p-value: 0.0420
- Statistically Significant: Yes

---

## 4. Key Findings

### Exploratory Summary

| Model | Sex Disparity | Age Disparity |
|-------|---------------|---------------|
| qwen3.5-9b | Yes | Yes |
| llama3.2-3b | Yes | Yes |
| modernbert-large | Yes | Yes |


### Interpretation Guidelines

- **Statistically Significant disparity (p < 0.05)**: The observed performance difference
  between proxy-defined groups is unlikely under the bootstrap procedure used here.
  
- **Non-significant disparity**: Any observed difference could be due to sampling variability.
  The model appears to perform similarly across groups.

### Notes

- Age is extracted from clinical note text using pattern matching and may be incomplete
- Implied sex is based on pronoun usage and may not reflect self-identified gender
- These subgroup definitions are proxy variables and should not be treated as validated demographics
- Small subgroup sizes limit the reliability of subgroup-level comparisons

---

*Report generated automatically by fairness_evaluation.py*
