# Gender-Swap Stress Test Plan

## Goal

Test whether the GLP-1 discontinuation reason classifier changes performance when gender-coded cues in the synthetic notes are changed. This directly addresses the reviewer comment that the benchmark appears male-skewed and asks whether gender prompting or balancing was considered.

The experiment should answer:

1. Does changing gender-coded language in the held-out test notes change model predictions?
2. Does training on a gender-swapped or gender-balanced version of the dataset improve robustness?
3. Are any labels especially sensitive to gender-coded wording?

## Key Idea

Use `google/gemma-4-31B-it` to create gender-swapped versions of the clinical notes while preserving:

- the original discontinuation reason label,
- the medication scenario,
- all clinical facts not related to gender,
- note style and section structure,
- the primary reason evidence.

Gemma endpoint:

- Model: `google/gemma-4-31B-it`
- Serving: local vLLM
- Port: `8091`
- Expected OpenAI-compatible base URL: `http://localhost:8091/v1`

Then train and evaluate the same model setup used in the completed branch-held-out and masking stress tests:

- `Llama3.2-3B-Instruct` with the same LoRA setup used in the branch-held-out and masking stress tests.

## Data Inputs

Use the same benchmark split as the paper:

- Training set: `2,914` notes.
- Held-out test set: `320` notes.
- Original labels unchanged.

Use these files:

- Train: `/home/mshahidul/GLP_project/GLP_project_ad/experiments/dataset/train.jsonl`
- Test: `/home/mshahidul/GLP_project/GLP_project_ad/experiments/dataset/test.jsonl`

These files use the clean simple schema:

```json
{
  "clinical_note": "...",
  "primary_reason": "...",
  "note_id": "..."
}
```

Do not use `/home/mshahidul/GLP_project/GLP_project_ad/experiments/branch_held_out/data/` for the main gender-swap experiment. Those files are branch-specific splits for the pooled-vs-GPT-5.4 experiment, not the paper's fixed 2,914/320 benchmark split.

Create two transformed datasets:

- `train_gender_swapped.jsonl`
- `test_gender_swapped.jsonl`

Optional balanced version:

- `train_gender_balanced_augmented.jsonl`, containing original + gender-swapped notes.

## Gender Rewriting Policy

Use Gemma only as a controlled text editor, not as a generator of new clinical scenarios.

### Transformations

For male-coded notes, rewrite to female-coded notes:

- `he` -> `she`
- `him` -> `her`
- `his` -> `her`
- `man/male/gentleman/father/husband` -> appropriate female-coded counterpart where clinically coherent

For female-coded notes, rewrite to male-coded notes:

- `she` -> `he`
- `her` -> `him/his`, depending on grammar
- `woman/female/lady/mother/wife` -> appropriate male-coded counterpart where clinically coherent

For unknown or neutral notes:

- Keep neutral unless the experiment explicitly creates a balanced variant.
- If forcing a binary swap, record that the original note was neutral/unknown.

### Clinical Consistency Rules

Gemma must not change:

- primary reason category,
- GLP-1 drug name or dose,
- symptoms,
- insurance/cost details,
- efficacy trajectory,
- patient preference rationale,
- provider decision rationale,
- dates, labs, vitals, BMI, A1c, comorbidities,
- discontinuation action.

Exception:

- Pregnancy, fertility, contraception, or sex-specific anatomy can require careful edits. Do not blindly swap these. Either preserve the clinical meaning with a coherent counterpart or flag the note for exclusion/manual review.

## Suggested Gemma Prompt

```text
You are editing synthetic outpatient EHR-style clinical notes for a controlled NLP robustness experiment.

Task:
Rewrite the note by changing only gender-coded language from {SOURCE_GENDER} to {TARGET_GENDER}.

Strict rules:
1. Preserve the original GLP-1 discontinuation reason and all clinical facts.
2. Do not add new symptoms, diagnoses, medications, labs, dates, or reasoning.
3. Do not remove the discontinuation evidence.
4. Preserve note structure, section headers, and clinical style.
5. Change only pronouns, gendered nouns, and directly related grammar.
6. If a sex-specific clinical detail makes the swap medically incoherent, return "FLAG_FOR_REVIEW" and explain briefly.
7. Return valid JSON only.

Input JSON:
{
  "id": "{ID}",
  "label": "{LABEL}",
  "source_gender": "{SOURCE_GENDER}",
  "target_gender": "{TARGET_GENDER}",
  "clinical_note": "{CLINICAL_NOTE}"
}

Output JSON:
{
  "id": "{ID}",
  "label": "{LABEL}",
  "swap_status": "swapped" | "unchanged_neutral" | "flag_for_review",
  "source_gender": "{SOURCE_GENDER}",
  "target_gender": "{TARGET_GENDER}",
  "clinical_note_swapped": "...",
  "change_summary": "brief summary of gender-coded edits only"
}
```

## Quality Checks

Before training/evaluation, run automatic checks:

1. JSON parse check.
2. Label unchanged check.
3. Empty note check.
4. Length ratio check: swapped note length should be within roughly 0.85-1.15 of original.
5. Medication preservation check.
6. Primary-reason keyword/evidence preservation check.
7. Pronoun direction check.
8. Manual review of flagged notes and a random sample of at least 50 successful swaps.

Keep a log:

- total notes,
- swapped notes,
- unchanged neutral notes,
- flagged notes,
- parse failures,
- manually excluded notes.

## Evaluation Conditions

Run at least these three conditions.

| Cond | Train | Test | Purpose |
|---|---|---|---|
| A | Original train | Original test | Paper baseline |
| B | Original train | Gender-swapped test | Inference robustness to gender-coded changes |
| C | Gender-swapped train | Gender-swapped test | Whether performance remains strong when both train/test are gender-swapped |

Optional fourth condition:

| Cond | Train | Test | Purpose |
|---|---|---|---|
| D | Original + gender-swapped augmented train | Original and gender-swapped test | Whether augmentation improves robustness without hurting original performance |

## Metrics

Report for each condition:

- Accuracy.
- Macro-F1.
- Weighted F1.
- Per-class F1 and support.
- Confusion matrix.
- Number of valid predictions.

Robustness deltas:

- `A_macro_F1 - B_macro_F1`: sensitivity to gender-coded changes at inference.
- `A_macro_F1 - C_macro_F1`: remaining performance under full gender-swapped training/evaluation.
- If running D, compare original-test and swapped-test scores to check whether augmentation helps.

Also report subgroup metrics on the original test set:

- male-coded notes,
- female-coded notes,
- unknown/neutral notes.

## Interpretation Guide

### If A -> B drop is small

This suggests the model is not relying heavily on gender-coded language for discontinuation reason classification.

Suggested rebuttal wording:

> To assess the male-coded skew noted by the reviewer, we added a gender-swap stress test using `google/gemma-4-31B-it` as a controlled editor. The rewritten notes preserve the original label and clinical rationale while changing only gender-coded language. Comparing the original model on the original test set versus the gender-swapped test set showed only a small macro-F1 change (`[INSERT DELTA]`), suggesting that classification performance is not driven primarily by gender-coded wording.

### If A -> B drop is large

This suggests gender-coded wording or correlated note patterns affect predictions.

Suggested rebuttal wording:

> The gender-swap stress test revealed a measurable performance drop (`[INSERT DELTA]` macro-F1), indicating that the current synthetic benchmark contains gender-correlated artifacts. We will report this as a limitation and include the transformed split as a robustness diagnostic. We will also explore gender-balanced generation or augmentation in a future benchmark release.

### If augmentation helps

Suggested rebuttal wording:

> Training with original plus gender-swapped notes improved robustness on the gender-swapped test set from `[B]` to `[D]` macro-F1 while preserving original-test performance (`[A]` vs. `[D_ORIG]`). This suggests that controlled demographic augmentation is a practical mitigation for the observed synthetic skew.

## Manuscript/Rebuttal Placement

Add to the rebuttal under `Male-coded imbalance`.

Add to the manuscript either:

- Appendix validation/stress-test section, or
- Limitations + subgroup robustness appendix.

Minimum table to include:

| Condition | Accuracy | Macro-F1 | Weighted F1 | Valid/Total |
|---|---:|---:|---:|---:|
| Original -> Original | `[A_ACC]` | `[A_MACRO]` | `[A_WEIGHTED]` | `[A_VALID]` |
| Original -> Gender-swapped | `[B_ACC]` | `[B_MACRO]` | `[B_WEIGHTED]` | `[B_VALID]` |
| Gender-swapped -> Gender-swapped | `[C_ACC]` | `[C_MACRO]` | `[C_WEIGHTED]` | `[C_VALID]` |
| Augmented -> Gender-swapped, optional | `[D_ACC]` | `[D_MACRO]` | `[D_WEIGHTED]` | `[D_VALID]` |

## Files To Produce

Recommended outputs:

- `gender_swap_notes.py`
- `data/train_gender_swapped.jsonl`
- `data/test_gender_swapped.jsonl`
- `data/train_gender_balanced_augmented.jsonl` optional
- `results/condition_A_original.json`
- `results/condition_B_gender_swapped_inference.json`
- `results/condition_C_gender_swapped_retrain.json`
- `results/condition_D_augmented.json` optional
- `results/confusion_matrices/*.csv`
- `gender_swap_stress_test_report.md`


use conda env: vllm and gpu index 7 for run experiment.
save all the data and results and codes in here: /home/mshahidul/GLP_project/GLP_project_ad/experiments/gender_swap_stress_test
