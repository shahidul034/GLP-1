# Statistical Uncertainty Experiment Plan

## Goal

Address Reviewer 3 W4/C2 by quantifying uncertainty around accuracy, macro-F1, and weighted F1 and testing whether reported model differences—especially 0.933 versus 0.918 macro-F1—are supported on the fixed 320-note test set.

The experiment should reuse saved prediction-level outputs. Model inference does not need to be rerun.

## Experiment directory

Create a self-contained experiment under `GLP_project_ad/experiments/statistical_uncertainty/` containing:

- `statistical_uncertainty_plan.md`: this protocol.
- `analyze_uncertainty.py`: prediction loader, bootstrap analysis, pairwise tests, and report generation.
- `test_analyze_uncertainty.py`: tests for alignment, invalid outputs, metrics, and reproducibility.
- `README.md`: dependencies and execution commands.
- `results/`: generated CSV, JSON, and Markdown results.

## Input files

Load prediction-level records from:

- `GLP_project_ad/code/finetune/testing_results/`
- `GLP_project_ad/code/gpt5.4/results/`
- `GLP_project_ad/code/few_shots/`

Use this file as the canonical source of note IDs and gold labels:

- `GLP_project_ad/code/finetune/training_data/train_test/test.jsonl`

Do not use summary JSON files as the input for statistical analysis. Recompute every metric from the saved note-level predictions.

## Step 1: Build and validate the prediction loader

The analysis script should:

1. Read the 320 canonical test notes in their original order.
2. Load each model's prediction-level result file.
3. Normalize model names and evaluation regimes:
   - base
   - fine-tuned
   - zero-shot
   - few-shot
4. Align predictions by `note_id`, not by row position.
5. Validate:
   - every canonical note ID is unique;
   - every result file contains unique note IDs;
   - stored gold labels match the canonical gold labels;
   - predictions belong to the seven allowed classes or are null/invalid;
   - missing and extra note IDs are reported;
   - unexpected result files are not silently included.
6. Recompute the stored point estimates and fail with a clear error when an unexplained discrepancy occurs.

## Step 2: Handle invalid model outputs

Some result files, particularly few-shot results, contain null predictions. Dropping these independently for each model creates different evaluation denominators and weakens paired comparisons.

Use two analysis policies:

### Primary policy: all 320 notes

- Retain every test note.
- Map null or unparseable predictions to `INVALID_OUTPUT`.
- Compute metrics over the seven gold classes.
- Treat invalid output as a model failure for the note.
- Report the number and percentage of invalid predictions for each model.

### Sensitivity policy: valid outputs only

- Reproduce the current valid-output-only metric calculation for each model.
- Report the valid sample size and invalid rate.
- For pairwise complete-case comparisons, use only notes for which both models produced valid predictions.
- Report the shared sample size for every comparison.

The report must explain differences between the primary all-note estimates and the manuscript's current point estimates.

## Step 3: Bootstrap confidence intervals

Use a paired, class-stratified bootstrap:

1. Group the 320 note IDs by their gold class.
2. Sample with replacement within each class.
3. Preserve the original support of all seven classes in every bootstrap replicate.
4. Use 10,000 bootstrap replicates.
5. Use random seed 42.
6. Use the same sampled note indices for all models in a paired comparison.

For every model and regime, calculate:

- Accuracy
- Macro-F1
- Weighted F1
- 95% percentile bootstrap confidence interval for each metric

## Step 4: Paired model comparisons

For each planned comparison, compute:

- Point difference: `metric_A - metric_B`
- Bootstrap distribution of the paired difference
- 95% confidence interval for the difference
- Two-sided bootstrap p-value
- Holm-adjusted p-value
- Number of paired notes
- Whether the difference confidence interval excludes zero

Use a finite-replicate correction when calculating bootstrap p-values.

Apply Holm correction separately to these comparison families:

1. All pairwise macro-F1 comparisons among fine-tuned models.
2. Matched base-versus-fine-tuned comparisons for each model.
3. Manuscript-highlighted comparisons:
   - Qwen3.5-9B fine-tuned versus Llama3.2-3B fine-tuned
   - Qwen3.5-9B fine-tuned versus Qwen3-14B fine-tuned

Do not claim that two models are equivalent merely because their difference is not statistically significant. State only that the experiment did not detect a supported difference on this test set.

## Step 5: Generate outputs

Write the following files:

### `results/model_intervals.csv`

Include:

- Model
- Regime
- Invalid-output policy
- Total notes
- Valid predictions
- Invalid predictions
- Invalid rate
- Metric
- Point estimate
- Lower 95% CI
- Upper 95% CI

### `results/pairwise_comparisons.csv`

Include:

- Comparison family
- Model A
- Model B
- Metric
- Point difference
- Lower 95% CI
- Upper 95% CI
- Unadjusted p-value
- Holm-adjusted p-value
- Paired sample size
- Interpretation flag

### `results/fine_tuned_macro_f1_matrix.csv`

Create a compact pairwise matrix suitable for an appendix or supplementary file.

### `results/analysis_summary.json`

Record:

- Input files
- Test-set size
- Class supports
- Random seed
- Number of bootstrap replicates
- Invalid-output policies
- Metrics
- All numerical results

### `results/statistical_uncertainty_report.md`

Include:

1. Research question.
2. Input-data validation results.
3. Statistical methods.
4. Confidence-interval table.
5. Pairwise-comparison table.
6. Invalid-output sensitivity analysis.
7. Main findings in plain language.
8. Limitations.
9. A clear answer about whether the differences between models are statistically supported.
10. A list of results that require cautious interpretation.
11. Recommended next analyses, without modifying any manuscript or rebuttal files.

## Step 6: Testing and verification

Add tests using small synthetic prediction arrays:

1. Identical models must have a paired difference of zero.
2. A perfect model compared with an incorrect model must have the expected positive difference.
3. Shuffled prediction rows must align correctly by note ID.
4. Duplicate note IDs must raise an error.
5. Missing note IDs must raise an error or be explicitly documented according to configuration.
6. Gold-label disagreement must raise an error.
7. Null predictions must be handled correctly under both invalid-output policies.
8. Bootstrap results must be reproducible with seed 42.
9. Valid-output-only metrics must match the stored metrics within a strict numerical tolerance.

Before accepting the results, confirm:

- Every included result file has been checked against the canonical 320-note test set.
- Point estimates can be reproduced from note-level predictions.
- Bootstrap outputs are deterministic.
- Holm correction is applied within the correct comparison family.
- The report clearly distinguishes test-set sampling uncertainty from training or inference variability.

## Statistical interpretation

Treat a model ranking as supported only when the paired difference confidence interval excludes zero and the multiplicity-adjusted result supports the comparison.

This experiment estimates uncertainty caused by having a finite 320-note test sample. It does not estimate uncertainty caused by:

- Different fine-tuning seeds
- Different model checkpoints
- Stochastic decoding
- Few-shot exemplar selection
- Prompt variation
- API model updates or drift

These limitations must be stated in the final report.

## Final report requirements

After all validation checks pass, produce a final report at:

- `results/statistical_uncertainty_report.md`

The final report should be self-contained and written for manual review. It must:

1. Summarize the data and models included.
2. Explain the bootstrap method in accessible language.
3. Present confidence intervals for every model.
4. Present corrected pairwise comparisons.
5. Directly analyze the 0.933 versus 0.918 macro-F1 comparison.
6. Explain how invalid predictions affect the results.
7. Separate statistically supported findings from unsupported rankings.
8. Identify discrepancies between recomputed and stored metrics.
9. State all methodological limitations.
10. End with a concise conclusions section and recommended next steps.

Do not modify `paper_writting/acl_latex.tex`, `reviewer_comments_organized.md`, or `rebuttal_responses.md`. The user will review the final report and decide how to use the findings.

## Agent execution checklist

- [ ] Build the canonical prediction loader.
- [ ] Validate all note IDs and gold labels.
- [ ] Recompute existing point metrics.
- [ ] Implement both invalid-output policies.
- [ ] Implement the paired class-stratified bootstrap.
- [ ] Implement pairwise difference intervals and p-values.
- [ ] Implement Holm correction by comparison family.
- [ ] Add unit tests.
- [ ] Run 10,000 bootstrap replicates with seed 42.
- [ ] Generate CSV, JSON, and Markdown outputs.
- [ ] Review discrepancies and invalid-output effects.
- [ ] Write the self-contained final analysis report.
- [ ] Leave the paper and rebuttal unchanged.
