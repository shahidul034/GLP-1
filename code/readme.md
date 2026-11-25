# GLP-1 RA Synthetic Clinical Notes — Code

Pipeline for generating, evaluating, and filtering synthetic clinical notes about GLP-1 receptor agonist (GLP-1 RA) medication discontinuation, inspired by the study [Using large language models to assess public perceptions around glucagon-like peptide-1 receptor agonists on social media](https://doi.org/10.1038/s43856-024-00566-z).

## Workflow

```
generate → judge → filter by judgment → diversity filter
```

## Scripts

| Script | Purpose |
|---|---|
| `generate_synthetic.py` | Generates synthetic clinical notes using a **single model** (default: Gemma-4-31B). Produces 100 notes per discontinuation category via batched vLLM inference with few-shot prompting. |
| `generate_synthetic_multi.py` | Multi-model variant of the generator. Supports **Qwen3.5-27B**, **Qwen3.6-27B**, and **GLM-4.7-Flash** with automatic chat template handling via the tokenizer. Outputs are organized into per-model subdirectories. |
| `judge_synthetic.py` | LLM-as-a-Judge evaluation of generated notes. Scores each note on 5 dimensions (1-5 Likert scale): **clinical realism**, **category alignment**, **clinical coherence**, **note variability**, and **GLP-1 relevance**. Outputs per-note judgments and aggregate statistics. |
| `filter_by_judgment.py` | Joins judge results with original synthetic data and applies configurable score thresholds to keep only high-quality notes. Supports per-dimension minimums, overall average floor, and accept/reject filtering. |
| `diversity_filter.py` | Embedding-based deduplication using **Qwen3-Embedding-8B**. Computes pairwise cosine similarity within each category and greedily removes near-duplicates above a similarity threshold (default: 0.90) to produce a diverse subset. |
| `phase1_data_acquisition.py` | **Phase 1: Data Acquisition & Multilingual Alignment.** Downloads the English baseline (~391K discussions) from Zenodo, builds English + Spanish keyword lists, scrapes Spanish Reddit data via PullPush API, and partitions corpora by language using fastText. |

## Discontinuation Categories

All scripts operate over four GLP-1 RA discontinuation reason categories:

1. **ADVERSE_EFFECTS_OR_TOLERABILITY** — side effects, intolerance
2. **ACCESS_COST_SYSTEM_BARRIERS** — insurance, shortages, pharmacy issues
3. **PATIENT_PREFERENCE_BELIEFS** — personal choice, beliefs, lifestyle
4. **PLANNED_GOAL_BASED_DISCONTINUATION** — reached weight/health goal, tapering plan

## Typical Usage

```bash
# 1. Generate notes (single model)
python code/generate_synthetic.py --model google/gemma-4-31B-it

# 1b. Generate notes (multiple models)
python code/generate_synthetic_multi.py --model Qwen/Qwen3.5-27B

# 2. Judge generated notes
python code/judge_synthetic.py --input synthetic_data/<model>/all_synthetic_notes.jsonl

# 3. Filter by quality scores
python code/filter_by_judgment.py --min-score 3 --min-avg 3.0

# 4. Remove near-duplicates
python code/diversity_filter.py --input filtered_data/all_filtered_notes.jsonl --threshold 0.90
```

## Output Directories

| Directory | Contents |
|---|---|
| `synthetic_data/` | Raw generated notes (per-model subfolders + combined JSONL) |
| `judge_results/` | Per-note judgments and aggregate statistics |
| `filtered_data/` | High-quality notes after judgment-based filtering |
| `diversity_results/` | Deduplicated notes after diversity filtering |
| `prompts/` | Prompt templates for generation |
| `seed_examples/` | Few-shot seed examples per category |
