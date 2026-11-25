# GLP-1 Discontinuation Reason Classification: A Controlled Clinical NLP Benchmark from Synthetic Notes

This repository contains the code, data, and evaluation protocols for our EMNLP 2025 paper.

## Overview

GLP-1 receptor agonist discontinuation is clinically consequential in obesity and type 2 diabetes care, but the primary reason for stopping therapy is usually documented in free-text notes rather than structured medication fields. This creates a critical clinical NLP gap: while the clinical imperative is high, shareable labeled corpora for reason classification remain remarkably scarce. We define and release a controlled synthetic benchmark for GLP-1 discontinuation-reason classification from clinical notes, serving as an intermediate resource when real EHR note sharing is constrained. The benchmark contains **3,234 synthetic outpatient EHR-style notes** organized into a **7-class taxonomy** spanning adverse effects, access barriers, efficacy mismatch, patient preference, planned discontinuation, clinical decision, and other/unknown reasons. We provide a human-reviewed 320-note test set and a common evaluation protocol spanning prompted and fine-tuned models. Baseline performance ranges from 0.804 macro-F1 in the strongest zero-shot setting to 0.933 for the best fine-tuned decoder and 0.974 for a fine-tuned clinical encoder.

### Discontinuation Categories

| Label | Description |
|---|---|
| `ADVERSE_EFFECTS_TOLERABILITY` | GI side effects, systemic symptoms, serious adverse events |
| `ACCESS_COST_SYSTEM_BARRIERS` | Insurance denial, high costs, shortages, formulary restrictions |
| `EFFICACY_EXPECTATION_MISMATCH` | Weight plateau, insufficient results, inadequate glycemic control |
| `PATIENT_PREFERENCE_BELIEFS` | Safety fears, misinformation, stigma, desire for natural approach |
| `PLANNED_GOAL_BASED_DISCONTINUATION` | Reached goal weight, planned tapering, pregnancy planning |
| `CLINICAL_DECISION_MEDICAL_APPROPRIATENESS` | Contraindications, drug interactions, provider-directed switch |
| `OTHER_UNKNOWN` | Reason unclear, lost to follow-up, ambiguous documentation |

---

## Repository Structure

```
GLP_discontinuation/
├── data/
│   ├── seed_examples/          # Few-shot seed examples (1 per category)
│   ├── final/                  # Final train/test splits (3,234 notes)
│   ├── synthetic/              # Generated notes by model
│   ├── human_eval/             # Human evaluation data
│   └── category_definitions.md # Full taxonomy with subcategories
├── prompts/
│   ├── generation/             # Base prompts for synthetic note generation
│   └── evaluation/             # Prompt templates for inference/evaluation
├── src/
│   ├── generation/             # Synthetic note generation scripts
│   ├── evaluation/             # LLM-as-a-Judge, filtering, few-shot eval
│   ├── finetuning/             # Fine-tuning scripts + evaluation
│   │   ├── finetune_code/      # Per-model finetuning scripts
│   │   ├── individual_scripts/ # Standalone finetuning per model
│   │   ├── evaluate/           # Evaluation scripts for finetuned models
│   │   └── training_data/      # Data conversion and prompt templates
│   ├── analysis/
│   │   ├── fidelity/           # BERTScore, diversity, concept density
│   │   ├── diversity_analysis/ # Lexical/semantic diversity, UMAP, demographics
│   │   ├── fairness/           # Fairness evaluation across demographic groups
│   │   ├── topic_modeling/     # Topic clustering and analysis
│   │   ├── ablation/           # Threshold and embedding model ablation
│   │   └── annotation/         # Annotator agreement computation
│   └── encoding/               # ModernBERT fine-tuning
├── experiments/
│   ├── ablation/               # Ablation study results
│   └── scripts/                # Pipeline shell scripts
├── results/
│   ├── evaluation/             # Evaluation logs and judge statistics
│   ├── figures/                # Reproducible figures
│   └── reports/                # Summary reports for all analyses
├── requirements.txt
├── rc.py                       # Runtime configuration
└── README.md
```

---

## Reproduction Steps

### 1. Setup

```bash
pip install -r requirements.txt
```

The project requires access to:
- **vLLM** for batched generation and inference
- **HuggingFace Transformers** for model fine-tuning
- **Unsloth** for memory-efficient fine-tuning
- **OpenAI-compatible API endpoints** for LLM-as-a-Judge

### 2. Data

The final benchmark dataset is in `data/final/`:
- `merged.jsonl` — 3,234 notes (2,914 train + 320 test)
- `oss.jsonl` — 1,635 OSS-model notes
- `gpt5.4.jsonl` — 907 GPT-5.4 notes

The 320-note human-reviewed test set is created by running:

```bash
python src/finetuning/training_data/split_dataset.py
```

### 3. Synthetic Note Generation

```bash
# Single model generation (e.g., Qwen3.5-27B)
python src/generation/generate_synthetic_multi.py \
    --model Qwen/Qwen3.5-27B \
    --output_dir data/synthetic/

# Full multi-category generation
python src/generation/generate_synthetic_all_categories_v2.py \
    --model Qwen/Qwen3.5-27B
```

Prompt templates in `prompts/generation/` are populated with category-specific seed examples from `data/seed_examples/`.

### 4. LLM-as-a-Judge Quality Filtering

```bash
python src/evaluation/judge_synthetic.py \
    --input data/synthetic/generated_notes.jsonl \
    --judge-model gemma-4-31B-it \
    --output-dir results/evaluation/judge_results/
```

### 5. Diversity Filtering

```bash
python src/evaluation/diversity_filter.py \
    --input data/synthetic/filtered_notes.jsonl \
    --threshold 0.95 \
    --output-dir data/synthetic/diversity_filtered/
```

### 6. Fine-tuning

```bash
# Example: fine-tune Qwen3.5-9B
python src/finetuning/finetune_code/qwen3_5_finetune.py \
    --model Qwen/Qwen3.5-9B \
    --train-file data/final/merged.jsonl \
    --output-dir models/qwen3.5-9b-finetune
```

Individual scripts for each model are in `src/finetuning/finetune_code/` and `src/finetuning/individual_scripts/`.

### 7. Evaluation

**Few-shot evaluation:**

```bash
python src/evaluation/evaluate_few_shot_vllm.py \
    --model Qwen/Qwen3.5-9B \
    --test-file data/final/merged.jsonl \
    --prompt-template prompts/evaluation/inference_prompt_template.md
```

**Fine-tuned model evaluation:**

```bash
python src/finetuning/evaluate/evaluate_final.py \
    --model-path models/qwen3.5-9b-finetune \
    --test-file data/final/merged.jsonl
```

**ModernBERT encoder evaluation:**

```bash
python src/encoding/train_modernbert.py \
    --train-file data/final/merged.jsonl \
    --output-dir models/modernbert-finetune
python src/encoding/evaluate_base_models.py
```

### 8. Fidelity Analysis

```bash
cd src/analysis/fidelity
python step2_bertscore.py --input data/final/merged.jsonl
python step3_diversity.py --input data/final/merged.jsonl
python step4_density.py --input data/final/merged.jsonl
```

### 9. Ablation Studies

```bash
cd src/analysis/ablation
bash ../../experiments/ablation/run_ablation.sh
```

---

## Key Results

| Method | Macro F1 |
|---|---|
| Best zero-shot prompt | 0.804 |
| Best few-shot prompt | 0.840 |
| Fine-tuned Qwen3.5-9B | **0.933** |
| Fine-tuned ModernBERT-large | **0.974** |

Full evaluation results in `results/reports/`.

---

## License

This project is released for academic research purposes. The synthetic clinical notes are generated data and do not contain identifiable patient information.
