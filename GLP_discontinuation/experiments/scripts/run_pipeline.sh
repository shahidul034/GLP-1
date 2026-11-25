#!/bin/bash
# Full pipeline: generate → judge → filter → diversity filter
# Usage: bash experiments/scripts/run_pipeline.sh

set -e

MODEL=${1:-"Qwen/Qwen3.5-27B"}
JUDGE_MODEL=${2:-"gemma-4-31B-it"}
SIM_THRESHOLD=${3:-0.95}
MIN_SCORE=${4:-3}
MIN_AVG=${5:-3.0}

echo "=== GLP-1 Discontinuation Full Pipeline ==="
echo "Generator:       $MODEL"
echo "Judge:           $JUDGE_MODEL"
echo "Diversity thr:   $SIM_THRESHOLD"
echo "Min score:       $MIN_SCORE"
echo "Min avg:         $MIN_AVG"
echo ""

# Step 1: Generate
echo "--- Step 1: Generation ---"
python src/generation/generate_synthetic_multi.py \
    --model "$MODEL" \
    --output-dir data/synthetic/

# Step 2: Judge
echo "--- Step 2: LLM-as-a-Judge ---"
python src/evaluation/judge_synthetic.py \
    --input data/synthetic/raw_notes.jsonl \
    --judge-model "$JUDGE_MODEL" \
    --output-dir results/evaluation/judge_results/

# Step 3: Filter by judgment
echo "--- Step 3: Quality Filtering ---"
python src/evaluation/filter_by_judgment.py \
    --input results/evaluation/judge_results/judgments.jsonl \
    --min-score "$MIN_SCORE" \
    --min-avg "$MIN_AVG" \
    --output-dir data/synthetic/filtered/

# Step 4: Diversity filter
echo "--- Step 4: Diversity Filtering ---"
python src/evaluation/diversity_filter.py \
    --input data/synthetic/filtered/filtered_notes.jsonl \
    --threshold "$SIM_THRESHOLD" \
    --output-dir data/synthetic/diversity_filtered/

echo "Pipeline complete."
