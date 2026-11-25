#!/bin/bash
# Run all evaluations on the final benchmark
# Usage: bash experiments/scripts/run_evaluation.sh

set -e

TEST_FILE="data/final/merged.jsonl"
PROMPT_DIR="prompts"
RESULTS_DIR="results/evaluation"

mkdir -p "$RESULTS_DIR"

echo "=== GLP-1 Benchmark Evaluation ==="
echo "Test file: $TEST_FILE"
echo ""

# 1. Few-shot evaluation on multiple models
echo "--- Few-Shot Evaluation ---"
python src/evaluation/evaluate_few_shot_vllm.py \
    --model "Qwen/Qwen3.5-9B" \
    --test-file "$TEST_FILE" \
    --prompt-template "$PROMPT_DIR/evaluation/inference_prompt_template.md" \
    --output "$RESULTS_DIR/few_shot_qwen3.5-9b.json"

# 2. Fine-tuned model evaluation
echo "--- Fine-tuned Evaluation ---"
MODELS=(
    "models/qwen3.5-9b-finetune"
    "models/llama3.2-3b-finetune"
    "models/phi4-finetune"
    "models/qwen3-4b-finetune"
)

for MODEL_PATH in "${MODELS[@]}"; do
    if [ -d "$MODEL_PATH" ]; then
        echo "Evaluating: $MODEL_PATH"
        python src/finetuning/evaluate/evaluate_final.py \
            --model-path "$MODEL_PATH" \
            --test-file "$TEST_FILE" \
            --output-dir "$RESULTS_DIR"
    else
        echo "Skipping $MODEL_PATH (not found)"
    fi
done

# 3. ModernBERT evaluation
echo "--- ModernBERT Encoder Evaluation ---"
python src/encoding/evaluate_base_models.py \
    --test-file "$TEST_FILE" \
    --output-dir "$RESULTS_DIR/modernbert/"

echo "All evaluations complete."
