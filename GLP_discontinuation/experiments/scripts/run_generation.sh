#!/bin/bash
# Generation pipeline: create synthetic clinical notes
# Usage: bash experiments/scripts/run_generation.sh

set -e

MODEL=${1:-"Qwen/Qwen3.5-27B"}
OUTPUT_DIR=${2:-"data/synthetic"}

echo "=== GLP-1 Synthetic Note Generation ==="
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo ""

python src/generation/generate_synthetic_multi.py \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR"

echo "Generation complete."
