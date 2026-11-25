#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate vllm
elif command -v mamba &> /dev/null; then
    eval "$(mamba shell hook --shell bash)"
    mamba activate vllm
fi

DATA="${1:-train}"

echo "Running topic modeling pipeline with data: $DATA"
echo "vLLM URL: http://localhost:8091"
echo "Output: results/$DATA/"
echo "GPU: CUDA_VISIBLE_DEVICES=0"
echo ""

python run_pipeline.py --data "$DATA"
