#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
source ~/miniconda3/bin/activate conda activate vllm

SCRIPT_DIR="/home/user/GLP_project/GLP_project_ad/code/finetune/finetune_code"
MODELS_DIR="/home/user/GLP_project/GLP_project_ad/code/finetune/models"

mkdir -p "$MODELS_DIR"

echo "========================================"
echo "Starting Finetuning Pipeline"
echo "========================================"

# 1. Qwen3-4B
echo "[1/12] Training Qwen3-4B..."
cd "$SCRIPT_DIR"
python qwen3_finetune_local.py
echo "Completed Qwen3-4B"
echo "========================================"

# 2. Qwen3-14B
echo "[2/12] Training Qwen3-14B..."
cd "$SCRIPT_DIR"
python qwen3_reasoning_finetune_local.py
echo "Completed Qwen3-14B"
echo "========================================"

# 3. Llama3.1-8B
echo "[3/12] Training Llama3.1-8B..."
cd "$SCRIPT_DIR"
python llama3_1_8b_finetune_local.py
echo "Completed Llama3.1-8B"
echo "========================================"

# 4. Llama3.2-3B
echo "[4/12] Training Llama3.2-3B..."
cd "$SCRIPT_DIR"
python llama3_2_finetune_local.py
echo "Completed Llama3.2-3B"
echo "========================================"

# 5. Gemma3-4B
echo "[5/12] Training Gemma3-4B..."
cd "$SCRIPT_DIR"
python gemma3_4b_finetune_local.py
echo "Completed Gemma3-4B"
echo "========================================"

# 6. Gemma4-E2B
echo "[6/12] Training Gemma4-E2B..."
cd "$SCRIPT_DIR"
python gemma4_finetune_local.py
echo "Completed Gemma4-E2B"
echo "========================================"

# 7. Phi-4
echo "[7/12] Training Phi-4..."
cd "$SCRIPT_DIR"
python phi4_finetune_local.py
echo "Completed Phi-4"
echo "========================================"

# 8. Qwen3.5 (all 4 sizes)
echo "[8/12] Training Qwen3.5 (all sizes)..."
cd "$SCRIPT_DIR"
python qwen3_5_finetune_local.py
echo "Completed Qwen3.5 (all sizes)"
echo "========================================"

# 9. GPT-OSS-20B
echo "[9/12] Training GPT-OSS-20B..."
cd "$SCRIPT_DIR"
python gpt_oss_20b_finetune_local.py
echo "Completed GPT-OSS-20B"
echo "========================================"

echo "========================================"
echo "All Finetuning Complete!"
echo "========================================"
echo "Models saved to: $MODELS_DIR"
ls -la "$MODELS_DIR"