#!/bin/bash
set -e

EXPERIMENTS_DIR="/home/mshahidul/GLP_project/GLP_project_ad/experiments/branch_held_out"
DATA_DIR="$EXPERIMENTS_DIR/data"
MODELS_DIR="$EXPERIMENTS_DIR/models"
FINETUNE_QWEN="$EXPERIMENTS_DIR/finetune_qwen.py"
FINETUNE_LLAMA="$EXPERIMENTS_DIR/finetune_llama.py"
CONDA_ENV="/home/mshahidul/miniconda3/envs/vllm/bin/python"

export CUDA_VISIBLE_DEVICES=2

echo "=========================================="
echo "Experiment A: Train on Pooled → Test on GPT-5.4"
echo "=========================================="

echo ""
echo "--- Qwen3.5-9B: pooled → gpt54 ---"
$CONDA_ENV $FINETUNE_QWEN \
    --train_file "$DATA_DIR/train_pooled.jsonl" \
    --output_dir "$MODELS_DIR/qwen3.5-9b_from_pooled"

echo ""
echo "--- Llama3.2-3B: pooled → gpt54 ---"
$CONDA_ENV $FINETUNE_LLAMA \
    --train_file "$DATA_DIR/train_pooled.jsonl" \
    --output_dir "$MODELS_DIR/llama3.2-3b_from_pooled"

echo ""
echo "=========================================="
echo "Experiment B: Train on GPT-5.4 → Test on Pooled"
echo "=========================================="

echo ""
echo "--- Qwen3.5-9B: gpt54 → pooled ---"
$CONDA_ENV $FINETUNE_QWEN \
    --train_file "$DATA_DIR/train_gpt54.jsonl" \
    --output_dir "$MODELS_DIR/qwen3.5-9b_from_gpt54"

echo ""
echo "--- Llama3.2-3B: gpt54 → pooled ---"
$CONDA_ENV $FINETUNE_LLAMA \
    --train_file "$DATA_DIR/train_gpt54.jsonl" \
    --output_dir "$MODELS_DIR/llama3.2-3b_from_gpt54"

echo ""
echo "=========================================="
echo "All finetuning complete!"
echo "=========================================="
