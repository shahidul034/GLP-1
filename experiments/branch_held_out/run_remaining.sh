#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=2
BASE="/home/mshahidul/GLP_project/GLP_project_ad/experiments/branch_held_out"

# Wait for any existing finetune to finish
while ps aux | grep -q "[f]inetune_qwen.*gpt54"; do sleep 10; done

# 2. Llama3.2-3B from pooled (764 → test pooled... wait, 1635 → test gpt54)
echo "[$(date)] Starting Llama3.2-3B from pooled..."
/home/mshahidul/miniconda3/envs/vllm/bin/python $BASE/finetune_llama.py \
  --train_file $BASE/data/train_pooled.jsonl \
  --output_dir $BASE/models/llama3.2-3b_from_pooled \
  > $BASE/log_llama_pooled.txt 2>&1
echo "[$(date)] Llama3.2-3B from pooled done."

# 3. Llama3.2-3B from GPT-5.4 (764 → test pooled)
echo "[$(date)] Starting Llama3.2-3B from gpt54..."
/home/mshahidul/miniconda3/envs/vllm/bin/python $BASE/finetune_llama.py \
  --train_file $BASE/data/train_gpt54.jsonl \
  --output_dir $BASE/models/llama3.2-3b_from_gpt54 \
  > $BASE/log_llama_gpt54.txt 2>&1
echo "[$(date)] Llama3.2-3B from gpt54 done."

echo "[$(date)] All training complete."
