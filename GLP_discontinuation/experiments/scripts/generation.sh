CUDA_VISIBLE_DEVICES=1 python code/generate_synthetic_missing.py \
  --model google/gemma-4-31B-it \
  --num-examples 1000 \
  --batch-size 25 \
  --temperature 0.9 \
  --top-p 0.95 \
  --gpu-memory-utilization 0.92 \
  --tensor-parallel-size 1 \
  --seed 42 \
  --output-dir /home/user/GLP_project/GLP_project_ad/data/synthetic_data/new_data_New_cat

# Qwen3.5-27B
CUDA_VISIBLE_DEVICES=3 python code/generate_synthetic_missing.py \
  --model Qwen/Qwen3.5-27B \
  --num-examples 100 \
  --batch-size 25 \
  --temperature 0.9 \
  --top-p 0.95 \
  --gpu-memory-utilization 0.92 \
  --tensor-parallel-size 1 \
  --seed 42 \
  --output-dir /home/user/GLP_project/GLP_project_ad/data/synthetic_data/new_data_New_cat

# Qwen3.6-27B
CUDA_VISIBLE_DEVICES=3 python code/generate_synthetic_missing.py \
  --model Qwen/Qwen3.6-27B \
  --num-examples 100 \
  --batch-size 25 \
  --temperature 0.9 \
  --top-p 0.95 \
  --gpu-memory-utilization 0.92 \
  --tensor-parallel-size 1 \
  --seed 42 \
  --output-dir /home/user/GLP_project/GLP_project_ad/data/synthetic_data/new_data_New_cat

# GLM-4.7-Flash
CUDA_VISIBLE_DEVICES=3 python code/generate_synthetic_missing.py \
  --model zai-org/GLM-4.7-Flash \
  --num-examples 100 \
  --batch-size 25 \
  --temperature 0.9 \
  --top-p 0.95 \
  --gpu-memory-utilization 0.92 \
  --tensor-parallel-size 1 \
  --seed 42 \
  --output-dir /home/user/GLP_project/GLP_project_ad/data/synthetic_data/new_data_New_cat


# Qwen/Qwen3.5-27B

CUDA_VISIBLE_DEVICES=1 python code/generate_synthetic_missing.py \
  --model Qwen/Qwen3.6-27B \
  --num-examples 100 \
  --batch-size 25 \
  --temperature 0.9 \
  --top-p 0.95 \
  --gpu-memory-utilization 0.92 \
  --tensor-parallel-size 1 \
  --seed 42 \
  --output-dir /home/user/GLP_project/GLP_project_ad/data/synthetic_data/new_data_New_cat

python /home/user/GLP_project/rc.py --g 3 --p 0.8