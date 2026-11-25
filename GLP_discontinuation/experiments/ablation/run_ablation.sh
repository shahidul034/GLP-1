#!/bin/bash
# ==============================================================
# Ablation Study: Semantic Similarity Threshold Sweep
# ==============================================================
# GPU: CUDA_VISIBLE_DEVICES=2
# Environment: vllm
# ==============================================================

set -e  # Exit on error

SCRIPT_DIR="/home/user/GLP_project/GLP_project_ad/code/ablation_studies"
RESULTS_DIR="$SCRIPT_DIR/results"
MODELS_DIR="$SCRIPT_DIR/models"

echo "============================================================"
echo "Ablation Study: Semantic Similarity Threshold Sweep"
echo "============================================================"
echo "GPU:        CUDA_VISIBLE_DEVICES=3"
echo "Environment: vllm"
echo "Script dir: $SCRIPT_DIR"
echo ""

# Activate conda environment
echo "[1/6] Activating conda environment..."
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate vllm
export CUDA_VISIBLE_DEVICES=2
echo "Done."
echo ""

# ----------------------------------------------------------------
# Phase 1: Embedding Generation
# ----------------------------------------------------------------
echo "============================================================"
echo "[2/6] Phase 1: Embedding Generation"
echo "============================================================"
echo "Model: Qwen/Qwen3-Embedding-8B"
echo "Input: /home/user/GLP_project/GLP_project_ad/data/combined_synthetic/combined.jsonl"
echo "Output: $SCRIPT_DIR/embeddings/"
echo ""
cd "$SCRIPT_DIR"
python generate_embeddings.py
echo ""

# ----------------------------------------------------------------
# Phase 2: Threshold Sweeping and Dataset Construction
# ----------------------------------------------------------------
echo "============================================================"
echo "[3/6] Phase 2: Threshold Sweeping and Dataset Construction"
echo "============================================================"
echo "Thresholds: [1.0, 0.95, 0.90, 0.85, 0.80]"
echo "Output: $SCRIPT_DIR/datasets/"
echo ""
python sweep_thresholds.py
echo ""

# ----------------------------------------------------------------
# Phase 3a: ModernBERT-large Training Sweep
# ----------------------------------------------------------------
echo "============================================================"
echo "[4/6] Phase 3a: ModernBERT-large Training Sweep"
echo "============================================================"
echo "Model: thomas-sounack/BioClinical-ModernBERT-large"
echo "Epochs: 3 | LR: 2e-5 | Batch: 8"
echo "Output: $MODELS_DIR/modernbert_large/"
echo ""
python train_modernbert_ablation.py
echo ""

# ----------------------------------------------------------------
# Phase 3b: Qwen3.5-9B Training Sweep
# ----------------------------------------------------------------
echo "============================================================"
echo "[5/6] Phase 3b: Qwen3.5-9B Training Sweep"
echo "============================================================"
echo "Model: Qwen/Qwen2.5-9B-Instruct via Unsloth"
echo "Epochs: 3 | LR: 2e-4 | Batch: 2+4grad_accum"
echo "Output: $MODELS_DIR/qwen3.5-9b/"
echo ""
python train_qwen_ablation.py
echo ""

# ----------------------------------------------------------------
# Phase 4: Evaluation
# ----------------------------------------------------------------
echo "============================================================"
echo "[6/6] Phase 4: Evaluation"
echo "============================================================"
echo "Test set: datasets/test_gpt54_200.jsonl (200 notes)"
echo "Encoder inference: transformers Trainer"
echo "Decoder inference: Unsloth (no vLLM)"
echo ""
python evaluate_ablation.py
echo ""

# ----------------------------------------------------------------
# Phase 5: Generate Report
# ----------------------------------------------------------------
echo "============================================================"
echo "[7/6] Phase 5: Generate Report"
echo "============================================================"
python generate_report.py
echo ""

# ----------------------------------------------------------------
# Final Summary
# ----------------------------------------------------------------
echo "============================================================"
echo "Ablation Study Complete"
echo "============================================================"
echo ""
echo "Output files:"
echo "  - Embeddings:       $SCRIPT_DIR/embeddings/"
echo "  - Datasets:         $SCRIPT_DIR/datasets/"
echo "  - Models:           $MODELS_DIR/"
echo "  - Filtering logs:   $RESULTS_DIR/filtering_reports/"
echo "  - Semantic leakage: $RESULTS_DIR/filtering_reports/semantic_leakage_report.json"
echo "  - Eval results:     $RESULTS_DIR/evaluation_results.json"
echo "  - Final report:     /home/user/GLP_project/GLP_project_ad/results_summary/results/ablation_study_results.md"
echo ""
echo "Trained models:"
echo "  ModernBERT-large: $MODELS_DIR/modernbert_large/threshold_{1.0,0.95,0.90,0.85,0.80}/"
echo "  Qwen3.5-9B:        $MODELS_DIR/qwen3.5-9b/threshold_{1.0,0.95,0.90,0.85,0.80}/"
echo ""

# Print results table if available
if [ -f "$RESULTS_DIR/modernbert_metrics.json" ]; then
    echo "ModernBERT-large Results:"
    python3 -c "
import json
with open('$RESULTS_DIR/modernbert_metrics.json') as f:
    data = json.load(f)
print(f'  Threshold   Train N   Macro-F1    Accuracy   Time(s)')
print(f'  ---------   -------   --------    --------   -------')
for th, res in data.items():
    m = res.get('metrics', {})
    print(f'  {th:<11} {res[\"train_samples\"]:<9} {m.get(\"eval_f1_macro\", 0):<10.4f} {m.get(\"eval_accuracy\", 0):<10.4f} {res.get(\"training_time_seconds\", \"N/A\")}')
"
    echo ""
fi

echo "Done."