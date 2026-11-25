#!/bin/bash
# ===========================================================================
# run_all.sh — Run all fidelity experiments (Steps 2–4)
#
# Usage:
#   cd fidelity_code_claude
#   bash run_all.sh
#
# Override defaults:
#   SYNTHETIC=../train.jsonl MIMIC=../cleaned_mimic_0_1000.json bash run_all.sh
# ===========================================================================

set -euo pipefail

# Configurable paths (override via environment variables)
SYNTHETIC="${SYNTHETIC:-./train.jsonl}"
MIMIC="${MIMIC:-./cleaned_mimic_combined.json}"
SAMPLE_SIZE="${SAMPLE_SIZE:-1000}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
SEED="${SEED:-42}"

echo "========================================================"
echo "  Fidelity Experiments — Synthetic GLP-1 Dataset"
echo "========================================================"
echo "  Synthetic  : $SYNTHETIC"
echo "  MIMIC      : $MIMIC"
echo "  Sample size: $SAMPLE_SIZE"
echo "  Output dir : $OUTPUT_DIR"
echo "  Seed       : $SEED"
echo "========================================================"

echo ""
echo "──────────────────────────────────────────────────────────"
echo "  [Step 2] Stylistic Fidelity via BERTScore"
echo "──────────────────────────────────────────────────────────"
python step2_bertscore.py \
    --synthetic "$SYNTHETIC" \
    --mimic "$MIMIC" \
    --sample_size "$SAMPLE_SIZE" \
    --output_dir "$OUTPUT_DIR" \
    --seed "$SEED"

echo ""
echo "──────────────────────────────────────────────────────────"
echo "  [Step 3] Lexical Diversity (Mode Collapse Check)"
echo "──────────────────────────────────────────────────────────"
python step3_diversity.py \
    --synthetic "$SYNTHETIC" \
    --mimic "$MIMIC" \
    --sample_size 0 \
    --output_dir "$OUTPUT_DIR" \
    --seed "$SEED"

echo ""
echo "──────────────────────────────────────────────────────────"
echo "  [Step 4] Clinical Concept Density (Entity Extraction)"
echo "──────────────────────────────────────────────────────────"
python step4_density.py \
    --synthetic "$SYNTHETIC" \
    --mimic "$MIMIC" \
    --sample_size "$SAMPLE_SIZE" \
    --model en_core_sci_md \
    --output_dir "$OUTPUT_DIR" \
    --seed "$SEED"

echo ""
echo "========================================================"
echo "  All fidelity experiments completed successfully!"
echo "  Results saved to: $OUTPUT_DIR/"
echo "========================================================"
