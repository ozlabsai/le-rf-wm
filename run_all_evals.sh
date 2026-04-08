#!/bin/bash
# Run all evaluation scripts and save results to files (both JSON and text).
# Usage: bash run_all_evals.sh /path/to/test.h5 model_policy [output_dir]
#
# Example:
#   bash run_all_evals.sh /workspace/data/test.h5 lewm_rf_epoch_99 ./results

set -e

DATA_PATH="${1:?Usage: run_all_evals.sh DATA_PATH MODEL_POLICY [OUTPUT_DIR]}"
POLICY="${2:?Usage: run_all_evals.sh DATA_PATH MODEL_POLICY [OUTPUT_DIR]}"
OUT_DIR="${3:-./results}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUT_DIR}/${POLICY}_${TIMESTAMP}"

mkdir -p "$RUN_DIR"

echo "Running all evaluations for policy=${POLICY}"
echo "Results will be saved to ${RUN_DIR}"
echo ""

echo "=== 1/4: Baselines ==="
python eval_baselines.py --data_path "$DATA_PATH" --model_policy "$POLICY" --output_dir "$RUN_DIR" 2>&1 | tee "${RUN_DIR}/baselines.txt"

echo ""
echo "=== 2/4: Diagnostics ==="
python eval_diagnostics.py --data_path "$DATA_PATH" --model_policy "$POLICY" --output_dir "$RUN_DIR" 2>&1 | tee "${RUN_DIR}/diagnostics.txt"

echo ""
echo "=== 3/4: Regimes ==="
python eval_regimes.py --data_path "$DATA_PATH" --model_policy "$POLICY" --output_dir "$RUN_DIR" 2>&1 | tee "${RUN_DIR}/regimes.txt"

echo ""
echo "=== 4/4: Surprise ==="
python eval_surprise.py --data_path "$DATA_PATH" --model_policy "$POLICY" --output_dir "$RUN_DIR" 2>&1 | tee "${RUN_DIR}/surprise.txt"

echo ""
echo "All results saved to ${RUN_DIR}/"
ls -lh "${RUN_DIR}/"
