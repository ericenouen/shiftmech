#!/usr/bin/env bash
# Table 1: mechanism allocation of the pre-training checkpoints, after the layer scans (Figure 4).
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SELF_DIR}/../.." && pwd)"
RESULTS_DIR="${SELF_DIR}/results"

mapfile -t MODELS < <(tail -n +2 "${SELF_DIR}/models.csv")

for model in "${MODELS[@]}"; do
    for mess in 0 4; do
        python "${REPO_DIR}/experiments/layer_scan/layer_scan.py" --model-id "$model" --messiness "$mess" --completion
    done
done

for model in "${MODELS[@]}"; do
    python "${REPO_DIR}/experiments/model_battery/run.py" \
        --model-id "$model" --layer 15 --schemas boxes --cat-to-query 1 \
        --num-samples 10000 --completion --tag clozescan10k \
        --out-dir "${RESULTS_DIR}/${model//\//__}"
done

python "${SELF_DIR}/triple_final.py" | tee "${RESULTS_DIR}/table1.txt"
