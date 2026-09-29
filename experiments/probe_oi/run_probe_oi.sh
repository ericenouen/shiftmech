#!/usr/bin/env bash
# Table 3: linear probes for the ordering ID (index) and the semantic key (object).
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SELF_DIR}/../.." && pwd)"
OUT_DIR="${SELF_DIR}/results/probe_oi"
mkdir -p "$OUT_DIR"

mapfile -t MODELS < <(tail -n +2 "${SELF_DIR}/models.csv")

cd "$REPO_DIR"
for model in "${MODELS[@]}"; do
    for target in index object; do
        out="${OUT_DIR}/${model##*/}_${target}"
        [[ -s "${out}.csv" ]] && continue
        python "${SELF_DIR}/probe_oi.py" --model-id "$model" --completion --target "$target" --out "$out"
    done
done
