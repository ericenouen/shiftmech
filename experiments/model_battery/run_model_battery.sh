#!/bin/bash
# Figure 3 / Table 7: mechanism allocation across the model battery.
# Models over ~25B need several GPUs; run.py spreads them over all visible GPUs.
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SELF_DIR}/results"
PLOTS_DIR="${SELF_DIR}/plots"

mapfile -t ROWS < <(tail -n +2 "${SELF_DIR}/models.csv")

dirs=()
for row in "${ROWS[@]}"; do
    IFS=, read -r model layer positions <<< "$row"
    dirs+=("${RESULTS_DIR}/${model//\//__}")
    python "${SELF_DIR}/run.py" --model-id "$model" --layer "$layer" --token-positions $positions
done

mkdir -p "$PLOTS_DIR"
cd "$SELF_DIR"
python plot.py --results-dirs "${dirs[@]}" --out-dir "$PLOTS_DIR"
