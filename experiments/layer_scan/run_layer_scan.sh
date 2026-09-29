#!/bin/bash
# Layer scans for Table 6 (Appendix B).
# Each model gets a messiness-0 and a messiness-4 scan. Models over ~25B need one large GPU.
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SELF_DIR}/../.." && pwd)"

mapfile -t ROWS < <(tail -n +2 "${SELF_DIR}/models.csv")

cd "$REPO_DIR"
for row in "${ROWS[@]}"; do
    IFS=, read -r model positions <<< "$row"
    for mess in 0 4; do
        python "${SELF_DIR}/layer_scan.py" --model-id "$model" --messiness "$mess" --token-positions $positions
    done
done
