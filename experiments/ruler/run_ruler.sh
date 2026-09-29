#!/usr/bin/env bash
# Tables 4 and 5: RULER and the confusable-key tasks.
# Scores land in results/nkc_root_n500/<model>/synthetic/<length>/pred/summary.csv
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RULER_DIR="${SELF_DIR}/RULER"
RESULTS_DIR="${SELF_DIR}/results/nkc_root_n500"

mapfile -t ROWS < <(tail -n +2 "${SELF_DIR}/models.csv")
TASKS=(niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2
       niah_multikey_3 niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2
       nkc_disjoint nkc_shared1 nkc_shared2 nkc_shared)

[ -s "${RULER_DIR}/scripts/data/synthetic/json/hotpotqa.json" ] || python "${SELF_DIR}/fetch_ruler_data.py"

export RULER_ROOT_DIR="$RESULTS_DIR"
export RULER_NUM_SAMPLES=500
export RULER_FA2_SINK_PATCH=1
cd "${RULER_DIR}/scripts"
for row in "${ROWS[@]}"; do
    IFS=, read -r model len <<< "$row"
    pred_dir="${RESULTS_DIR}/${model}/synthetic/${len}/pred"
    for task in "${TASKS[@]}"; do
        # skip tasks already scored
        grep -m1 '^Tasks,' "${pred_dir}/summary.csv" 2>/dev/null | tr ',' '\n' | grep -Fxq "$task" && continue
        rm -f "${pred_dir}/${task}.jsonl"  # RULER can't resume a partial task
        RULER_SEQ_LENGTHS="$len" RULER_TASKS="$task" bash run.sh "$model" synthetic
    done
done
