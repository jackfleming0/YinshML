#!/usr/bin/env bash
# Phase C ablation sweep launcher.
#
# Runs all 5 ablation configs sequentially under the same tmux session so an
# ssh disconnect doesn't stop anything. Each config writes to its own log
# file. If any single config errors out mid-sweep, the next config still runs
# — we'd rather have partial data than zero data.
#
# Usage (from /workspace/YinshML on the cloud box):
#   bash scripts/run_phase_c.sh [iterations]
#
# `iterations` defaults to 10 (the plan §4.1 value). Pass a smaller number
# to do a shorter sweep:
#   bash scripts/run_phase_c.sh 6

set -o pipefail

ITERATIONS="${1:-10}"

CONFIGS=(
    "ablation_baseline"
    "ablation_buffer100k_epochs2"
    "ablation_games150"
    "ablation_curriculum_slow"
    "ablation_lr_conservative"
)

LOG_DIR="cloud_logs/phase_c"
PROGRESS_LOG="${LOG_DIR}/_progress.log"
mkdir -p "${LOG_DIR}"

echo "============================================================" | tee -a "${PROGRESS_LOG}"
echo "Phase C sweep started $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${PROGRESS_LOG}"
echo "Iterations per config: ${ITERATIONS}" | tee -a "${PROGRESS_LOG}"
echo "Configs: ${CONFIGS[*]}" | tee -a "${PROGRESS_LOG}"
echo "Log dir: ${LOG_DIR}" | tee -a "${PROGRESS_LOG}"
echo "============================================================" | tee -a "${PROGRESS_LOG}"

# Pull latest on launch — safety net in case we pushed any last-minute fixes
# between the user prepping the box and the sweep actually firing off.
git pull origin clean-slate 2>&1 | tail -3 | tee -a "${PROGRESS_LOG}"

SWEEP_START_EPOCH=$(date +%s)

for cfg in "${CONFIGS[@]}"; do
    CONFIG_PATH="configs/${cfg}.yaml"
    if [[ ! -f "${CONFIG_PATH}" ]]; then
        echo "[WARN] ${CONFIG_PATH} missing; skipping" | tee -a "${PROGRESS_LOG}"
        continue
    fi

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="${LOG_DIR}/${cfg}_${TIMESTAMP}.log"

    echo "" | tee -a "${PROGRESS_LOG}"
    echo "------------------------------------------------------------" | tee -a "${PROGRESS_LOG}"
    echo "START  ${cfg}  at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${PROGRESS_LOG}"
    echo "  log: ${LOG_FILE}" | tee -a "${PROGRESS_LOG}"
    echo "------------------------------------------------------------" | tee -a "${PROGRESS_LOG}"

    CFG_START_EPOCH=$(date +%s)
    # --iterations overrides num_iterations from the YAML. Each config writes
    # to runs_ablation/<cfg>/<timestamp>/ per its save_dir setting.
    python scripts/run_training.py -c "${CONFIG_PATH}" --iterations "${ITERATIONS}" \
        2>&1 | tee "${LOG_FILE}"
    RC=$?
    CFG_END_EPOCH=$(date +%s)
    CFG_ELAPSED_MIN=$(( (CFG_END_EPOCH - CFG_START_EPOCH) / 60 ))

    if [[ ${RC} -eq 0 ]]; then
        echo "DONE   ${cfg}  exit=0  elapsed=${CFG_ELAPSED_MIN}m" | tee -a "${PROGRESS_LOG}"
    else
        echo "FAILED ${cfg}  exit=${RC}  elapsed=${CFG_ELAPSED_MIN}m (continuing)" | tee -a "${PROGRESS_LOG}"
    fi
done

SWEEP_END_EPOCH=$(date +%s)
TOTAL_ELAPSED_HR=$(awk "BEGIN {printf \"%.2f\", (${SWEEP_END_EPOCH} - ${SWEEP_START_EPOCH}) / 3600}")

echo "" | tee -a "${PROGRESS_LOG}"
echo "============================================================" | tee -a "${PROGRESS_LOG}"
echo "Phase C sweep complete $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${PROGRESS_LOG}"
echo "Total wall time: ${TOTAL_ELAPSED_HR}h" | tee -a "${PROGRESS_LOG}"
echo "============================================================" | tee -a "${PROGRESS_LOG}"

# Quick at-a-glance final-iter anchor rates for each config — the primary
# decision variable per the plan.
echo "" | tee -a "${PROGRESS_LOG}"
echo "Anchor win rates (final iteration per config):" | tee -a "${PROGRESS_LOG}"
for cfg in "${CONFIGS[@]}"; do
    RUN_DIR=$(ls -td runs_ablation/${cfg}/*/ 2>/dev/null | head -1)
    if [[ -d "${RUN_DIR}" ]]; then
        FINAL_IT=$(ls -td "${RUN_DIR}"iteration_*/ 2>/dev/null | head -1)
        if [[ -n "${FINAL_IT}" ]]; then
            ANCHOR=$(python3 -c "
import json, sys
try:
    with open('${FINAL_IT}metrics.json') as f:
        m = json.load(f)
    print(f\"{m.get('anchor_win_rate', 'n/a')}\")
except Exception as e:
    print('n/a', file=sys.stderr)
" 2>/dev/null)
            echo "  ${cfg}: anchor_win_rate=${ANCHOR}" | tee -a "${PROGRESS_LOG}"
        fi
    fi
done

echo "============================================================" | tee -a "${PROGRESS_LOG}"
