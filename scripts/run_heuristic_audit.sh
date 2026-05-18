#!/bin/bash
# Run a heuristic-vs-heuristic audit corpus and stash the parquet
# output under self_play_data/<run_name>/.
#
# Designed to be the canonical "first run before you commit to viz/
# heuristic improvements" command. See yinsh_ml/viz/README.md
# "Running a real audit corpus" for the decision tree on what the
# resulting numbers mean.
#
# Usage:
#   scripts/run_heuristic_audit.sh                          # default v1
#   scripts/run_heuristic_audit.sh my_run_name              # custom name
#   NUM_GAMES=50 scripts/run_heuristic_audit.sh smoke       # override count
#   DEPTH_MIX=3:100 WORKERS=4 scripts/run_heuristic_audit.sh

set -euo pipefail

# ---- Knobs (env-overridable) ------------------------------------------------
RUN_NAME="${1:-audit_v1}"
NUM_GAMES="${NUM_GAMES:-200}"
DEPTH_MIX="${DEPTH_MIX:-2:20,3:60,4:20}"
TIME_LIMIT_SEC="${TIME_LIMIT_SEC:-2.0}"
MAX_MOVES="${MAX_MOVES:-200}"
WORKERS="${WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"        # 1 = one parquet/game, live-mode friendly
EPSILON="${EPSILON:-0.05}"           # trajectory diversity
SEED_BASE="${SEED_BASE:-$(date +%s)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-self_play_data}"

OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

# ---- Echo & run -------------------------------------------------------------
cat <<EOF
=== heuristic audit run ===
  output dir : ${OUTPUT_DIR}
  num games  : ${NUM_GAMES}
  depth mix  : ${DEPTH_MIX}
  time / mv  : ${TIME_LIMIT_SEC}s
  max moves  : ${MAX_MOVES}
  workers    : ${WORKERS}
  batch size : ${BATCH_SIZE}  (1 = live mode)
  epsilon    : ${EPSILON}
  seed base  : ${SEED_BASE}

Tip: in another terminal run
    streamlit run scripts/dashboard_games.py
and point the sidebar at ${OUTPUT_DIR}/parquet_data — enable Auto-refresh.

EOF

mkdir -p "${OUTPUT_DIR}"

python scripts/generate_heuristic_games.py \
    --output-dir "${OUTPUT_DIR}" \
    --num-games "${NUM_GAMES}" \
    --depth-mix "${DEPTH_MIX}" \
    --time-limit-sec "${TIME_LIMIT_SEC}" \
    --max-moves "${MAX_MOVES}" \
    --workers "${WORKERS}" \
    --batch-size "${BATCH_SIZE}" \
    --epsilon "${EPSILON}" \
    --seed-base "${SEED_BASE}" \
    --game-id-prefix "${RUN_NAME}" \
    --log-every 10 \
    2>&1 | tee "${OUTPUT_DIR}/run.log"

echo
echo "Done. Parquet output in ${OUTPUT_DIR}/parquet_data/"
echo "Run.log saved to ${OUTPUT_DIR}/run.log"
