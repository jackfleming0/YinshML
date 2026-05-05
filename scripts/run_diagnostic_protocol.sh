#!/usr/bin/env bash
# Self-contained diagnostic protocol for the white-wins-100% investigation.
# See INVESTIGATION_PLAN.md for what this is and why.
#
# Usage:  bash scripts/run_diagnostic_protocol.sh [--no-checkpoints]
#
# Behavior:
#   - Sets up env, runs sanity tests, runs short training, runs eval battery,
#     bundles output. Each phase logs to diagnostic_output/<TIMESTAMP>/<phase>.log
#   - Phases write a sentinel file when complete; rerunning the script after
#     a crash skips already-completed phases.
#   - At the end, prints the scp command needed to download the bundle.
#
# Args:
#   --no-checkpoints     Drop checkpoint .pt files from the bundle (saves ~1.3GB).
#                        Default keeps checkpoints — they let you re-run evals later.

set -euo pipefail

INCLUDE_CHECKPOINTS=1
for arg in "$@"; do
    case "$arg" in
        --no-checkpoints) INCLUDE_CHECKPOINTS=0 ;;
        *) echo "Unknown arg: $arg"; exit 2 ;;
    esac
done

ts() { date -u +%Y%m%dT%H%M%S; }
log() { echo "[$(ts)] $*"; }

# --------------------------------------------------------------------------
# Resolve paths and timestamps
# --------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TIMESTAMP="${DIAG_TIMESTAMP:-$(ts)}"
OUT_DIR="$REPO_ROOT/diagnostic_output/$TIMESTAMP"
mkdir -p "$OUT_DIR"

CONFIG_PATH="configs/diagnostic_short.yaml"

log "Diagnostic timestamp: $TIMESTAMP"
log "Output dir: $OUT_DIR"
log "Config: $CONFIG_PATH"
log "Include checkpoints in bundle: $INCLUDE_CHECKPOINTS"

# --------------------------------------------------------------------------
# Phase 0 — environment setup + sanity
# --------------------------------------------------------------------------
PHASE0_DONE="$OUT_DIR/.phase0.done"
if [[ ! -f "$PHASE0_DONE" ]]; then
    log "=== PHASE 0: SETUP ==="
    {
        log "Installing build deps..."
        pip install pybind11 matplotlib --quiet || true
        pip install -r requirements.txt --quiet
        pip install -e . --quiet

        log "Building C++ engine..."
        python setup.py build_ext --inplace 2>&1 | tail -3

        log "Sanity: import CppGameState"
        python -c "from yinsh_ml.game_cpp import CppGameState; print('OK')"

        log "Sanity: run a small subset of game_cpp tests"
        # --timeout requires pytest-timeout plugin; skip gracefully if missing
        pytest yinsh_ml/game_cpp/tests/ -q -x 2>&1 | tail -5 || {
            log "WARN: some game_cpp tests failed — continuing, but watch for engine bugs"
        }
    } 2>&1 | tee "$OUT_DIR/phase0_setup.log"
    touch "$PHASE0_DONE"
    log "Phase 0 complete."
else
    log "Phase 0 already complete (skipping)."
fi

# --------------------------------------------------------------------------
# Phase 1 — diagnostic training run
# --------------------------------------------------------------------------
PHASE1_DONE="$OUT_DIR/.phase1.done"
TRAIN_RUN_DIR_FILE="$OUT_DIR/.training_run_dir"

if [[ ! -f "$PHASE1_DONE" ]]; then
    log "=== PHASE 1: TRAINING (~90 min) ==="
    log "Command: python scripts/run_training.py --config $CONFIG_PATH"

    # The supervisor creates runs/<DATETIME>/. We capture the path it picks
    # by snapshotting before/after.
    BEFORE=$(ls -d runs/*/ 2>/dev/null | sort)
    python scripts/run_training.py --config "$CONFIG_PATH" 2>&1 \
        | tee "$OUT_DIR/training.log"
    AFTER=$(ls -d runs/*/ 2>/dev/null | sort)
    TRAIN_RUN_DIR=$(comm -13 <(echo "$BEFORE") <(echo "$AFTER") | tail -1 | sed 's:/*$::')

    if [[ -z "$TRAIN_RUN_DIR" ]]; then
        log "ERROR: Could not determine training run dir."
        exit 1
    fi
    echo "$TRAIN_RUN_DIR" > "$TRAIN_RUN_DIR_FILE"
    log "Training output: $TRAIN_RUN_DIR"
    touch "$PHASE1_DONE"
    log "Phase 1 complete."
else
    log "Phase 1 already complete (skipping)."
fi

TRAIN_RUN_DIR=$(cat "$TRAIN_RUN_DIR_FILE")

# Pick representative iterations to evaluate. With 10 iters total (0..9),
# sample [0, 3, 5, 7, 9] for a 5-checkpoint matrix (10 pairs each phase).
EVAL_ITERATIONS="0 3 5 7 9"

# --------------------------------------------------------------------------
# Phase 2 — eval battery
# --------------------------------------------------------------------------
PHASE2_DONE="$OUT_DIR/.phase2.done"
if [[ ! -f "$PHASE2_DONE" ]]; then
    log "=== PHASE 2: EVAL BATTERY ==="

    log "  2a) Raw-policy H2H @ temp=0.0 (baseline; tests for white-wins pattern)"
    python scripts/eval_head_to_head.py \
        --run-dir "$TRAIN_RUN_DIR" \
        --iterations $EVAL_ITERATIONS \
        --num-games 40 \
        --temperature 0.0 \
        --device cuda \
        --output-json "$OUT_DIR/eval_h2h_temp0.json" \
        2>&1 | tee "$OUT_DIR/eval_h2h_temp0.log"

    log "  2b) Raw-policy H2H @ temp=0.5 (tests determ.-argmax brittleness)"
    python scripts/eval_head_to_head.py \
        --run-dir "$TRAIN_RUN_DIR" \
        --iterations $EVAL_ITERATIONS \
        --num-games 40 \
        --temperature 0.5 \
        --device cuda \
        --output-json "$OUT_DIR/eval_h2h_temp0.5.json" \
        2>&1 | tee "$OUT_DIR/eval_h2h_temp0.5.log"

    log "  2c) MCTS H2H (production-realistic; tests if pattern affects real play)"
    python scripts/eval_head_to_head_mcts.py \
        --run-dir "$TRAIN_RUN_DIR" \
        --iterations $EVAL_ITERATIONS \
        --num-games 40 \
        --num-simulations 50 \
        --device cuda \
        --output-json "$OUT_DIR/eval_h2h_mcts.json" \
        2>&1 | tee "$OUT_DIR/eval_h2h_mcts.log"

    touch "$PHASE2_DONE"
    log "Phase 2 complete."
else
    log "Phase 2 already complete (skipping)."
fi

# --------------------------------------------------------------------------
# Phase 3 — heuristic sanity
# --------------------------------------------------------------------------
PHASE3_DONE="$OUT_DIR/.phase3.done"
if [[ ! -f "$PHASE3_DONE" ]]; then
    log "=== PHASE 3: HEURISTIC SANITY (10 games) ==="
    python scripts/replay_heuristic_vs_heuristic.py \
        --depth 2 --num-games 10 --seed 200 \
        2>&1 | tee "$OUT_DIR/heuristic_sanity.log" \
        | tail -5
    touch "$PHASE3_DONE"
    log "Phase 3 complete."
else
    log "Phase 3 already complete (skipping)."
fi

# --------------------------------------------------------------------------
# Phase 4 — summary + bundle
# --------------------------------------------------------------------------
log "=== PHASE 4: SUMMARY + BUNDLE ==="

log "Generating summary.md..."
python scripts/diagnostic_summarize.py --output-dir "$OUT_DIR" \
    > "$OUT_DIR/summary.md" || log "WARN: summary generation hit an error"

# Stage artifacts to bundle
STAGING="$OUT_DIR/_bundle_staging"
rm -rf "$STAGING"
mkdir -p "$STAGING"

cp "$OUT_DIR"/*.json "$STAGING/" 2>/dev/null || true
cp "$OUT_DIR"/*.log "$STAGING/" 2>/dev/null || true
cp "$OUT_DIR"/summary.md "$STAGING/" 2>/dev/null || true
cp INVESTIGATION_PLAN.md "$STAGING/" 2>/dev/null || true
cp configs/diagnostic_short.yaml "$STAGING/" 2>/dev/null || true

# Per-iter metrics (small)
mkdir -p "$STAGING/training_run"
cp "$TRAIN_RUN_DIR/manifest_final.json" "$STAGING/training_run/" 2>/dev/null || true
for d in "$TRAIN_RUN_DIR"/iteration_*/; do
    iter_name=$(basename "$d")
    mkdir -p "$STAGING/training_run/$iter_name"
    cp "$d/metrics.json" "$STAGING/training_run/$iter_name/" 2>/dev/null || true
done

# Optionally include checkpoints (drives bundle size)
if [[ "$INCLUDE_CHECKPOINTS" -eq 1 ]]; then
    log "Including checkpoints in bundle (~1.3 GB)..."
    for d in "$TRAIN_RUN_DIR"/iteration_*/; do
        iter_name=$(basename "$d")
        # Only copy the .pt for representative iters; saves space without
        # losing ability to re-run evals on the iters we cared about.
        for it in $EVAL_ITERATIONS; do
            if [[ "$iter_name" == "iteration_$it" ]]; then
                cp "$d"/checkpoint_iteration_*.pt "$STAGING/training_run/$iter_name/" 2>/dev/null || true
            fi
        done
    done
fi

BUNDLE_PATH="$REPO_ROOT/diagnostic_bundle_${TIMESTAMP}.tgz"
log "Creating bundle: $BUNDLE_PATH"
tar -czf "$BUNDLE_PATH" -C "$OUT_DIR" "_bundle_staging" \
    --transform "s|^_bundle_staging|diagnostic_bundle_${TIMESTAMP}|"

BUNDLE_SIZE=$(du -h "$BUNDLE_PATH" | cut -f1)

log "=== DIAGNOSTIC COMPLETE ==="
log "Bundle: $BUNDLE_PATH ($BUNDLE_SIZE)"
log ""
log "To pull locally:"
log "    scp \$CLOUD_HOST:$BUNDLE_PATH ."
log ""
log "Or, if you only want the eval results (skip checkpoints):"
log "    Re-run with: bash scripts/run_diagnostic_protocol.sh --no-checkpoints"
log ""
log "Inside the bundle:"
log "    summary.md          - hypothesis verdicts + recommendation"
log "    eval_h2h_*.json     - raw eval results (per-color split included)"
log "    eval_h2h_*.log      - human-readable eval output"
log "    heuristic_sanity.log - heuristic-vs-heuristic confirmation"
log "    training.log        - full supervisor log from the diagnostic run"
log "    training_run/       - per-iter metrics.json + (optional) checkpoints"
log ""
log "Read summary.md first."
