#!/usr/bin/env bash
# Tiny end-to-end smoke of the FULL symmetry-fixes pipeline, matching the real
# launch path: build an E10-mixed corpus -> from-scratch pretrain (L1+L2+E16) ->
# a self-play iteration driven by the real config (E16 + E2 must fire). Run
# before the real spend. Exits non-zero on any failure; "SMOKE OK" on success.
#
# Usage:  bash scripts/smoke_symmetry_pipeline.sh
# Env:    ENGINE_CORPUS (default: a subset of hvh_full_game_15ch.npz as stand-in)
#         HUMAN_PLACEMENTS (default expert_games/hvh_placement_only_15ch.npz)
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-.}"
export NPY_DISABLE_MACOS_ACCELERATE=1

HUMAN_PLACEMENTS="${HUMAN_PLACEMENTS:-expert_games/hvh_placement_only_15ch.npz}"
TMP="$(mktemp -d)"
ENGINE="${ENGINE_CORPUS:-$TMP/engine_15ch.npz}"
CORPUS="$TMP/e10_corpus"
PRETRAIN_OUT="$TMP/pretrain"

if [ -z "${ENGINE_CORPUS:-}" ]; then
  echo "=== [0/4] engine stand-in (subset of hvh_full_game_15ch.npz) ==="
  python - "$ENGINE" <<'PY'
import sys, numpy as np
d = np.load('expert_games/hvh_full_game_15ch.npz'); n = 3000
np.savez(sys.argv[1], states=d['states'][:n], policy_indices=d['policy_indices'][:n], values=d['values'][:n])
print(f"  wrote {sys.argv[1]} ({n} positions)")
PY
fi

echo "=== [1/4] build E10-mixed placement corpus ==="
python scripts/build_e10_corpus.py \
    --engine-corpus "$ENGINE" --human-placements "$HUMAN_PLACEMENTS" \
    --output "$CORPUS" --max-placement 800 --max-main-game 1500 --seed 0 2>&1 \
    | grep -E "split:|placement|main-game|Wrote" || true
[ -f "$CORPUS/states.npy" ] || { echo "FAIL: E10 corpus not built"; exit 1; }

echo "=== [2/4] FROM-SCRATCH pretrain on the E10 corpus (L2 + E16) ==="
python scripts/run_supervised_pretraining.py \
    --data-dir "$CORPUS" \
    --use-enhanced-encoding --value-mode classification --value-head-type spatial \
    --label-smoothing 0.1 --enable-symmetric-reg --symmetric-reg-every-k-steps 2 \
    --epochs 1 --batch-size 64 --lr 1e-3 \
    --output-dir "$PRETRAIN_OUT" 2>&1 \
    | grep -E "regularizer ON|Loaded|Epoch|Training complete" || true
PRETRAIN_CKPT="$PRETRAIN_OUT/best_supervised.pt"
[ -f "$PRETRAIN_CKPT" ] || PRETRAIN_CKPT="$PRETRAIN_OUT/supervised_final.pt"
[ -f "$PRETRAIN_CKPT" ] || { echo "FAIL: no pretrain checkpoint produced"; exit 1; }

echo "=== [3/4] one self-play iteration via the config (E16 + E2 must fire) ==="
LOG="$TMP/selfplay.log"
python scripts/run_training.py \
    --config configs/symmetry_fixes_smoke.yaml \
    --init-checkpoint "$PRETRAIN_CKPT" \
    --save-dir "$TMP/run" > "$LOG" 2>&1 || {
        echo "FAIL: self-play iteration crashed. Tail:"; tail -30 "$LOG"; exit 1; }

echo "=== [4/4] verify E16 fired (E2 fires once placement positions accrue) ==="
if grep -qE "E16 sym-reg: kl=" "$LOG"; then
    echo "  E16 fired:"; grep -E "E16 sym-reg: kl=" "$LOG" | head -2
else
    echo "FAIL: no 'E16 sym-reg' lines — regularizer did not activate."; tail -30 "$LOG"; exit 1
fi

echo ""
echo "SMOKE OK  (E10 corpus built, from-scratch pretrain + E16 ran, self-play completed, E16 active)"
rm -rf "$TMP"
