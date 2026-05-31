#!/usr/bin/env bash
# Tiny end-to-end smoke of the symmetry-fixes pipeline: continued-pretrain WITH
# E16 -> a self-play iteration driven by the real config (so the
# symmetric_reg YAML->mode_settings->trainer wiring actually fires). Run before
# the real spend. Exits non-zero on any failure; prints "SMOKE OK" on success.
#
# Usage:  bash scripts/smoke_symmetry_pipeline.sh
# Env:    INIT_CKPT (default models/supervised_2026-05-27/best_supervised.pt)
#         CORPUS    (default expert_games/hvh_full_game_15ch.npz)
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-.}"
export NPY_DISABLE_MACOS_ACCELERATE=1

INIT_CKPT="${INIT_CKPT:-models/supervised_2026-05-27/best_supervised.pt}"
CORPUS="${CORPUS:-expert_games/hvh_full_game_15ch.npz}"
TMP="$(mktemp -d)"
PRETRAIN_OUT="$TMP/pretrain"
TINY="$TMP/tiny_15ch.npz"

echo "=== [0/3] make a tiny 15ch corpus subset from $CORPUS ==="
python - "$CORPUS" "$TINY" <<'PY'
import sys, numpy as np
d = np.load(sys.argv[1]); n = 400
out = {'states': d['states'][:n], 'values': d['values'][:n]}
out['policy_indices' if 'policy_indices' in d.files else 'policies'] = \
    (d['policy_indices'] if 'policy_indices' in d.files else d['policies'])[:n]
np.savez(sys.argv[2], **out)
print(f"  wrote {sys.argv[2]} ({n} positions)")
PY

echo "=== [1/3] continued-pretrain with L2 + E16 (1 epoch, tiny) ==="
python scripts/run_supervised_pretraining.py \
    --data "$TINY" \
    --checkpoint "$INIT_CKPT" \
    --use-enhanced-encoding --value-mode classification --value-head-type spatial \
    --label-smoothing 0.1 --enable-symmetric-reg --symmetric-reg-every-k-steps 1 \
    --epochs 1 --batch-size 64 --lr 5e-5 \
    --output-dir "$PRETRAIN_OUT" 2>&1 | grep -E "regularizer ON|Epoch|Saved|best_supervised|Training complete" || true

PRETRAIN_CKPT="$PRETRAIN_OUT/best_supervised.pt"
[ -f "$PRETRAIN_CKPT" ] || PRETRAIN_CKPT="$PRETRAIN_OUT/supervised_final.pt"
[ -f "$PRETRAIN_CKPT" ] || { echo "FAIL: no pretrain checkpoint produced"; exit 1; }
echo "  pretrain checkpoint: $PRETRAIN_CKPT"

echo "=== [2/3] one self-play iteration via the config (E16 must fire in trainer) ==="
SELFPLAY_LOG="$TMP/selfplay.log"
python scripts/run_training.py \
    --config configs/symmetry_fixes_smoke.yaml \
    --init-checkpoint "$PRETRAIN_CKPT" \
    --save-dir "$TMP/run" > "$SELFPLAY_LOG" 2>&1 || {
        echo "FAIL: self-play iteration crashed. Tail:"; tail -30 "$SELFPLAY_LOG"; exit 1; }

echo "=== [3/3] verify E16 actually fired in the self-play trainer ==="
if grep -qE "E16 sym-reg: kl=" "$SELFPLAY_LOG"; then
    echo "  E16 fired in self-play:"; grep -E "E16 sym-reg: kl=" "$SELFPLAY_LOG" | head -3
else
    echo "FAIL: no 'E16 sym-reg' lines in self-play log — regularizer did not activate."
    echo "  (config->mode_settings->trainer wiring is broken)"; tail -30 "$SELFPLAY_LOG"; exit 1
fi

echo ""
echo "SMOKE OK  (pretrain+E16 ran, self-play iteration completed, E16 active in trainer)"
rm -rf "$TMP"
