#!/bin/bash
# e19_dualarm.sh — E19 Rung-1 dual-arm depth experiment driver (run ON the box).
#
# Sequential on one >=160-core single-GPU box. Both arms share configs/e19_rung1_depth.yaml
# (the ONLY difference is the seed model). After each arm's 3 iters, every per-iteration
# EMA checkpoint is H2H'd vs a FROZEN copy of iter1_ema, color-balanced (2x30 games), and
# scored. Compare A's slope (starts ~50% vs frozen-self) against B's (starts ~27%).
#
# USAGE (inside a screen/tmux session on the box):
#   PY=/venv/main/bin/python bash scripts/e19_dualarm.sh
#
# Resumable-ish: training uses --save-dir per arm; re-running re-evaluates existing
# checkpoints. To run a single arm:  ARM=A bash scripts/e19_dualarm.sh
set -euo pipefail

PY="${PY:-/venv/main/bin/python}"
CFG="configs/e19_rung1_depth.yaml"
SEED_A="models/iter1_ema_2026-05-27/iter1_ema.pt"
SEED_B="models/symmetry_run/symmetric-15ch-iter1-ema.pt"
FROZEN="models/frozen_iter1_ema.pt"          # the fixed yardstick for BOTH arms
H2H_GAMES="${H2H_GAMES:-30}"                  # per color; 30+30 = 60 color-balanced
ARM="${ARM:-AB}"                             # A, B, or AB (both)

mkdir -p logs runs_e19 h2h_e19
[ -f "$FROZEN" ] || cp "$SEED_A" "$FROZEN"   # frozen yardstick = an immutable iter1_ema copy

run_arm() {
  local arm="$1" seed="$2"
  local savedir="runs_e19/arm${arm}"
  echo "==================================================================="
  echo "ARM ${arm}: seed=${seed}  ->  ${savedir}   $(date -u +%H:%M:%SZ)"
  echo "==================================================================="
  [ -f "$seed" ] || { echo "MISSING seed: $seed"; exit 1; }
  if ls "$savedir"/*/iteration_3/checkpoint_iteration_3_ema.pt >/dev/null 2>&1; then
    echo "arm ${arm}: iter-3 checkpoint already present — skipping training, going to eval"
  else
    $PY scripts/run_training.py -c "$CFG" \
        --init-checkpoint "$seed" \
        --save-dir "$savedir" 2>&1 | tee "logs/e19_arm${arm}_train.log"
  fi
}

eval_arm() {
  local arm="$1"
  local savedir="runs_e19/arm${arm}"
  echo "=== H2H eval arm ${arm} vs frozen iter1_ema (color-balanced ${H2H_GAMES}+${H2H_GAMES}) ==="
  # iter0 = the seed itself; iters 1..3 = per-iteration EMA checkpoints.
  for ckpt in $(ls "$savedir"/*/iteration_*/checkpoint_iteration_*_ema.pt 2>/dev/null | sort -t_ -k3 -n); do
    local n; n=$(echo "$ckpt" | sed -E 's/.*checkpoint_iteration_([0-9]+)_ema\.pt/\1/')
    local tag="arm${arm}_iter${n}"
    echo "--- $tag : $ckpt ---"
    # challenger as WHITE vs frozen BLACK
    $PY scripts/measure_h2h.py --white "$ckpt" --black "$FROZEN" \
        --white-label "$tag" --black-label frozen_iter1 \
        --games "$H2H_GAMES" --seed 20260601 \
        --output "h2h_e19/${tag}_as_white.json" 2>&1 | tail -2
    # challenger as BLACK vs frozen WHITE
    $PY scripts/measure_h2h.py --white "$FROZEN" --black "$ckpt" \
        --white-label frozen_iter1 --black-label "$tag" \
        --games "$H2H_GAMES" --seed 20260601 \
        --output "h2h_e19/${tag}_as_black.json" 2>&1 | tail -2
  done
}

[[ "$ARM" == *A* ]] && { run_arm A "$SEED_A"; eval_arm A; }
[[ "$ARM" == *B* ]] && { run_arm B "$SEED_B"; eval_arm B; }

echo "=== SLOPE SUMMARY ==="
$PY scripts/e19_summarize.py h2h_e19 || true
echo "Done $(date -u +%H:%M:%SZ). Pull h2h_e19/ + runs_e19/*/iteration_*/*_ema.pt, then DESTROY the box."
