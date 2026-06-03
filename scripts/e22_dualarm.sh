#!/bin/bash
# e22_dualarm.sh — E22 cross-teacher dual-arm driver (run ON the box).
#
# Both arms warm-start from the SAME seed (iter1_ema); the ONLY difference is the
# config — Arm A is mirror self-play, Arm B is cross-teacher (iter1_ema vs sym15).
# After each arm's 5 iters, every per-iteration EMA checkpoint is H2H'd vs a
# FROZEN copy of iter1_ema, color-balanced (2x30). Compare the SLOPES: if the
# cross arm climbs vs frozen and the mirror arm doesn't, decisive-outcome signal
# (a sharper value target) was the lever.
#
# USAGE (inside tmux/nohup on the box):
#   PY=/venv/main/bin/python bash scripts/e22_dualarm.sh
# Single arm: ARM=A (mirror) or ARM=B (cross).
set -euo pipefail

PY="${PY:-/venv/main/bin/python}"
SEED="models/iter1_ema_2026-05-27/iter1_ema.pt"     # both arms warm-start from this
FROZEN="models/frozen_iter1_ema.pt"                  # fixed H2H yardstick
H2H_GAMES="${H2H_GAMES:-30}"                          # per color; 30+30 = 60
ARM="${ARM:-AB}"

declare -A CFG=( [A]="configs/e22_mirror.yaml" [B]="configs/e22_cross.yaml" )

mkdir -p logs runs_e22 h2h_e22
[ -f "$FROZEN" ] || cp "$SEED" "$FROZEN"

run_arm() {
  local arm="$1" cfg="${CFG[$1]}"
  local savedir="runs_e22/arm${arm}"
  echo "==================================================================="
  echo "ARM ${arm} ($cfg): seed=${SEED} -> ${savedir}   $(date -u +%H:%M:%SZ)"
  echo "==================================================================="
  # iterations are 0-indexed: a 5-iter run writes iteration_0..4 (last = _4).
  if ls "$savedir"/*/iteration_4/checkpoint_iteration_4_ema.pt >/dev/null 2>&1; then
    echo "arm ${arm}: final (iter-4) checkpoint present — skipping training, eval only"
  else
    $PY scripts/run_training.py -c "$cfg" \
        --init-checkpoint "$SEED" \
        --save-dir "$savedir" 2>&1 | tee "logs/e22_arm${arm}_train.log"
  fi
}

eval_arm() {
  local arm="$1"
  local savedir="runs_e22/arm${arm}"
  echo "=== H2H eval arm ${arm} vs frozen iter1_ema (color-balanced ${H2H_GAMES}+${H2H_GAMES}) ==="
  for ckpt in $(ls "$savedir"/*/iteration_*/checkpoint_iteration_*_ema.pt 2>/dev/null | sort -t_ -k3 -n); do
    local n; n=$(echo "$ckpt" | sed -E 's/.*checkpoint_iteration_([0-9]+)_ema\.pt/\1/')
    local tag="arm${arm}_iter${n}"
    echo "--- $tag : $ckpt ---"
    $PY scripts/measure_h2h.py --white "$ckpt" --black "$FROZEN" \
        --white-label "$tag" --black-label frozen_iter1 \
        --games "$H2H_GAMES" --seed 20260602 \
        --output "h2h_e22/${tag}_as_white.json" 2>&1 | tail -2
    $PY scripts/measure_h2h.py --white "$FROZEN" --black "$ckpt" \
        --white-label frozen_iter1 --black-label "$tag" \
        --games "$H2H_GAMES" --seed 20260602 \
        --output "h2h_e22/${tag}_as_black.json" 2>&1 | tail -2
  done
}

[[ "$ARM" == *A* ]] && { run_arm A; eval_arm A; }
[[ "$ARM" == *B* ]] && { run_arm B; eval_arm B; }

echo "=== SLOPE SUMMARY ==="
$PY scripts/e19_summarize.py h2h_e22 || true   # same parser (armA/armB tags)
echo "Done $(date -u +%H:%M:%SZ). Pull h2h_e22/ + runs_e22/*/iteration_*/*_ema.pt, then DESTROY the box."
