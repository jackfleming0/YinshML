#!/bin/bash
# e24_phase1a_sweep.sh — E24 Phase 1a LR sweep driver (run ON the box).
#
# THE QUESTION: does ANY real LR move iter1_ema's value head and trend H2H up,
# vs freeze (too low) / degrade (too high)? Three arms, ONE variable = trainer.lr
# in {3e-5, 1e-4, 3e-4}. All else = the champion recipe (configs/e24_phase1a_lr_*.yaml),
# warm-started from iter1_ema, with a tight 0.55 gate + revert-on-gate-failure so
# a hotter LR's degradation cannot COMPOUND (the gate-confound test: the old
# "1e-4 degrades" came from runs that ALSO ran a loose 0.20 gate).
#
# PRIMARY SIGNAL = per-iter value-head AUC on an ENGINE-LABELED held-out corpus
# (scripts/value_head_calibration.py). H2H vs frozen iter1_ema is CONFIRMATORY and
# thin at 3 iters (~1 point/arm) — it catches binary degrade-and-revert, not a
# slow climb, so weight the AUC trend over H2H here.
#
# USAGE (inside tmux/nohup on the box):
#   PY=/venv/main/bin/python bash scripts/e24_phase1a_sweep.sh
#   # single arm:           LR=1e-4 bash scripts/e24_phase1a_sweep.sh
#   # fresher-seed fallback: SEED=models/<fresher_preEMA>.pt TAGSUFFIX=_fresh bash ... (only if all 3 flat)
set -euo pipefail

PY="${PY:-/venv/main/bin/python}"
SEED="${SEED:-models/iter1_ema_2026-05-27/iter1_ema.pt}"  # warm-start (override for fresher-seed fallback)
FROZEN="models/frozen_iter1_ema.pt"                        # fixed H2H yardstick (always the canonical champion)
# Engine-labeled held-out corpus for the value-head AUC (PREREQUISITE — see runbook).
# Falls back to the human corpus with a LOUD warning if absent (the ~0.70 AUC floor
# is then partly human-label noise — generate an engine-labeled .npz to remove it).
VAL_DATA="${VAL_DATA:-expert_games/engine_labeled_15ch.npz}"
H2H_GAMES="${H2H_GAMES:-30}"                                # per color; 30+30 = 60
LR="${LR:-3e-5 1e-4 3e-4}"                                  # which arms to run
TAGSUFFIX="${TAGSUFFIX:-}"

mkdir -p logs runs_e24 h2h_e24 auc_e24
[ -f "$FROZEN" ] || cp "$SEED" "$FROZEN"   # NB: for the fresher-seed fallback, keep FROZEN = canonical iter1_ema
if [ ! -f "$VAL_DATA" ]; then
  echo "!!! WARNING: engine-labeled corpus $VAL_DATA not found — falling back to the human corpus."
  echo "!!! The value-AUC floor will then include human-label noise (the confound E24 wants removed)."
  echo "!!! Generate an engine-labeled holdout (.npz of 15ch states + engine outcomes) and set VAL_DATA."
  VAL_DATA="expert_games/hvh_full_game_15ch.npz"
fi

run_arm() {
  local tag="$1"; local cfg="${CFG_OVERRIDE:-configs/e24_phase1a_lr_${tag}.yaml}"
  local savedir="runs_e24/lr_${tag}${TAGSUFFIX}"
  echo "==================================================================="
  echo "ARM lr=${tag}${TAGSUFFIX} ($cfg): seed=${SEED} -> ${savedir}   $(date -u +%H:%M:%SZ)"
  echo "==================================================================="
  if ls "$savedir"/*/iteration_2/checkpoint_iteration_2_ema.pt >/dev/null 2>&1; then
    echo "lr=${tag}: final (iter-2) checkpoint present — skipping training, eval only"
  else
    $PY scripts/run_training.py -c "$cfg" \
        --init-checkpoint "$SEED" --save-dir "$savedir" 2>&1 | tee "logs/e24_lr${tag}${TAGSUFFIX}_train.log"
  fi
}

eval_arm() {
  local tag="$1"; local savedir="runs_e24/lr_${tag}${TAGSUFFIX}"
  echo "=== value-head AUC (engine-labeled) + H2H vs frozen iter1_ema, arm lr=${tag} ==="
  # value-head AUC per iter — the PRIMARY signal (did the value head move?)
  local specs=()
  for ckpt in $(ls "$savedir"/*/iteration_*/checkpoint_iteration_*_ema.pt 2>/dev/null | sort -t_ -k3 -n); do
    local n; n=$(echo "$ckpt" | sed -E 's/.*checkpoint_iteration_([0-9]+)_ema\.pt/\1/')
    specs+=( "lr${tag}_i${n}:${ckpt}" )
  done
  if [ "${#specs[@]}" -gt 0 ]; then
    $PY scripts/value_head_calibration.py --data "$VAL_DATA" \
        "baseline_iter1:${FROZEN}" "${specs[@]}" 2>&1 | tee "auc_e24/lr${tag}${TAGSUFFIX}_auc.txt"
  fi
  # H2H vs frozen iter1_ema per iter — CONFIRMATORY (thin at 3 iters)
  for ckpt in $(ls "$savedir"/*/iteration_*/checkpoint_iteration_*_ema.pt 2>/dev/null | sort -t_ -k3 -n); do
    local n; n=$(echo "$ckpt" | sed -E 's/.*checkpoint_iteration_([0-9]+)_ema\.pt/\1/')
    local t="lr${tag}${TAGSUFFIX}_iter${n}"
    $PY scripts/measure_h2h.py --white "$ckpt" --black "$FROZEN" \
        --white-label "$t" --black-label frozen_iter1 \
        --games "$H2H_GAMES" --seed 20260603 --output "h2h_e24/${t}_as_white.json" 2>&1 | tail -2
    $PY scripts/measure_h2h.py --white "$FROZEN" --black "$ckpt" \
        --white-label frozen_iter1 --black-label "$t" \
        --games "$H2H_GAMES" --seed 20260603 --output "h2h_e24/${t}_as_black.json" 2>&1 | tail -2
  done
}

for tag in $LR; do run_arm "$tag"; eval_arm "$tag"; done

echo "=== H2H SLOPE SUMMARY (confirmatory) ==="
$PY scripts/e19_summarize.py h2h_e24 || true
echo
echo "=== READ THE RESULT (docs/experiments/e24_phase1a.md) ==="
echo "PRIMARY = the value-head AUC trend in auc_e24/*.txt:"
echo "  any LR's AUC rises across iters AND H2H >= 45% -> GREEN, extend that LR (Phase 1b)."
echo "  AUC degrades + H2H reverts every iter           -> forgetting is the blocker -> Phase 2 build."
echo "  AUC flat at every LR                            -> rerun best LR from a FRESHER seed before"
echo "                                                     calling it stuck (SEED=... TAGSUFFIX=_fresh)."
echo "Done $(date -u +%H:%M:%SZ). Pull auc_e24/ h2h_e24/ runs_e24/*/iteration_*/*_ema.pt, then DESTROY the box."
