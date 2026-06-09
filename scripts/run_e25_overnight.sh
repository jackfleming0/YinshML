#!/usr/bin/env bash
# E25 binding-constraint diagnostic — full overnight run (local Mac).
# Chains the three steps sequentially so CPU-worker corpus gen and MPS ablation
# never contend. Re-runnable. Logs everything with timestamps.
#
#   bash scripts/run_e25_overnight.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

PY=venv/bin/python
MODEL=models/iter1_ema_2026-05-27/iter1_ema.pt
CORPUS=expert_games/selfplay_strong_15ch.npz
HUMAN=expert_games/hvh_full_game_15ch.npz   # the corpus that produced AUC 0.737
ABL_OUT=docs/experiments/e25_ablation.json
LOG=docs/experiments/e25_run.log

mkdir -p docs/experiments expert_games
echo "=== E25 run started $(date) ===" | tee "$LOG"   # fresh log each run

# --- Part 1a: in-distribution-strong corpus (neural-MCTS self-play, outcome-labeled) ---
# 200 sims = iter1_ema's actual play regime; max-moves 250 prevents runaway games;
# stops early at 8000 positions (the calibration sample size); checkpoints every 10
# games so the run is crash-safe and observable.
echo "[$(date +%H:%M:%S)] STEP 1: corpus gen (200 sims, 8 CPU workers, stop @ 8000 positions)" | tee -a "$LOG"
$PY scripts/gen_selfplay_labeled_corpus.py \
    --model "$MODEL" --out "$CORPUS" \
    --games 250 --sims 200 --workers 8 --device cpu \
    --dirichlet-alpha 0.3 --max-moves 250 \
    --max-positions 8000 --checkpoint-every 10 2>&1 | tee -a "$LOG"

# --- Part 1b: re-measure value-head discrimination on clean vs human corpora ---
echo "[$(date +%H:%M:%S)] STEP 2: value-head calibration (clean corpus)" | tee -a "$LOG"
$PY scripts/value_head_calibration.py --data "$CORPUS" --n 8000 \
    iter1_ema:"$MODEL" 2>&1 | tee -a "$LOG"

echo "[$(date +%H:%M:%S)] STEP 2b: calibration on the HUMAN corpus (the 0.737 baseline)" | tee -a "$LOG"
$PY scripts/value_head_calibration.py --data "$HUMAN" --n 8000 \
    iter1_ema:"$MODEL" 2>&1 | tee -a "$LOG"

# --- Part 2: policy/value ablation H2H (which head bounds played strength) ---
echo "[$(date +%H:%M:%S)] STEP 3: ablation H2H (60 games/pairing @ 200 sims, MPS)" | tee -a "$LOG"
$PY scripts/e25_ablation_h2h.py --model "$MODEL" \
    --games 60 --sims 200 --output "$ABL_OUT" 2>&1 | tee -a "$LOG"

echo "=== E25 run finished $(date) ===" | tee -a "$LOG"
echo "Corpus:     $CORPUS" | tee -a "$LOG"
echo "Ablation:   $ABL_OUT" | tee -a "$LOG"
echo "Full log:   $LOG" | tee -a "$LOG"
