#!/bin/bash
# D.2 autopilot — runs phases sequentially, writes phase sentinels.
# NO auto-stop. Manual termination via vast.ai dashboard only.
# Idempotent: skips phases whose .done sentinel exists.
set -u
set -o pipefail

cd /root/work/YinshML
source venv/bin/activate

STATE=/root/run_state
mkdir -p "$STATE" logs

CONSOLIDATED=/root/work/YinshML/logs/d2_autopilot.log
ts() { date -u +%FT%TZ; }
log() { echo "[$(ts)] $*" | tee -a "$CONSOLIDATED"; }

run_phase() {
    local name="$1"; shift
    local logfile="logs/d2_${name}.log"
    log "=== PHASE: ${name} ==="
    if [[ -f "$STATE/${name}.done" ]]; then
        log "  skip (already done)"
        return 0
    fi
    {
        echo "=== START ${name} at $(ts) ==="
        "$@"
        local rc=$?
        echo "=== END ${name} rc=$rc at $(ts) ==="
        exit $rc
    } 2>&1 | tee -a "$logfile"
    local rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
        log "FAILED phase=${name} rc=${rc}"
        echo "${name} failed rc=${rc} at $(ts)" > "$STATE/STOP_READY"
        exit $rc
    fi
    touch "$STATE/${name}.done"
    log "  done"
}

# Run-dir is timestamped — capture at first launch, then reuse.
run_dir_var() {
    if [[ -f "$STATE/run_dir" ]]; then
        cat "$STATE/run_dir"
    else
        ls -1dt runs_branchD2/2*/ 2>/dev/null | head -1 | sed 's:/$::'
    fi
}

log "===== D.2 AUTOPILOT START ====="
log "git: $(git log --oneline -1)"
log "gpu: $(nvidia-smi -L | head -1)"

# Phase 1: re-encode 6ch → 15ch corpus
run_phase regen \
    python scripts/regenerate_npz_with_enhanced_encoder.py \
        --input  expert_games/yngine_volume.npz \
        --output expert_games/yngine_volume_15ch_mmap/

# Phase 2: supervised pretrain (15-ch corpus, spatial head)
run_phase pretrain \
    python scripts/run_supervised_pretraining.py \
        --data-dir expert_games/yngine_volume_15ch_mmap/ \
        --use-enhanced-encoding \
        --value-head-type spatial \
        --output-dir models/yngine_volume_15ch_pretrain \
        --epochs 6 --batch-size 512 --lr 1e-3 \
        --num-channels 256 --num-blocks 12

# Phase 3: self-play 5-iter D.2 loop
run_phase selfplay \
    python scripts/run_training.py \
        --config configs/branchD2_enhanced_mcts200.yaml \
        --init-checkpoint models/yngine_volume_15ch_pretrain/best_supervised.pt

# Capture run_dir now (created during selfplay)
if [[ ! -f "$STATE/run_dir" ]]; then
    RD=$(ls -1dt runs_branchD2/2*/ 2>/dev/null | head -1 | sed 's:/$::')
    if [[ -n "$RD" ]]; then
        echo "$RD" > "$STATE/run_dir"
        log "captured run_dir=$RD"
    fi
fi

# Phase 4: SPRT screen
run_phase sprt bash -c '
    set -e
    cd /root/work/YinshML
    source venv/bin/activate
    RD=$(cat /root/run_state/run_dir 2>/dev/null || ls -1dt runs_branchD2/2*/ | head -1 | sed "s:/$::")
    if [[ -z "$RD" ]]; then echo "ERROR: no run_dir"; exit 1; fi
    CKPT="$RD/iteration_4/checkpoint_iteration_4_ema.pt"
    if [[ ! -f "$CKPT" ]]; then
        echo "ERROR: iter4 ema ckpt missing: $CKPT"
        ls -la "$RD/" || true
        exit 1
    fi
    echo "SPRT candidate: $CKPT"
    python scripts/eval_vs_frozen_anchor.py \
        --candidate "$CKPT" \
        --anchor models/branchC_volume_pretrain/best_iter_4.pt \
        --sprt --sprt-p1 0.60 --sprt-max-games 400 \
        --device cuda --quiet-mcts \
        --output-json logs/branchD2_iter4_vs_frozen.json
'

# Phase 5: write SUMMARY.md
run_phase summary python - <<'PYEOF'
import json, pathlib, datetime
sprt_path = pathlib.Path("logs/branchD2_iter4_vs_frozen.json")
if not sprt_path.exists():
    raise SystemExit(f"SPRT json missing: {sprt_path}")
data = json.loads(sprt_path.read_text())
# Schema may be a list of candidates or a flat dict — handle both.
record = data[0] if isinstance(data, list) else data
verdict = record.get("verdict") or record.get("sprt_verdict") or "UNKNOWN"
w = record.get("candidate_wins", record.get("wins", 0))
l = record.get("candidate_losses", record.get("losses", 0))
d = record.get("draws", 0)
wr = record.get("win_rate", record.get("wr", None))
ci = record.get("ci_95", record.get("wilson_ci_95", [None, None]))
g = record.get("games_played", record.get("games", w + l + d))
llr = record.get("llr", record.get("log_likelihood_ratio", None))
color = record.get("color_split", record.get("score_by_color", None))
summary = f"""# Branch D.2 — 15-channel encoding result

Run completed: {datetime.datetime.utcnow().isoformat()}Z

## SPRT verdict
- **Verdict:** {verdict}
- **Score (W-L-D):** {w}-{l}-{d}
- **Win rate:** {wr}
- **CI95:** {ci}
- **Games played:** {g}
- **LLR:** {llr}
- **Color split:** {color}

## Interpretation
- STRONGER (lower CI > 0.60): encoding is the lever; freeze + re-anchor + move to D.3 / scale-up
- INCONCLUSIVE WR ~ 0.55: small real edge (Step-2-style); not promotion-worthy
- INCONCLUSIVE WR ~ 0.50 or WEAKER: encoding isn't the binding constraint; pivot to Path 2 or other architecture

Raw SPRT json: logs/branchD2_iter4_vs_frozen.json
"""
pathlib.Path("SUMMARY.md").write_text(summary)
print(summary)
PYEOF

log "===== ALL PHASES COMPLETE ====="
echo "ALL_DONE at $(ts)" > "$STATE/STOP_READY"
log "Wrote STOP_READY sentinel. Box left RUNNING for manual termination."
