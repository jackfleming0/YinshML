# E24 Phase 1a — LR sweep runbook (config-only, no build)

The cheap, decisive first rung of the E24 real-self-play-campaign (see
`EXPERIMENT_BACKLOG.md` → E24). **~1 day, ~$30–50, one box.** Answers one question:

> Does *any* real learning rate move iter1_ema's value head and trend H2H up —
> vs **freeze** (LR too low) or **degrade** (LR too high)?

## The hypothesis it tests (the gate-confound)
Our history tangled two failure modes and never separated them:
- **lr 1e-5 → value head frozen** (measured).
- **lr 1e-4 → degradation** — but those runs *also* used a loose **0.20** promotion
  gate that enshrined degraded models, so the spiral compounded. "1e-4 degrades"
  is therefore confounded with "bad gating let it compound."

Phase 1a re-runs the LR axis with a **tight 0.55 gate + `revert_self_play_on_gate_failure`**
so degradation *cannot compound*. If a hotter LR now survives, the plateau was a
regime artifact, not a ceiling — and we never needed the anti-forgetting build.

## What's in this rung
- Three configs, identical to the champion recipe except **`trainer.lr`**:
  `configs/e24_phase1a_lr_3e-5.yaml`, `…_1e-4.yaml`, `…_3e-4.yaml`
  (cosine + warmup, 200 sims, 400 games/iter, 3 iters, gate 0.55 + revert, 15ch).
  Depth is held at champion-default 200 sims **on purpose** — depth is not the
  lever (E19/E22), and it keeps the rung cheap.
- Driver: `scripts/e24_phase1a_sweep.sh` — runs the 3 arms from iter1_ema and,
  per iteration, measures the two signals below.

## Signals (and how to weight them)
1. **PRIMARY — value-head AUC on an ENGINE-LABELED held-out corpus**
   (`scripts/value_head_calibration.py`, Mann-Whitney win-vs-loss AUC). *Did the
   value head actually move?* Use **engine** labels, not the default human corpus —
   the ~0.70 AUC floor is partly human-label noise (a won position lost to a
   blunder), which is exactly the confound this rung must remove.
2. **CONFIRMATORY — H2H vs a frozen iter1_ema** (`measure_h2h.py`, color-balanced
   60). Thin at 3 iters (~1 point/arm) — it reliably catches *binary*
   degrade-and-revert, but not a slow climb. **Weight the AUC trend over H2H here.**

### Value yardstick — use the HUMAN corpus (`expert_games/hvh_full_game_15ch.npz`)
iter1_ema reproduces its known **AUC 0.737 / Brier 0.67** on it, so it's the
representative, reproducible yardstick. Track the per-iter AUC *trend* on it.

> **Resolved 2026-06-04 — the engine-labeled corpus idea backfired.**
> `scripts/gen_engine_labeled_corpus.py` (HeuristicAgent self-play) was meant to
> remove human-label noise, but it gave baseline iter1_ema **AUC 0.575 / Brier
> 1.086 (worse than blind)** — because the *positions* come from myopic heuristic
> play the net never trained on (out-of-distribution), not because the value head
> is bad. The flaw: I made the *labeler* independent but also let it generate the
> *positions*. **Don't use that corpus.** A correct engine corpus would relabel
> *representative* positions (self-play / human) with strong-engine outcomes —
> TODO, not yet built. For now the human corpus is the right call; its label noise
> is a known, bounded limitation, not a showstopper.

## Launch
```bash
# inside tmux/nohup on a >=160-core / 1×GPU>=16GB box:
PY=/venv/main/bin/python VAL_DATA=expert_games/engine_labeled_15ch.npz \
  bash scripts/e24_phase1a_sweep.sh
# single arm:  LR=1e-4 PY=/venv/main/bin/python bash scripts/e24_phase1a_sweep.sh
```

### Smoke first (~5 min — validates the full path before the real run)
The real 15ch encode + `run_training` + the driver's eval/glob/calibration/H2H
wiring can't be tested off-box. Run this once after checkout to shake them out:
```bash
# tiny engine corpus (exercises the real encoder + .npz write)
/venv/main/bin/python scripts/gen_engine_labeled_corpus.py \
    --out /tmp/smoke.npz --games 2 --depth 1 --workers 2 --max-positions 50
# one tiny arm through the SAME driver (1 iter / 4 games / 20 sims)
CFG_OVERRIDE=configs/e24_phase1a_smoke.yaml LR=smoke H2H_GAMES=4 \
    VAL_DATA=/tmp/smoke.npz PY=/venv/main/bin/python \
    bash scripts/e24_phase1a_sweep.sh 2>&1 | tee /tmp/e24_smoke.log
```
Clean smoke ⇒ the full sweep will run. Then do the corpus + sweep above.

Outputs: `auc_e24/lr*_auc.txt` (primary), `h2h_e24/*.json` (+ `e19_summarize.py`
slope table), `runs_e24/lr_*/…/*_ema.pt`.

## Decision gates
| Read | Meaning | Next |
|---|---|---|
| Some LR's **AUC rises** across iters **and** H2H ≥ 45% | a real LR moves it without collapse | **GREEN → Phase 1b**: extend that LR, 15–20 iters, 400–1000 games/iter |
| **AUC degrades + H2H reverts** every iter | forgetting is the blocker | **Phase 2 build**: buffer-mixing + KL-to-anchor, then re-run 1b at that LR |
| **AUC flat at every LR** | maybe stuck — *or maybe the seed* | **Fresher-seed fallback** (below) before concluding |

### Fresher-seed fallback (this session's amendment)
iter1_ema is an over-converged EMA peak (low plasticity). If *all three* LRs read
flat, re-run the best LR from an earlier **pre-over-convergence** checkpoint —
otherwise "the game is stuck" and "this seed won't move" are indistinguishable.
Keep the H2H yardstick `FROZEN` = canonical iter1_ema (don't move the goalposts):
```bash
SEED=models/<fresher_preEMA_checkpoint>.pt TAGSUFFIX=_fresh LR=<best_lr> \
  PY=/venv/main/bin/python bash scripts/e24_phase1a_sweep.sh
```
Only if the fresher seed is *also* flat is "genuinely stuck → bank iter1_ema+E8"
earned.

## Discipline
H2H vs the FIXED champion iter1_ema (R1); one regime change at a time; hard
go/no-go at this gate before spending anything on Phase 1b / Phase 2 / throughput.
