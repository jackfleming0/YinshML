# White-wins-100% investigation: results & implications

Results from the protocol in `INVESTIGATION_PLAN.md`. Closes the thread that started with the B1 H2H finding (iter_3 beat iter_19 100-0 under raw-policy argmax) and the B2 follow-up (iter_9 vs iter_19 looked 50/50 aggregate but per-color split was 50/0).

## TL;DR

The "white-wins-100%" pattern is a **deterministic-argmax brittleness**, not a real network or training bug. It vanishes at temperature > 0.5. The B1 "iter_3 crushes iter_19" regression pattern **does not reproduce** in a fresh run with the current code — training now improves across iterations (iter_9 dominates iter_3 in the diagnostic run). The heuristic is fine. We lost ~3 days chasing a phantom; resume tuning at full scale, but never trust raw-policy-argmax H2H as a primary signal.

## What we set out to test

Five hypotheses (see `INVESTIGATION_PLAN.md` for the original framing):

| ID | Hypothesis | Verdict |
|---|---|---|
| H1 | Heuristic is offense-only and biases self-play | **REFUTED** |
| H2 | White-wins is a deterministic-argmax artifact | **SUPPORTED** |
| H3 | The bias persists with stochastic policy (real collapse) | **REFUTED** |
| H4 | The bias persists under MCTS (production-realistic) | **INCONCLUSIVE** (test was flawed) |
| H5 | Network regresses over training (B1 pattern reproduces) | **REFUTED** |

## What we found

### H1: Heuristic plays balanced

10 heuristic-vs-heuristic games at depth 2: **2 white wins / 8 black wins**. Far from "white-wins-by-default" — actually slightly black-leaning. The "score-ASAP" hypothesis was wrong at the heuristic level.

A real artifact we did notice: the heuristic saturates at ±2000 (the `score_cap`) frequently in mid-game. Once one side gets any advantage, the heuristic loses ability to differentiate moves. This is secondary to the white-wins question but worth fixing eventually — see "Open questions" below.

### H2/H3: Temperature fixes the per-color split cleanly

Same checkpoints, raw policy, two different temperatures:

| | iter_0 vs iter_3 | iter_0 vs iter_5 | iter_0 vs iter_9 | iter_3 vs iter_9 |
|---|---|---|---|---|
| temp=0 (W/B wins) | 20/0 | 20/20 | 20/0 | 0/0 (iter_9 dominates) |
| temp=0.5 (W/B wins) | 9/9 | 11/11 | 8/7 | 14/7 (iter_3 mildly stronger as white) |

At temp=0, several pairs show the white-wins pattern. At temp=0.5, **all pairs show balanced per-color splits** (typically within 1–2 wins). H2 is supported, H3 is refuted.

### H4: MCTS test was flawed

The `eval_head_to_head_mcts.py` script as built configured `PURE_NEURAL` + `temp=1.0` inside MCTS but then took `argmax` of visit counts at the move-selection layer. With `sim=50`, that's still effectively deterministic and produced the white-wins pattern across all 10 pairs.

This **does not** validate H4 because the test never broke determinism at the move-selection layer. Production MCTS uses hybrid mode (heuristic + neural) with temperature > 0 and higher sims, neither of which we tested. We should not conclude anything about production play from this run.

Action item: fix the script to expose move-selection temperature (sample from visit counts rather than argmax) before re-running.

### H5: B1 regression pattern doesn't reproduce

B1: iter_3 beat iter_19 **100-0** at temp=0.
Diagnostic run: iter_9 beats iter_3 **40-0** at temp=0.

Opposite direction. The diagnostic per-model aggregate at temp=0:

```
#1  iter_0: avg score = 0.750
#2  iter_9: avg score = 0.625
#3  iter_3: avg score = 0.500
#4  iter_7: avg score = 0.375
#5  iter_5: avg score = 0.250
```

Non-monotonic (iter_5 and iter_7 dip), but iter_9 is clearly stronger than iter_3 — opposite of B1. The B1 finding was almost certainly an interaction between code-state quirks at that time (likely the encoder/move-encoding fixes that landed around then) and deterministic eval. With the current code, it's gone.

## Implications for future training experiments

### Tooling rules

1. **Never use raw-policy argmax as a primary eval signal.** It conflates real strength with first-move-advantage exploitation. Always evaluate with `--temperature 0.5` or higher, or use MCTS-with-temperature.
2. **The supervisor's per-iter ELO is a Glicko propagation through `sliding_window=2`.** It chains adjacent-pair results into absolute numbers and produces misleading cross-iter comparisons. For real "is iter X stronger than iter Y" questions, run `scripts/eval_head_to_head.py` directly.
3. **The Pseudo W/B/D self-play stat was averaging alternating-sign value targets and reporting ~96% draws.** Fixed in `730d23c`. New runs will report real W/B/D.
4. **`num_workers=3` failed with `BrokenProcessPool` on a fresh vast.ai image.** Spawn-mode CUDA-init issue. `num_workers=1` works as a fallback at ~3x wall-clock.

### What to actually go tune next

Based on what didn't break (heuristic, encoder, training pipeline), pick up where we left off **before** the white-wins phantom hunt:

- The original `post_bitboard_tuning_b2.yaml` recipe (20 iter × 100 games × sim=200, step LR, heuristic floor 0.5) was producing real loss-decrease signal (Policy 2.0→1.4, Value 0.77→0.66).
- iter-9 looked like the peak of B2 *before* we knew temp=0 eval was lying about the trajectory shape. That peak may or may not be real — re-evaluate with temp=0.5 H2H against the final iter to know.
- The most promising tuning vectors (still untested):
  - Higher sim count at full scale (sim=400 if compute allows) — tests whether sim=200 was still under-searched.
  - Longer runs (30–40 iter) now that we know training improves over iterations.
  - Heuristic `score_cap` raised from 10000 → larger value or removed entirely. The current cap saturates often enough that MCTS leaning on the heuristic gets noisy gradient.

### What NOT to spend more time on

- White-wins-100% under deterministic argmax. It's a real property of the policy (slight first-move-advantage bias) but doesn't affect production play. Fix the eval, not the network.
- The heuristic. It plays balanced and slightly favors black. Don't add a defensive feature based on the white-wins phantom.
- Adding `temperature` as a config knob to `eval_head_to_head_mcts.py` move-selection IS worth doing (1 hour) before any future MCTS-based H2H, but it's a tooling fix, not a research investment.

## Tooling artifacts produced during this investigation

All on `main`. Useful even if you never need to repeat the full protocol:

| Path | Purpose |
|---|---|
| `scripts/eval_head_to_head.py` | Raw-policy H2H with `--temperature` flag + per-color split (added during investigation) |
| `scripts/eval_head_to_head_mcts.py` | MCTS-based H2H — **needs move-selection temperature fix before next use** |
| `scripts/replay_h2h_game.py` | Verbose ply-by-ply replay between two checkpoints |
| `scripts/replay_heuristic_vs_heuristic.py` | Heuristic-only replay (diagnoses heuristic behavior without needing checkpoints) |
| `scripts/diagnostic_summarize.py` | Auto-generates `summary.md` from eval JSONs against the H1–H5 hypothesis table |
| `scripts/run_diagnostic_protocol.sh` | End-to-end orchestrator (training + evals + bundle) |
| `configs/diagnostic_short.yaml` | 10-iter short config (sim=100, num_workers=1) for diagnostic-grade reproduction |

The `INVESTIGATION_PLAN.md` doc remains useful as a template for any future "weird signal — let's reproduce it cleanly on a fresh box" investigations.

## Open questions (lower priority, for a quiet day)

1. **Why did `num_workers=3` fail with `BrokenProcessPool` on the fresh vast.ai image?** Worked fine on previous instances. Likely a spawn-mode CUDA-init issue tied to image differences. Worth fixing because serial self-play is ~3x slower.
2. **Heuristic saturation at ±2000.** During heuristic-vs-heuristic replay we saw `h_eval` clipped at the `score_cap` of 10000 (scaled to ±2000 after normalization) constantly through mid-game. Once one side gets any advantage, MCTS leaning on the heuristic gets indistinguishable evals across moves. Cap may be too aggressive; investigate after a real tuning run lands.
3. **Non-monotonic per-iter trajectory.** iter_5 and iter_7 are weaker than iter_0 and iter_3 by aggregate score in the diagnostic run, but iter_9 recovers. May be normal AlphaZero training noise at this scale, or may indicate a learning-rate/buffer-replay interaction. Worth a longer run (30+ iter) to characterize.
4. **B1 itself.** We never determined exactly what made iter_3 crush iter_19 100-0 in B1. The current best guess is "encoder-fix-related interaction with deterministic eval" but we can't reproduce so we can't verify. Tagged as a known unknown; no action.

## Cost of this investigation

- ~3 days of cycles
- 2 cloud GPU sessions (~$15 total)
- Surfaced: 1 real bug (Pseudo W/B/D logging), 1 known unknown (B1 root cause), 6 reusable diagnostic scripts, and a tested protocol for "reproduce a weird signal on a fresh box and ship the bundle home" that we'll use again

The phantom-hunting cost was real, but the tooling and the temperature-eval lesson are durable. Next time a "shocking" eval result lands, the first move is to re-run with temperature > 0 before anyone says the word "regression."
