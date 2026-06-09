# E24 — A REAL self-play campaign (the actually-untested axis)

**Status:** DONE: NOT_STRONGER (Phase 1a ran; LR is not the lever — ladder stops)
**Date(s):** scoped 2026-06-03; Phase 1a RAN 2026-06-04
**Cost:** Phase 1a ~1 day, ~$30-50. (Phase 1b ~2-3 days, ~$80-150; Phases 2/3 not run.)
**Branch / artifacts:** configs `configs/e24_phase1a_lr_{3e-5,1e-4,3e-4}.yaml`; driver `scripts/e24_phase1a_sweep.sh`; runbook `docs/experiments/completed/e24_phase1a.md`; results `docs/experiments/completed/e24_phase1a_results.md`; corpus gen `scripts/gen_engine_labeled_corpus.py`; eval `scripts/value_head_calibration.py`, `scripts/measure_h2h.py`.

> **This file captures the FORWARD write-up rationale** (the multi-phase ladder design, the two-tangled-failure-modes reframe, the Phase 1b/2/3 plans). For the **Phase 1a RESULTS** (the falsification, the per-arm tables, operational lessons), see [`e24_phase1a_results.md`](e24_phase1a_results.md) and the runbook [`e24_phase1a.md`](e24_phase1a.md). This strategic rationale is kept because it is durable even though the LR hypothesis was falsified.

## Description
**Provenance:** consolidates two independent session scopes — the parallel session's LR-sweep-first ladder + gate-confound insight (the spine), and this session's seed-hedge + engine-labeled-eval amendments. Renumbered from a draft "E23" to avoid colliding with main's [[e23]] (opponent league, DROPPED).

**The correction this entry records (2026-06-03, Jack pushed, AI conceded).** We concluded "the self-play loop can't beat iter1_ema" — an **overclaim**. What we actually tested is a *cautious micro-continuation*: lr 1e-5, 3-5 iters, ~200 games/iter (~1K total), from an over-converged EMA optimum, LR pinned low to avoid degrading the warm start. AlphaZero-class self-play is millions of games, real LR schedules, from scratch. **We ran a rounding error of self-play and generalized to "self-play fails for YINSH."** It works for Go/chess/shogi/Hex/Othello; nothing makes YINSH the exception. The "frozen value head" is a consequence of *our* lr=1e-5 timidity, not a property of the game.

**The two tangled failure modes (the key reframe — credit: parallel session).** History conflates two things we never separated:
- **Too-low LR (1e-5) → value head frozen** (measured).
- **Too-high LR (1e-4) → degradation** — BUT those runs *also* ran a loose 0.20 promotion gate that enshrined degraded models, so the spiral compounded. So "1e-4 degrades" is confounded with "bad gating let it compound." With a tight 0.55 Wilson gate + `revert_self_play_on_gate_failure` (rollback so degradation can't compound), high LR may be survivable *without* an anti-forgetting build. **Untested.**

**Hypothesis:** the plateau is an artifact of the cautious regime (low LR + loose gate + tiny scale + over-converged seed), not a ceiling.

## Outcome
**DONE — NOT_STRONGER. LR is not the lever.** Phase 1a (the 3-arm LR sweep) RAN 2026-06-04: AUC never lifts off 0.737 at any LR; more LR = more erosion (3e-4 H2H *collapsed* 53→32%). The "too timid LR" hypothesis is **falsified**, and **Phase 1b/2 are NOT the next step**. The ladder redirects OUT of optimizer-space to the value-TARGET path ([[e21]] ensemble-teacher / value-head architecture). [[e18]] (deploy symmetric MCTS) remains the cheap free win. (Full per-arm numbers and operational lessons in [`e24_phase1a_results.md`](e24_phase1a_results.md).)

## Details (the forward ladder design — durable even though Phase 1a falsified the premise)

**Config vs build.** EXISTS as config: cosine LR + warmup (`lr_schedule`, `warmup_epochs`), `value_head_lr_factor` (5.0), `discrimination_weight`, EMA, tight gate + `revert_self_play_on_gate_failure`, `measure_h2h.py`, NaN guards. BUILD items (Phase 2 only): KL/entropy anchor to the frozen reference (absent), and replay-buffer composition (buffer is pure FIFO, supervisor.py:190-191).

**Cheap-first ladder, each rung gated:**
- **Phase 1a — LR sweep, CONFIG-ONLY, no build (~1 day, ~$30-50).** Warm-start iter1_ema. Three short runs: lr ∈ {3e-5, 1e-4, 3e-4}, cosine+warmup, ~3 iters each, 400 games/iter, tight 0.55 gate + revert. Q: does *any* LR move the value head and trend H2H up, vs freeze (low) / degrade (high)?
  - **Primary signal = per-iter ENGINE-LABELED held-out value AUC/Brier** (NOT human labels — that noise inflated the ~0.70 AUC floor). H2H vs frozen iter1_ema is *confirmatory* and thin at 3 iters (~1 point/arm — catches binary degrade-and-revert, not a slow climb), so weight AUC-trend over H2H here. *(Lesson from the run: the engine-labeled corpus proved OOD; the human corpus was used as primary instead — see results file.)*
  - **Seed hedge (amendment):** iter1_ema is over-converged (EMA peak → low plasticity). If the sweep is flat across *all* LRs, re-run the best LR from a fresher *pre-over-convergence* checkpoint before calling it "stuck" — otherwise "game stuck" and "this seed won't move" are indistinguishable. *(Run result weakened this branch: the head DOES move (down) → a signal problem, not a plasticity/seed problem.)*
- **Phase 1b — extend the winner (~2-3 days, ~$80-150).** Best LR, 15-20 iters, 400-1000 games/iter, same guards. North-star: positive H2H slope vs frozen iter1_ema crossing >55%. *(NOT run — no winner emerged.)*
- **Phase 2 — conditional anti-forgetting BUILD (~2-3 days eng), ONLY if Phase 1 degrades-and-reverts every iter** (forgetting confirmed as the blocker). Buffer-mixing (~30-50% engine/supervised positions into the FIFO buffer) and/or KL-to-anchor (penalize policy divergence from frozen iter1_ema; the search-consistency distillation scaffold is a starting point). Re-run Phase 1b at the LR that previously forgot. *(Demoted — it would preserve 0.737 but can't push it up; no signal to push.)*
- **Phase 3 — full scale / AZ-class (weeks, $$$).** Only on a proven 1b/2 slope; fold in [[e20]] throughput (R9: prove the lever first). *(Not reached.)*

**Decision logic:** gate-passing improvement → path found, scale. Degrade-and-revert every iter → forgetting is the blocker → Phase 2. Flat (AND fresher seed also flat) → genuinely stuck → bank iter1_ema+E8.

**The one honest YINSH-specific factor (cuts both ways):** 22% draws + value AUC ceiling ~0.74 off a *large engine corpus* = low value-signal density per game. Argues FOR more games (less signal/game ⇒ need more), but may ALSO be a partial evaluability floor. Phase 1a is the cheap disambiguator.

**Reasons to not believe / watch:** the gate-confound reframe is a *hypothesis*, not a guarantee — high LR may degrade even with tight gating, in which case Phase 2 is mandatory. Over-converged seed hedged by the fresher arm. No promise it clears iter1_ema. **But "we proved the loop can't work" was false — we proved the cautious micro-version can't, a much weaker claim.** This is the lever the evidence points to and the one never pulled.

**Discipline:** H2H vs the FIXED champion iter1_ema (R1); one regime change at a time; hard go/no-go at each rung.

**Phase 1a artifacts (BUILT, ready to launch):** `configs/e24_phase1a_lr_{3e-5,1e-4,3e-4}.yaml` (champion recipe, one variable = `trainer.lr`), driver `scripts/e24_phase1a_sweep.sh`, runbook `docs/experiments/completed/e24_phase1a.md`. The value-head AUC reuses the existing `scripts/value_head_calibration.py` (no build); the engine-labeled held-out `.npz` corpus is generated by `scripts/gen_engine_labeled_corpus.py` (consistent, model-independent HeuristicAgent labels — one quick data-gen on the box). So Phase 1a is fully buildable now: gen corpus → run the sweep driver.

## Provenance & links
- Source snapshots: 2026-06-03 scope (forward write-up, lines 345–441); Phase 1a RAN 2026-06-04 (Done entry, lines 2572–2639); 2026-06-07 post-E24 lever board.
- Results file (numbers + lessons): [`e24_phase1a_results.md`](e24_phase1a_results.md); runbook [`e24_phase1a.md`](e24_phase1a.md).
- Related: [[e19]] / [[e22]] (the prior NOT_STRONGER swings whose read E24 re-confirms — value-TARGET is the bottleneck), [[e23]] (the DROPPED "E23" this was renumbered to avoid), [[e21]] (the value-target redirect), [[e26]] (the surviving target-improvement lever), [[e18]] (the cheap free win that remains).
- Cross-doc: memory `project_e18_e19.md`.
