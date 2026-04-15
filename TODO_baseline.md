# YinshML — Baseline TODO (AlphaZero polish menu)

Static menu of AZ-recipe improvements. This file is a **pool**, not a plan — pick from here when the dynamic sequencer (`NEXT_UP.md`) needs the next polish item. Active work, decision gates, and completion history live in `NEXT_UP.md`. Frontier research directions live in `TODO_frontier.md`. Durable lessons live in `RESEARCH_LOG.md`.

## Training / research

- [ ] **Remaining value-head audit items (classification-mode correctness).**
  - `YinshTrainer._monitor_value_head` (`trainer.py:855-906`) — assumes regression: pre-tanh saturation, MSE/MAE against scalar targets. Rewrite for classification (CE, class accuracy, per-class histogram from `_value_logits`) or short-circuit in classification mode.
  - `trainer.py:680` — class histogram in classification branch references `pred_class` but `value_logits` is fetched as `_value_logits`; reread before touching.
  - `model.py:159-168` `outcome_values` shape — already covered by load-time check, but consider validating in `__init__` too.
  - `mcts.py:255-259, 305-320` — hybrid-mode neural/heuristic value scale alignment.
  - `ValueHeadMetrics` (`yinsh_ml/utils/value_head_metrics.py`) — interpretation in classification mode.
  - Misleading "value is already tanh'd" comment at `wrapper.py:173` (behavior is correct, comment is stale).
- [ ] **Investigate parser move-drop bug** — still getting ~150 "drop board without preceding place" warnings per reparse. Likely some SGF variant we don't handle. Would recover more valid games (currently 3906/31896).
- [ ] **Expert-data validation rate** — 12% valid (3906/31896) is low. Dig into rejection reasons only if warm-start still underperforms after the CE fix is proven.
- [ ] **BGA game selection strategy** — when we have more than a few hundred games, filter by: both players above ELO threshold, |Δelo| < 100, min 30 moves. Don't optimize this before the warm-start is proven; volume isn't the bottleneck right now.

### Tier 1 — Quick wins (infra exists, just needs wiring)

- [ ] **Subtree reuse across moves.** MCTS root is discarded after every move (`yinsh_ml/training/self_play.py:468+`, `yinsh_ml/search/mcts.py`). Reseat the new root at `old_root.children[played_move]` and reuse visits — roughly doubles effective sims-per-move. Node-pool interaction needs care. Biggest single playing-strength lever still on the table.
- [ ] **Promote `epsilon_mix` to config + add a move-number taper.** `yinsh_ml/search/mcts.py:473` hardcodes `epsilon_mix = 0.25`. Move to `MCTSConfig`, default 0.25, and allow a per-move-number decay (higher early, zero after move ~20) so root exploration stops injecting noise into late-game tactical positions.

### Tier 2 — Medium effort, solid ELO

- [ ] **EMA (or SWA) checkpoint for tournament eval.** Training keeps latest weights; tournament uses the same latest weights, so single-iteration loss dips leak directly into gate decisions. Add a parallel EMA copy (decay ≈ 0.999) updated in `supervisor.py` after each train step, use it for tournament play and ELO while the live net keeps training. Should reduce the 55%-promotion-threshold false-positive rate as a side effect.
- [ ] **Cosine + warmup LR schedule.** `trainer.py:469-484` uses `StepLR(step_size=10, gamma=0.9)` for both heads (comment records a prior CyclicLR instability). `CosineAnnealingLR` with a short linear warmup (~5% of steps) is the middle ground — smoother decay, no 10×-drop cliffs. Keep the value head's 5× LR multiplier.
- [ ] **Soft value targets (Gaussian around class).** `trainer.py:624-636` discretizes the scalar target to a hard one-hot over 7 classes before CE. Hard targets on a discretized regression problem encourage overconfidence and waste neighboring-class signal. Replace with a Gaussian smoothed distribution (σ≈0.5 class-width) — same tensor shape, directly targets the discrimination ceiling the research log keeps hitting.
- [ ] **FPU (First-Play Urgency) in PUCT.** `yinsh_ml/search/mcts.py:105-110` scores unvisited children by prior only. Add standard KataGo-style `fpu = q_parent − fpu_reduction · sqrt(Σ visited_policy)` so early expansion prefers high-prior unseen children over low-visit explored ones. Near-zero wall-clock cost.
- [ ] **Mixed-precision training on MPS.** `trainer.py` has no autocast. Wrap forward passes in `torch.autocast(device_type='mps'|'cuda')` — expect 20-30% training wall-clock reduction and lower RSS (helps the buffer OOM question indirectly). Sanity-check value-head CE loss scaling first.

### Tier 3 — Diagnostics & evaluation reliability

- [ ] **Opponent-pool expansion (tournament_sliding_window 3 → 5).** Part (a) of the original "Tournament CI / SE reporting + opponent pool expansion" item landed 2026-04-14 (per-matchup Wilson CI + SE now logged and stored under `pair_cis`; `arena.games_per_match: 50 → 100` for a real CI-width improvement). Part (b) — widening the rolling pool from 3 to 5 most-recent models — was deferred because it 3.3× arena cost (10 pairs vs 3). Re-open once the CI logs reveal whether matchups are diversity-bound.
- [ ] **Value-head calibration + entropy logging.** `yinsh_ml/utils/value_head_metrics.py` is skeletal. Add per-move policy entropy (flag sudden collapses), 7-class value calibration curve + ECE, value-vs-outcome scatter by game phase. These distinguish "loss is dropping but discrimination is flat" (the historical failure mode) from real progress.
- [ ] **Deterministic eval mode.** `yinsh_ml/utils/tournament.py::_play_match` has no fixed-seed path. Add a `deterministic=True` flag that pins seeds and selects greedy (argmax) moves post-search, so A/B comparisons (e.g. "cosine vs StepLR") aren't dominated by tournament noise.

### Tier 4 — Model capacity / encoding

- [ ] **Positional-threat heuristic feature.** `RESEARCH_LOG.md` calls this "the most obvious addition" — current 7-feature linear heuristic ceilings at 52% vs random. Add a "cells that complete or extend an opponent 4-marker run" differential in `yinsh_ml/heuristics/features.py`; learn a phase-specific weight via `run_complete_analysis.py`. Directly improves the hybrid-MCTS leaf evaluator.
- [ ] **History planes in state encoding.** Both 6-ch and 15-ch encoders (`yinsh_ml/utils/encoding.py`, `enhanced_encoding.py`) encode only the current board. Add last-N board planes (N=2-3) so the network can see ring trajectories and recent captures directly — and as a hedge against the policy head overfitting to repetition/shuffling patterns.
- [ ] **Gumbel AlphaZero sampling.** Sample-efficiency gains at small sim budgets. Explore only after the Tier-1 items land — trades against subtree-reuse priority.

### Tier 5 — Test / correctness hygiene (quality-sensitive)

- [ ] **Encoder roundtrip + symmetry-transform tests.** `yinsh_ml/tests/test_encoding.py` covers move-index roundtrip but not state encode→decode→re-encode equivalence, nor "rotating the state + rotating the policy gives consistent network output." These are the silent bugs that let a 12× augmenter feed subtly wrong targets for a week.
- [ ] **State-pool mutation guards.** `yinsh_ml/memory/game_state_pool.py` is shared across concurrent self-play workers with no inter-worker mutation assertion. Add a debug-only hash-before / hash-after check on release to catch any caller that forgets to copy.

## Refactor / cleanup

- [ ] **Per-game TensorPool churn** — `yinsh_ml/self_play/self_play.py:1165-1173` allocates/releases pool tensors once per game, adding up over long runs. Flagged during the 2026-04-13 iter-3 OOM investigation as a real leak but not the proximate cause. **Deferred 2026-04-14**: the 10-iter warm-start showed no cross-iter memory growth (peak MPS 5.10GB stable, RSS no upward trend), so this churn is benign in practice. Revisit only if a longer (≥30 iter) run starts trending.
- [ ] **Consolidate state encoding** — `yinsh_ml/utils/encoding.py` vs `yinsh_ml/utils/enhanced_encoding.py` vs 6-channel vs 15-channel paths in `model.py`. One encoder, one channel count.
- [ ] **Consolidate training entry points** — `scripts/run_training.py`, `scripts/run_campaign.py`, plus `supervisor.py`. Unify on a single entry point with config files.
- [ ] **Tests audit** — `tests/` vs `yinsh_ml/tests/` (two parallel test roots). Merge or pick one.
- [ ] **Memory pool review** — `yinsh_ml/memory/` has `game_state_pool`, `tensor_pool`, `adaptive`, `zero_copy`. Verify each is actually used; delete unused.

## Ops / infra

- [ ] **Experiment tracking cleanup** — experiment tracking DB + legacy dirs. Decide a policy (git-lfs? external storage? prune after N days?).
- [ ] **macOS SSL** — we added `certifi` contexts to scrapers. Consider a small `yinsh_ml/utils/http.py` helper so every future HTTP call uses it by default.
