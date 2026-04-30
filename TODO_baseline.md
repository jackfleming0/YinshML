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

### Tier 1-3 — Shipped (see `NEXT_UP.md` Completed section for details)

All Tier 1, 2, and 3 polish items landed between 2026-04-14 and 2026-04-16 in sequence: subtree reuse, epsilon_mix taper, EMA shadow, cosine + warmup LR, soft value targets, FPU in PUCT, mixed-precision autocast (PyTorch 2.7.1 upgrade), deterministic eval, Wilson-CI tournament logging, value-head calibration + entropy logging, move-encoder rework.

### Tier 3 — Open (deferred)

- [ ] **Opponent-pool expansion (tournament_sliding_window 3 → 5).** Part (a) of the original "Tournament CI / SE reporting + opponent pool expansion" item landed 2026-04-14 (per-matchup Wilson CI + SE now logged and stored under `pair_cis`; `arena.games_per_match: 50 → 100` for a real CI-width improvement). Part (b) — widening the rolling pool from 3 to 5 most-recent models — was deferred because it 3.3× arena cost (10 pairs vs 3). Re-open once the CI logs reveal whether matchups are diversity-bound.

### Tier 4 — Model capacity / encoding

- [ ] **Positional-threat heuristic feature.** `RESEARCH_LOG.md` calls this "the most obvious addition" — current 7-feature linear heuristic ceilings at 52% vs random. Add a "cells that complete or extend an opponent 4-marker run" differential in `yinsh_ml/heuristics/features.py`; learn a phase-specific weight via `run_complete_analysis.py`. Directly improves the hybrid-MCTS leaf evaluator.
- [ ] **History planes in state encoding.** Both 6-ch and 15-ch encoders (`yinsh_ml/utils/encoding.py`, `enhanced_encoding.py`) encode only the current board. Add last-N board planes (N=2-3) so the network can see ring trajectories and recent captures directly — and as a hedge against the policy head overfitting to repetition/shuffling patterns.
- [ ] **Gumbel AlphaZero sampling.** Sample-efficiency gains at small sim budgets. Explore only after the Tier-1 items land — trades against subtree-reuse priority.

### Tier 5 — Engine performance (added 2026-04-30 from alphazero-comparison study)

- [ ] **Bitboard port — replace Python game engine with C++ extension.** Source of design: temhelk/yinsh "yngine" (https://github.com/temhelk/yinsh). Use `__uint128_t` for white_rings/black_rings/white_markers/black_markers + precomputed `TABLE_RAYS[121][6]` rays per direction. Move generation becomes O(1) bit-scan vs current Python nested-loop iteration. Expected 10-100× speedup on `get_valid_moves`, `make_move`, `is_terminal`, which is the current self-play bottleneck. Implementation: pybind11 wrapper exposing `apply_move`/`generate_moves`/`is_terminal`; plug into `yinsh_ml/game/` as an alternative `GameState` that the existing tests can validate against the Python reference. Effort: 1-2 weeks. Payoff: every recipe experiment becomes 10× cheaper, which is the unblocking step for Phase E options ranking in `WARMSTART_PHASE_LOG.md` §9c. **Highest-EV engineering project as of 2026-04-30.**
- [ ] **Probabilistic fast/slow sim split.** Already implemented as opt-in flag in PR #9: `self_play.fast_simulations` + `self_play.fast_sim_prob`. Defaults off. RiverNewbury default is fast=20 sims with prob=0.75, slow=100 with prob=0.25 — halves self-play cost without obvious quality loss in their ablations. Tune once we have an active recipe to test against.
- [ ] **Root policy temperature.** Already implemented as opt-in flag in PR #9: `self_play.root_policy_temp`. Default 1.0 (no-op); RiverNewbury uses 1.1 (slight flattening of root prior, encourages exploration). Quick A/B once we have a recipe baseline.
- [ ] **Lock-free MCTS tree reuse with atomics.** yngine's design (`mcts.cpp:22-59, 127-152`) parallelizes MCTS via atomic CAS on child expansion. Current implementation batches 32 leaves serially. Defer until bitboards are in — gains scale with CPU cores, larger lift after the Python bottleneck is removed.
- [ ] **Growing replay window (Action B from alphazero-comparison study).** Replace fixed-position FIFO with iteration-based window (4-iter min, 20-iter max, grows over training). Mirrors RiverNewbury's `Coach.py:800-850`. Cleaner training-data dynamics; might reduce iter-to-iter overfitting that contributed to the Phase D plateau. Synergistic with bitboards (more games = larger window matters more).

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
