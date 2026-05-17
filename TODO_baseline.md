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

## Viz / audit tooling

Added 2026-05-17 with the live game viewer (`yinsh_ml/viz/`, `scripts/dashboard_games.py`, `scripts/generate_heuristic_games.py`). The viewer works end-to-end; these are the next iterations once an audit run produces real findings.

- [ ] **Drop coremltools/torch/seaborn from `HeuristicAgent`'s import chain.** A pure-search agent should not need `torch` (pulled via `search/__init__.py` → `mcts.py` → `utils/encoding.py`), `coremltools` (via `network/__init__.py`), or `seaborn` (via `utils/__init__.py` → `visualization.py`). Move the offending imports to lazy locals. Unblocks running HA on a minimal CPU box / in a notebook without 500MB of ML deps. The viewer dodges this by calling `extract_all_features` directly.
- [ ] **Fix `tracking/yinsh_visualizer.py` zig-zag offset.** Uses `y_offset = (col_idx % 2) * 0.5` which doesn't match YINSH's matching-sign-diagonal hex axis. Correct transform is in `yinsh_ml/viz/board_render.py::position_to_xy`; this module is older and feeds TensorBoard. Risk: someone is visually reading the buggy boards and inferring wrong adjacency relationships.
- [ ] **Incremental parquet writer.** `ParquetDataStorage` only flushes when `batch_size` is reached or `flush()` is called explicitly. Live mode uses `--batch-size 1` as a workaround (one parquet per game), which is fine for the audit workflow but produces many tiny files. A real incremental-append mode would let normal-sized batches still be visible mid-write.
- [ ] **Capture-aware game-end signal in the harness.** `generate_heuristic_games.py` currently caps games at `--max-moves`. When games consistently time out without a winner (depth 1-2 with random play), the value labels become noisy. Add a `--min-captures` filter that re-runs games that ended at 0-0 (signal that HA collapsed into shuffle mode).
- [x] **Basic missed-defensive-move detector.** ✅ Landed 2026-05-17 (`scripts/dashboard_games.py::_compute_trajectory`). Per-turn boolean: opponent had a 4-row at the start of player's turn AND it survived the player's move. Catches case (a) gradual-buildup defensive failures; blind to case (b) single-move path-flip captures (no visible 4-row state precedes them). Necessary-but-not-sufficient — see next item.
- [ ] **Tactical-quality defensive analysis.** The basic detector above is binary: did you respond to the 4-row at all? Real YINSH defense is multi-dimensional and "which flip" matters as much as "did I flip":
  - **Flip resilience.** When you flip an opponent's marker to break their 4-row, can they trivially flip it back next turn? Score each defensive option by whether the flipped marker is reachable by an opponent ring along a valid path. Prefer flips where the counter-flip would (a) require traversing the opponent's own markers (which would flip them too, costing material), or (b) sit on an axis the opponent's rings can't easily access.
  - **Counter-attack setup.** A defensive flip that also creates one of YOUR markers in a productive position is strictly better than a pure-defense flip. Score each option by `defense_value + α · offense_value` where offense_value comes from running the standard heuristic over the post-defense position.
  - **Multi-step tactical search.** The above two are 1-ply analyses. Real tactical quality requires search depth — "if I flip here, opponent flips there, can I still capture?" This is what minimax already does, but isolating *defensive quality* specifically (vs. raw eval) requires per-move attribution.
  - Engineering: needs `Board.legal_flips_of(position, by_player)` (small game-engine extension), a `defensive_options(state, threats)` enumerator, and a scoring composer. Probably 2-3 days of work for v1; opens the door to actually quantifying *how good* heuristic defense is, not just whether it happened. Discussion thread: 2026-05-17.
- [ ] **Fleet trajectory view.** Currently one game at a time. A "fleet" view (mean ± p25/p75 of each audit feature across N games) would surface corpus-level offense-only patterns much faster than reading individual games.

## Ops / infra

- [ ] **Experiment tracking cleanup** — experiment tracking DB + legacy dirs. Decide a policy (git-lfs? external storage? prune after N days?).
- [ ] **macOS SSL** — we added `certifi` contexts to scrapers. Consider a small `yinsh_ml/utils/http.py` helper so every future HTTP call uses it by default.
