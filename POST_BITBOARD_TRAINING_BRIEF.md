# Post-bitboard training run — does the speedup translate to a stronger model?

> Self-contained brief for a future session. Paste this as the first
> message, or `cat` it. Don't load prior conversation context — this
> doc has everything.

## What just landed

The `bitboard-port` branch (now merged to main, or about to be) shipped
the C++ engine plus four Python-side optimizations:

| commit cluster | what | single-game wall (sim=400) |
|---|---|---:|
| pre-port baseline | Python engine | ~340s |
| bitboard port | C++ engine | 235s |
| + heuristic eval cache | A | 110s |
| + Move.__hash__ cache | A' | 95s |
| + vectorized MCTS UCB | B' | 85s |
| + Position lookup table | C-1 | **76s** |

Cumulative: **78% wall-clock reduction**, ~4.5× total speedup.
Steady-state engine ratio (paired stress, `cloud_smoke_*_stress.yaml`,
sim=400, iter-2): **1.86× py/cpp**. Heuristic eval cache hit rate
**95.8%** in MCTS hot path.

See `BITBOARD_FOLLOWUP_PLAN.md` for the full per-candidate measured
outcome and parity-test inventory.

## The question

We just bought back a lot of compute. **Does that translate to a
stronger model?** Three ways speed could become quality:

1. **Same recipe, less wall-clock.** Run the previous "winning" config
   and measure: same final ELO, much faster. (Validates that the
   optimizations didn't silently regress training dynamics.)
2. **More iterations in the same budget.** Run 2× more iterations than
   before; does ELO continue climbing or plateau?
3. **Higher sim count.** sim=400 was prohibitive pre-port; now it's
   tractable. Stronger search → higher-quality training data → better
   model? Or does heuristic-cached low-sim already extract most of the
   signal?

(1) is mandatory — it's the regression check on training quality.
(2) and (3) are the actual experiments. Pick one based on goal: (2)
tests whether the agent was iteration-bound; (3) tests whether it was
search-quality-bound.

## Comparison baseline

Most recent "real" training runs lived under `phase_d_*` configs:

- `configs/phase_d_warmstart_long.yaml` — 30 iter × 100 games × sim=96
  (warm-started; the most directly comparable "long run")
- `configs/phase_d_winner.yaml` — 50 iter × 50 games × sim=96 (the
  current winning recipe per its own header comment)

Their ELO trajectories should be in:
- `runs_from_cloud/` (where finished cloud runs were synced) — empty
  on the current branch; the actual data lives in S3 / wherever
  `SYNC_RUN_DEST` pointed
- `cloud_logs/` for raw stdout

**Before kicking off the new run:** find the final ELO from the most
recent successful phase_d run and write it down. That number is the
bar to beat or match-with-less-wall-clock.

## Recommended starting setup

### Start with experiment (1): the regression check

Take the same recipe that won last time and just toggle the C++ engine
on:

```yaml
# configs/post_bitboard_regression.yaml
# = configs/phase_d_warmstart_long.yaml verbatim, plus:
self_play:
  use_cpp_engine: true
```

**Run it.** Same iteration count, same sim count, same everything else.
Expected outcome:
- Wall-clock per iteration drops by 30-50% (sim=96 isn't the regime
  where the bitboard win was largest, but it's still meaningful)
- Final ELO matches prior run within the tournament's confidence interval
- Cache hit rate logged per-game shows ~95%+ (telemetry already wired
  in `yinsh_ml/training/self_play.py:1690`)

If ELO doesn't match: **stop.** Something regressed silently. Check
the parity test logs first, then bisect across the four candidate
commits (`f6d3617`, `cdc63c1`, `b80e28d`, `56f0955`).

### Then experiment (2) or (3) based on what (1) showed

If (1) hits the same ELO faster: spend the saved budget on either
- **(2) longer training**: 2× iterations at the same recipe. Does ELO
  keep climbing past the prior plateau?
- **(3) higher sim count**: bump self-play `num_simulations: 96 → 200`
  (or 400) at the same iteration count. Does the higher-quality
  training data accelerate ELO climb or improve final ceiling?

(3) is the more ambitious experiment but also more interpretable — the
failure mode "more iterations = same plateau" is already documented in
the codebase's own training history (look for ablations under
`configs/ablation_*`).

## What to monitor

The supervisor logs everything you need; the dashboard surfaces the
important bits:

```bash
python run_dashboard.py
```

Watch for:
1. **ELO trajectory.** Per-iteration `tournament_*` results in the run
   dir; should climb monotonically or close to it.
2. **Self-play wall-clock per iteration.** Should be lower than the
   baseline by the speedup ratio. Logged as "Generated and processed
   N games in T s" (supervisor.py:879).
3. **Heuristic cache hit rate.** Per-worker log line "Game N
   eval-cache: hit_rate=X% hits=Y misses=Z size=A/B" (self_play.py:1696).
   If this drops below ~90%, something's misconfigured.
4. **Value head loss.** Should converge; spikes mean training instability.
5. **Ring mobility / search quality stats** in the iteration summary.

## Common pitfalls

- **`use_cpp_engine: true` requires the .so to be built on the cloud
  box.** `pip install -e . && python setup.py build_ext --inplace`.
  Sanity: `pytest yinsh_ml/game_cpp/tests/ -q` should pass ~140.
- **Memory pools and CppGameState.** The C++ engine wrapper bypasses
  the GameStatePool for cloning (uses `_engine.State.clone` directly).
  If a worker logs "GameStatePool" warnings, that's expected and
  benign — see `yinsh_ml/game_cpp/game_state.py` docstring.
- **Multiprocess RNG seeding.** Workers spawn fresh — confirm the
  config doesn't pin a seed in a way that makes all workers play
  identical games (look for `seed:` in the config).
- **Heuristic cache and `enable_forced_sequence_detection`.** MCTS
  callers pass `False` (per `self_play.py:1232`). If you wire a
  HeuristicAgent path that defaults to `True`, the cache still works
  but per-call cost is higher — re-profile if perf looks off.
- **Move pickling.** `Move.__hash__` cache strips on pickle; if a
  custom code path bypasses `__getstate__` (e.g. raw dict copy across
  processes), hashes can desync. Stick to `pickle` / `copy.deepcopy`
  / standard multiprocessing.

## Out of scope for this run

- **Architecture changes.** Don't change the network, encoder
  channels, or move encoding scheme. We want a clean comparison vs the
  previous winning recipe.
- **More C++ optimizations.** `BITBOARD_FOLLOWUP_PLAN.md` parks
  Candidate C-2 (port move conversion to C++) and B'' (stats-as-arrays
  in Node) as not-worth-it. Only revisit if this training run shows
  recipe cost is still the bottleneck.
- **Heuristic weight retuning.** The 7-feature weights are already
  phase-tuned from the 100K-game analysis. Refit only if model play
  reveals systematic feature gaps.

## Kickoff sequence

```bash
# 1. Verify the .so is built and tests pass
cd /workspace/YinshML
git pull
pip install -e . && python setup.py build_ext --inplace
pytest yinsh_ml/game_cpp/tests/ -q  # expect ~140 passing

# 2. Sanity: paired stress confirms 1.86× still holds on this machine
python scripts/paired_stress_benchmark.py

# 3. Create your run config (copy + add use_cpp_engine: true)
cp configs/phase_d_warmstart_long.yaml configs/post_bitboard_regression.yaml
# edit the new file: under self_play, add `use_cpp_engine: true`

# 4. Kick off
python scripts/run_training.py --config configs/post_bitboard_regression.yaml \
    2>&1 | tee cloud_logs/post_bitboard_$(date -u +%Y%m%dT%H%M%S).log

# 5. Watch the dashboard in another terminal
python run_dashboard.py
```

Expected duration: phase_d_warmstart_long was 30 iter × 100 games × sim=96.
Pre-bitboard that took ~24h. Post-bitboard should be ~14-18h based on
the 1.86× ratio applied to a sim=96 mix (the ratio is smaller at low
sim because NN forward dominates).

## Stop conditions

- **Stop and report success** if (1) the regression check matches prior
  ELO within tournament CI AND completes in <60% of prior wall-clock.
- **Stop and investigate** if ELO trajectory diverges from prior runs
  by more than the tournament CI in either direction (positive or
  negative — both are interesting). Drop a profile, look at the loss
  curves, check whether a candidate's parity assumption broke under
  long-run dynamics.
- **Stop and report a finding** if (2) or (3) shows ELO continues to
  climb past the prior plateau. That's the headline result — the
  speedup compounded into model quality.

## Repo state when this brief starts

- Branch: `main` (after the bitboard-port merge), or `bitboard-port`
  if not yet merged.
- Build: `pip install -e . && python setup.py build_ext --inplace`.
- Sanity: `pytest yinsh_ml/heuristics/ yinsh_ml/tests/test_move_hash_cache.py
  yinsh_ml/tests/test_select_action_vector.py
  yinsh_ml/tests/test_cell_to_position_lookup.py -q` should report
  60+ passing on cloud (the cell_to_position one needs the .so).
- See `BITBOARD_FOLLOWUP_PLAN.md` for the full optimization history.
