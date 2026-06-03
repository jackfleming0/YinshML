# Heuristic-weight experiment harness

End-to-end workflow for re-fitting the production heuristic's feature weights,
validating them offline, training candidate models in parallel, and pitting
them against your champion(s). Motivated by the review in
`docs/game_reviews/bga_862307561_review.md` (which found `potential_runs_count`
was a dead feature whose weight was fit against a constant).

The pieces split by where they run:

| Step | Script | Runs |
|---|---|---|
| 1. Re-fit weights | `fit_heuristic_weights.py` | cloud (parquet) / anywhere (`--demo`, `--dump-baseline`) |
| 2. Offline A/B gate | `validate_weights.py` | anywhere (pure heuristic, no torch) |
| 3. Generate arm configs | `gen_weight_experiment.py` | anywhere |
| 4. Train arms in parallel | `run_experiment.py` | cloud (calls `run_training.py`) |
| 5. Tournament vs champion | `tournament_vs_champion.py` | cloud (torch / NetworkWrapper) |

Plumbing: training reads `self_play.heuristic_weight_config_file` from the
config (null ⇒ hardcoded default weights). That key threads
`run_training.py → supervisor → SelfPlay → YinshHeuristics(weight_config_file=…)`,
and the evaluator object (with its weights) is pickled to the self-play workers.

## 1. Re-fit weights

```bash
# baseline arm = current default weights, dumped to JSON
python scripts/experiments/fit_heuristic_weights.py --dump-baseline \
    --out configs/heuristic_weights/baseline.json

# re-fit the 6 production features on real games (cloud; needs pandas/pyarrow)
python scripts/experiments/fit_heuristic_weights.py \
    --games-dir large_scale_selfplay_data/parquet_data \
    --method logreg --scale 10 \
    --out configs/heuristic_weights/refit_logreg.json

# re-fit production + the experimental palette into a LOADABLE extended JSON
python scripts/experiments/fit_heuristic_weights.py \
    --games-dir large_scale_selfplay_data/parquet_data \
    --method logreg --with-experimental \
    --out configs/heuristic_weights/refit_palette.json
```

### Testing the experimental palette features

The evaluator runs on a **configurable feature set**. A weights JSON that
includes a palette feature (e.g. `defensive_disruption`) **self-activates** it:
the evaluator reads the active set from the weights' keys, so no code change and
no extra config are needed — the existing `heuristic_weight_config_file` plumbing
carries it into self-play. `--with-experimental` (or `--features a,b,c`) emits
such a JSON. Per-arm ablation is then just "which features does this arm's
weights JSON contain".

`WeightManager` requires the 6 production features in every file and accepts any
of the palette features (`OPTIONAL_FEATURES`); unknown keys are rejected.
Weights are clamped to `[0, 50]`; a fitted negative coefficient clamps to 0
(the feature is dropped).

> Runtime caveat: the palette extractors use the Python `Board` API
> (`valid_move_positions`, `board.pieces`). Verify a palette arm under your
> training engine setting — if you run `use_cpp_engine: true`, confirm the C++
> game-state exposes those methods to the heuristic, or run palette arms with
> `use_cpp_engine: false`. The palette is also heavier than the 6, so it adds
> self-play leaf-eval cost.

## 2. Offline A/B gate (do this before spending GPU)

```bash
python scripts/experiments/validate_weights.py \
    --weights-a configs/heuristic_weights/baseline.json \
    --weights-b configs/heuristic_weights/refit_logreg.json \
    --games 60 --depth 2
```

If the re-fit weights can't beat the baseline agent-vs-agent, don't promote them
to training.

## 3–4. Generate + launch the arms

```bash
python scripts/experiments/gen_weight_experiment.py \
    --base-config configs/ablation_baseline.yaml --exp-name refit_v1 \
    --arm baseline=default \
    --arm refit_logreg=configs/heuristic_weights/refit_logreg.json \
    --arm refit_corr=configs/heuristic_weights/refit_corr.json

python scripts/experiments/run_experiment.py \
    --manifest configs/experiments/refit_v1/manifest.json --max-parallel 2
```

`--max-parallel` = GPUs on the node. Across nodes, shard the arms over multiple
invocations. Each arm logs to `<save_dir>/train.log`; `run_status.json` records
exit codes. `--dry-run` prints the commands without launching.

## 5. Tournament vs champion

```bash
python scripts/experiments/tournament_vs_champion.py \
    --champion runs/champion/best.pt \
    --from-manifest configs/experiments/refit_v1/manifest.json \
    --games 40 --out configs/experiments/refit_v1/tournament.json
# pass --use-enhanced-encoding / --value-head-type to match how arms were trained
```

## What's validated here vs. on the cloud

The torch-free cores — weight fitting (`yinsh_ml/heuristics/weight_fitting.py`),
the weights-JSON load contract, the A/B match runner, config generation, and the
launcher orchestration — are covered by tests in `yinsh_ml/tests/`
(`test_weight_fitting.py`, `test_weight_config_plumbing.py`,
`test_weight_experiment_gen.py`). The training run and the NetworkWrapper
tournament require torch/GPU and run on the cloud.
```
