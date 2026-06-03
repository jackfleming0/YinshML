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

# re-fit on real games (cloud; needs pandas/pyarrow)
python scripts/experiments/fit_heuristic_weights.py \
    --games-dir large_scale_selfplay_data/parquet_data \
    --method logreg --scale 10 \
    --include-experimental \
    --out configs/heuristic_weights/refit_logreg.json
```

`--include-experimental` additionally *reports* fitted coefficients for the
experimental palette (`yinsh_ml/heuristics/experimental_features.py`). Those are
**not** written into the loadable JSON — the production evaluator only consumes
the 6 features in `WeightManager.VALID_FEATURES`. Wiring an experimental feature
into the eval (extend `extract_all_features` + `evaluator._feature_names` +
`WeightManager.VALID_FEATURES`) is a deliberate, separate experiment arm.

Weights are clamped to `[0, 50]` (WeightManager constraint); a fitted negative
coefficient clamps to 0, i.e. the feature is dropped.

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
