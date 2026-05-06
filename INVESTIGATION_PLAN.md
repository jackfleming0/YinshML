# White-wins-100% investigation: contained protocol

## Why this exists

Across two consecutive 20-iter training runs (B1 cosine LR, B2 step LR + 0.5 heuristic floor), we observed:

1. **iter-3 in B1 beat iter-19 100-0** in head-to-head (raw policy, temperature=0). 100% confidence interval.
2. **iter-9 in B2 vs iter-19 in B2** showed 50/50 aggregate but per-color split was iter_9 winning **50/0 as white, 0/50 as black** — a textbook "whoever is white wins" degeneracy under deterministic argmax.

The cloud instance has been wiped, so we have no surviving checkpoints. We need a self-contained protocol that, on a fresh cloud box, reproduces the symptom (or fails to — useful too) and returns enough data to diagnose it without a second iteration cycle.

## Hypotheses being tested

| ID | Hypothesis | Refuted if... | Implication |
|---|---|---|---|
| H1 | Heuristic itself is offense-only and biases self-play | Heuristic-vs-heuristic plays ≥30% black wins (we already saw this locally — confirming) | Heuristic is OK, look at network |
| H2 | White-wins is a deterministic-argmax brittleness; tiny logit asymmetry compounds at temperature=0 | H2H at temperature=0.5 shows balanced per-color split | Cosmetic issue; production MCTS play unaffected |
| H3 | The bias persists with stochastic policy (real policy collapse) | H2H at temp=0.5 still shows >70% white-when-current-net-is-white | Deep issue: value head shortcut or training pipeline bias |
| H4 | The bias persists even with MCTS (production-realistic eval) | MCTS-based H2H shows white-wins-anyway pattern | Production play is also affected; high-priority fix needed |
| H5 | Network regresses against earlier iters because of distillation, not training dynamics | Mid-iter checkpoint (iter-3 or iter-5) does NOT crush final iter under MCTS | Confirms B1 finding wasn't a one-off |

The protocol below tests all of these in one shot.

## Phases

| Phase | What | Wall-clock | Outputs |
|---|---|---|---|
| 0. Setup | Clone, deps, .so, sanity | ~5 min | Confirmed-good environment |
| 1. Training | 10-iter short run (B2 recipe, sim=100) | ~90 min | 10 checkpoints + training log |
| 2. Eval battery | H2H tests at temp=0, temp=0.5, MCTS | ~30 min | 3 JSON results + per-color breakdowns |
| 3. Heuristic sanity | Heuristic-vs-heuristic 10 games | ~5 min | Confirms H1 refutation reproduces |
| 4. Bundle | Tar everything | ~1 min | One `.tgz` to scp down |

Total cloud time: **~2.5 hours**, fully autonomous.

## How to run it

On a fresh cloud box with the YinshML repo accessible:

```bash
git clone https://github.com/jackfleming0/YinshML.git
cd YinshML
git checkout main

# One command runs everything end-to-end. Check the script before running
# if you want to override defaults (run dir name, iteration count, etc.).
bash scripts/run_diagnostic_protocol.sh
```

The script:
- Builds the .so and runs sanity tests; aborts if anything fails
- Kicks off the diagnostic training run
- After training completes, runs all eval batteries against the produced checkpoints
- Bundles every artifact into `diagnostic_bundle_<timestamp>.tgz` in the repo root
- Prints the scp command you'll need to download the bundle

Each phase logs to its own file under `diagnostic_output/<timestamp>/` so a partial run is recoverable. If the cloud box dies mid-run, restart and the script picks up from the last completed phase.

## Pulling the bundle

When the script finishes, it prints something like:

```
=== DIAGNOSTIC COMPLETE ===
Bundle: /workspace/YinshML/diagnostic_bundle_20260505T120000.tgz (~250 MB)
To pull locally:
    scp HOST:/workspace/YinshML/diagnostic_bundle_20260505T120000.tgz .
```

Run that scp on your Mac. The bundle contains:

- `training.log` — full supervisor output
- `manifest_final.json` — training run metadata
- `iteration_*/checkpoint_iteration_*.pt` — all 10 checkpoints (~1.3 GB raw, smaller compressed; can drop with `--no-checkpoints` flag)
- `iteration_*/metrics.json` — per-iter loss/cache/timing
- `eval_h2h_temp0.json` — raw-policy H2H @ temp=0
- `eval_h2h_temp0.5.json` — raw-policy H2H @ temp=0.5 (the H2 test)
- `eval_h2h_mcts.json` — MCTS-based H2H (the H4 test)
- `heuristic_sanity.log` — heuristic-vs-heuristic confirmation
- `summary.md` — auto-generated readout that walks the hypothesis table above with the actual numbers from this run

You can then terminate the cloud box.

## Reading the summary

`summary.md` is structured to either let you say "we're done, problem is X" or "we need a deeper dive into Y." Specifically it:

1. Reports per-color split at each temperature/MCTS configuration
2. Walks H1-H5 with the run's actual data and emits a verdict per hypothesis
3. If H4 is supported (white-wins under MCTS), recommends a targeted follow-up rather than continuing tuning

If the white-wins pattern doesn't reproduce on the fresh run at all, that's also a clear signal — it tells us B1/B2 had something specific to those runs (older code, transient), and we go straight to longer training instead of debugging a phantom.

## What this protocol does NOT do

- Does not test heuristic `score_cap` saturation. That's a known issue but secondary; investigate after this protocol resolves the primary question.
- Does not test value-head architecture changes. We're characterizing the bug, not yet fixing it.
- Does not run the full 20-iter B2 recipe — that's overkill for diagnosis. If H5 is supported and we want to confirm the iter-3-crushes-iter-N pattern at full scale, run the full `post_bitboard_tuning_b2.yaml` afterward.

## Cost note

- ~2.5h on a cloud GPU box at typical rates is single-digit dollars
- The bundle is ~250 MB compressed (1.3 GB checkpoints + small JSONs/logs)
- If you don't need the checkpoints (just the eval results), pass `--no-checkpoints` to the orchestrator and the bundle drops to ~10 MB
