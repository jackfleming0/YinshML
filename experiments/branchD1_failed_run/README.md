# Branch D.1 v1 — failed run forensics (2026-05-23)

This directory archives the post-mortem artifacts from Branch D.1 v1's
training run + SPRT screen. v1 used the minimal GAP value head
(`Conv → BN → ReLU → GAP → Linear(64, 7)`, ~17K params, direct projection).

## Result: NOT_STRONGER, decisive

SPRT (vs frozen `best_iter_4`, p1=0.60) concluded **NOT_STRONGER in 16
games** with candidate score **1-15-0** (6.25% WR), LLR -3.16 crossing
the lower boundary at -2.94. Color split 1/0 W/B — both colors lost,
not the deterministic-side artifact.

## What's in here

- `branchD1_iter4_vs_frozen.json` — full SPRT result JSON
- `manifest_final.json` — Branch D.1 v1 run manifest (config, hardware, timing)
- `feedback.md` — per-iter loss / suggestions emitted by the supervisor
- `tournament_history.json` — internal Elo tournament results across iters

## What the internal metrics looked like (all healthy)

- Loss monotonically decreased: policy 2.92→2.14 (−27%), value 1.96→1.89
- Anchor vs HA(d=1) stayed 95-100% all 4 evaluated iters (raw + mcts)
- 5/5 promotion gates passed
- 7.23h wall on RTX 5090 (vs 16h projection for 4090)

The internal metrics were uniformly green. The failure surfaced ONLY at
the SPRT level vs a strong opponent, confirming HA(d=1) is saturated.

## Diagnosis

The direct 64-dim → 7-class projection had no hidden layer to compose
pooled features. KataGo/Leela canonical small value heads always include
a hidden composition step (typical width: 80). v1 was likely producing
near-constant value estimates across positions, leaving MCTS unable to
discriminate branches → systematically losing to the spatial-head trunk
which has proper value estimation.

## Successor

Branch D.1 v2 (`configs/branchD1_v2_gap_mcts200.yaml`) adds the canonical
hidden layer: `... → GAP → Linear(64, 80) → ReLU → Linear(80, 7)`.
~22K params. Same recipe otherwise.
