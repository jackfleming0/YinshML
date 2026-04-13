# YinshML Research Log

One-line durable lessons distilled from ~70 experiment-snapshot docs that were deleted in the 2026-04-12 cleanup. Full originals remain in git history on the `architectural-improvements` branch.

## Value head architecture & loss functions

- Classification value head (7-outcome cross-entropy) outperforms MSE regression by ~46% on discrimination (0.082 vs 0.056); MSE inherently minimizes variance regardless of target diversity.
- Discrimination ceiling under MSE is ~0.056–0.059; increasing variance-penalty weight from 0.5→1.5 makes discrimination worse (diminishing/negative returns).
- Auxiliary discrimination loss (CE − 0.5 × batch variance) prevents multi-iteration collapse; best single-iteration discrimination 0.104 vs 0.090 without it.
- Double-tanh was a real bug: once inside `YinshNetwork.value_head`, once inside `NetworkWrapper` — compressed predictions toward 0. Keep tanh in exactly one place.
- Training/inference value-representation mismatch (train MSE+BCE, play pure MSE) breaks the correlation between training loss and playing strength; align the forms.

## MCTS & bootstrap

- MCTS value targets raise discrimination ~37% over raw game outcomes (0.059 vs 0.043) but still plateau under MSE around 0.056–0.059 — target quality isn't the bottleneck, the value-head loss is.
- Bootstrap from 100% heuristic (initial iteration) reaches the same plateau as MCTS bootstrap; the virtuous cycle needs a curriculum that starts heuristic-heavy (heuristic_weight=1.0) and anneals down.
- Hybrid self-play (heuristic_weight=0.3) with pure-neural evaluation creates a train/eval distribution gap; either play with the same mix you train with, or schedule heuristic_weight → 0 before evaluation.
- Early-iteration self-play is data-poor: weak policy head (~8-9% accuracy) × low sim count = noisy targets. Don't evaluate before iteration 2-3.

## Multi-iteration training & data diversity

- 100 games/iteration enables multi-iteration improvement (iter 2: +1 ELO); 50 games causes consistent −72 to −94 ELO per iteration via mode collapse. Do not drop below ~100 games.
- Self-play collapses to ~50 unique trajectories/iter if you don't diversify (temperature schedule, openings); each iteration specializes further, losing generalization.
- Buffer size (10K vs 50K) does **not** fix diversity collapse — the issue is game-count diversity, not buffer capacity; 5× larger buffer shows identical degradation curves.
- Buffer reversion on model rejection prevents cross-iteration contamination but is not sufficient on its own; keep reversion AND raise games/iter.

## Memory & infrastructure

- `retain_graph=True` in backward pass accumulates ~17K tensors/iter (≈99% of historic training-phase leak). Remove it unless you genuinely need graph re-use.
- Per-sample evaluation loop in the trainer adds ~35K tensor refs/iter — bulk-evaluate and disable per-sample eval hooks (98% leak reduction, 17,830 → 414 tensors/iter).
- Worker count 7 in tournament caused OOM; cap at 3 saves ~4 GB and only costs ~15% iteration time.
- MPS leak on Apple Silicon: tournament paths allocated outside the tensor pool; `torch.mps.synchronize()` before `empty_cache()` avoids orphaned allocations.
- Tensor-pool-bypass in any tournament/eval hot path silently re-introduces leaks; always go through the pool or explicitly acquire/release.

## Tournament & evaluation

- A 35-44% training-loss improvement can still fail the tournament gate if the value-head form (classification vs regression) differs between training and play — the gate is an alignment test, not a loss test.
- Promotion threshold 55% win rate is load-bearing; plateau at ~48% is the signal that value discrimination is too weak for MCTS to exploit.
- "Training loss trending down + flat ELO" was a logging bug historically (loss reported as 0.0). Check the loss-tracking pipeline before blaming the model.

## Heuristic evaluation (7-feature set)

- Linear heuristic: ~52% vs random; same 7 features under Random Forest reach 55.1% — non-linear headroom is ~3%, not more.
- Feature weight order: runs >> centrality ≈ spread > potential_runs >> chains ≈ mobility > edge_proximity. Adjust phase-specific weights, don't add more features speculatively.
- No positional-threat feature yet; manually engineered run-threat detection would be the most obvious addition if pushing past 55% linear.

## Training schedule

- StepLR (policy 1e-3→3e-5 over 4 epochs, value 5× higher) is more stable than CyclicLR for short (4-epoch) iterations.
- Batch size 256 with 33-66 batches/epoch (4-17K samples) is the safe zone. 40 epochs on 4K samples caused mode collapse; 4 epochs is the upper end.
- Training loss ↓ does NOT imply tournament ↑ unless value discrimination also improves; accuracy-on-7-classes (99.5%) can mask near-zero discrimination.

## Known limitations & open questions

- Discrimination target 0.15+ has never been reached (best: 0.104 with discrimination loss — 69% of target).
- High-confidence value predictions (|v| > 0.7) are <1% of positions; network is calibratedly cautious but may be leaving MCTS guidance on the table.
- Pure-neural play on hybrid-trained models (30% heuristic during training, 0% at eval) has a suspected distribution gap; curriculum schedule (1.0 → 0.3 → 0.0) is hypothesized but untested.
- Supervised warm-start with 240K Boardspace positions achieved 28.3% val-acc (random = 0.014%) — strong prior, but still unproven whether it shortens the self-play trajectory to target ELO.
