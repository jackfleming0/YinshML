# YinshML Research Log

One-line durable lessons distilled from ~70 experiment-snapshot docs that were deleted in the 2026-04-12 cleanup. Full originals remain in git history on the `architectural-improvements` branch.

## Value head architecture & loss functions

- Classification value head (7-outcome cross-entropy) outperforms MSE regression by ~46% on discrimination (0.082 vs 0.056); MSE inherently minimizes variance regardless of target diversity.
- Discrimination ceiling under MSE is ~0.056–0.059; increasing variance-penalty weight from 0.5→1.5 makes discrimination worse (diminishing/negative returns).
- Auxiliary discrimination loss (CE − 0.5 × batch variance) prevents multi-iteration collapse; best single-iteration discrimination 0.104 vs 0.090 without it.
- Double-tanh was a real bug: once inside `YinshNetwork.value_head`, once inside `NetworkWrapper` — compressed predictions toward 0. Keep tanh in exactly one place. **Resurfaced 2026-04-13** in `NetworkWrapper.predict_batch` even after the same fix landed in `predict()` — audit every leaf-eval path when fixing one, not just the path you're looking at.
- Training/inference value-representation mismatch (train MSE+BCE, play pure MSE) breaks the correlation between training loss and playing strength; align the forms.
- Supervised-vs-self-play value-loss mismatch washes out the warm-start: if supervised trains the value head with MSE on the scalar expected-value output while self-play trains with CE on 7-class logits, the value head is effectively retrained from scratch on iter_1 — tournament-measurable as iter_0 beating every later iteration. Align supervised to use CE on the same discretized targets (`round((v+1)/2·(K-1))`) the trainer uses.
- `NetworkWrapper.load_model` should hard-fail on `value_mode` / `num_value_classes` mismatch (detectable via presence + shape of `outcome_values` in state_dict), mirroring the existing channel-mismatch check. Silent mode-mix loads corrupt inference without any surface symptom.

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
- CE-aligned warm-start from a supervised iter-0 regresses iter 1-2 (41-44% win rate vs iter 0) before recovering at iter 3 (57.5%, promoted, +34 ELO over seed) and peaking iter 6 (63.5%, +61 ELO). Iter 1-2 is value-head readjustment from the supervised target distribution to MCTS-rooted self-play targets; it self-corrects when the 55% promotion gate prevents the regressed model from poisoning the next iter's buffer. Warm-start *does* survive self-play once value-loss form is aligned (CE on `_value_logits`); the prior "iter_0 beat iter_1+2" pattern was the value-loss form mismatch, not warm-start fragility.

## Memory & infrastructure

- `retain_graph=True` in backward pass accumulates ~17K tensors/iter (≈99% of historic training-phase leak). Remove it unless you genuinely need graph re-use.
- Per-sample evaluation loop in the trainer adds ~35K tensor refs/iter — bulk-evaluate and disable per-sample eval hooks (98% leak reduction, 17,830 → 414 tensors/iter).
- Worker count 7 in tournament caused OOM; cap at 3 saves ~4 GB and only costs ~15% iteration time.
- MPS leak on Apple Silicon: tournament paths allocated outside the tensor pool; `torch.mps.synchronize()` before `empty_cache()` avoids orphaned allocations.
- Tensor-pool-bypass in any tournament/eval hot path silently re-introduces leaks; always go through the pool or explicitly acquire/release.
- Training (backprop) — not self-play — is the MPS allocator pressure source on Apple Silicon. `torch.mps.driver_allocated_memory()` inflates from ~1GB → 3-5GB during the training epoch loop and stays elevated through tournament; self-play stays at ~1GB. The proximate cause of the 2026-04-12 iter-3 OOM was the overlap of buffer-fill RSS (~1.6GB at 50K samples) + 4 concurrent `num_workers` MPS tensor-pool copies + the training-phase MPS inflation, all hitting at the self-play→training transition. Buffer cap, augmentation, and ring-buffer eviction were *not* the cause — the deque (`maxlen=N`) cannot overflow.
- Per-iteration-boundary `[Memory]` snapshots (RSS + MPS driver/current + buffer fill on one line at every transition) are the cheapest diagnostic and caught a kill-point gap that the existing per-phase `_log_mps_memory` missed. Specifically: log just *before* the training epoch loop, not just before/after self-play and after training — the OOM lands inside the gap between "after self-play" and "starting epoch 1/N."
- `num_workers=3` for self-play (already established for tournament) is sufficient to keep MPS+RSS in budget across 10 warm-start iterations on a Mac Mini; peak MPS driver 5.1GB, RSS no upward trend, no cross-iter leak.

## Tournament & evaluation

- A 35-44% training-loss improvement can still fail the tournament gate if the value-head form (classification vs regression) differs between training and play — the gate is an alignment test, not a loss test.
- Promotion threshold 55% win rate is load-bearing; plateau at ~48% is the signal that value discrimination is too weak for MCTS to exploit.
- "Training loss trending down + flat ELO" was a logging bug historically (loss reported as 0.0). Check the loss-tracking pipeline before blaming the model.
- The 55% promotion gate is doing real work, not just bookkeeping: in the CE-aligned warm-start, iter 1-2 candidates landed at 41-44% (well below threshold) so didn't promote — preventing buffer contamination that would otherwise compound the dip. Iter 3 (57.5%) and iter 6 (63.5%) both cleared cleanly. Without the gate, the regressed iter 1 weights would be the seed for iter 2, and recovery by iter 3 would not be observed.

## Heuristic evaluation (7-feature set)

- Linear heuristic: ~52% vs random; same 7 features under Random Forest reach 55.1% — non-linear headroom is ~3%, not more.
- Feature weight order: runs >> centrality ≈ spread > potential_runs >> chains ≈ mobility > edge_proximity. Adjust phase-specific weights, don't add more features speculatively.
- No positional-threat feature yet; manually engineered run-threat detection would be the most obvious addition if pushing past 55% linear.

## Training schedule

- StepLR (policy 1e-3→3e-5 over 4 epochs, value 5× higher) is more stable than CyclicLR for short (4-epoch) iterations.
- Batch size 256 with 33-66 batches/epoch (4-17K samples) is the safe zone. 40 epochs on 4K samples caused mode collapse; 4 epochs is the upper end.
- Training loss ↓ does NOT imply tournament ↑ unless value discrimination also improves; accuracy-on-7-classes (99.5%) can mask near-zero discrimination.

## Data acquisition & scraping

- BGA replay access requires **all 7** session cookies, not just `PHPSESSID` + the rotating `*idt`/`*tkt` pair. The persistent `TournoiEnLigneid` + `TournoiEnLignetk` pair (no trailing `t`) is required for session promotion; `/player` succeeds without them but `/archive` rejects with "not logged in". Export the full set from Chrome DevTools each session.
- BGA YINSH notifications: single `type='move'` for all game actions; dispatch via the `log` template string (`places a ring`, `places a marker`, `moves a ring from`, `removes a row of markers`, `removes a ring from`). `restartUndo` pops one prior `move` notification by the same `player_id`. `REMOVE_RING` holds the board square in `locationFrom` (not `locationTo`, which is a `@N` reserve sentinel). Color: `#ffffff`=white, `#000000`=black.
- Log-template regex dispatch needs word boundaries: `'removes a ring from'` contains the substring `'moves a ring from'` (within `re-MOVES`), so a bare substring-match routes ring-removals into the MOVE_RING branch.
- BGA enforces a per-day replay-view cap and scraping violates their T&Cs. Rate-limit aggressively (≥8s between replay fetches), separate fetch (network, quota-bound) from parse (local, free) on disk so parser iteration never burns quota, persist seen/failed tables for graceful resume after cap-hit.

## Symmetry augmentation — D6 label was wishful; board is D2

- Advertised as "D6 symmetry" with a 12× data multiplier. Audit 2026-04-14 found two compounding bugs: (a) policy transform was a no-op for ~99% of policy indices (ring movement / marker removal / ring removal all returned `old_idx` unchanged); (b) the board does not have D6 symmetry — only 180° rotation preserves all 85 cells. Column sizes [4,7,8,9,10,9,10,9,8,7,4] are 180°-symmetric but a 60° rotation would need to map a 4-cell column somewhere and no such image column exists.
- **Actual symmetry group is D2 (Klein 4-group, 4 elements).** Verified empirically: each of {identity, 180° rotation, diagonal swap `(r,c)→(c,r)`, anti-diagonal swap `(r,c)→(10-c, 10-r)`} covers all 85 cells. Klein closure: `180° = diag ∘ anti-diag`; every element is order 2. Effective data multiplier is **4×, not 12×** — but every sample is now geometrically faithful.
- Policy fix: build the old_idx → new_idx permutation by forward-encoding. Decode state → get valid moves → for each move compute old_idx, apply coord_map, re-encode to new_idx. Sidesteps the encoder's lossy move-hash entirely. Drop-and-renormalize semantics for mass at invalid-in-transformed-state indices. Applies correctly across all 4 D2 transforms; tests in `TestPolicyGeometricCorrectness` cover PLACE_RING / MOVE_RING / REMOVE_RING round-trips under every non-identity transform.
- State-channel fix: `encode_state` fills channel 5 (phase scalar) uniformly across all 121 grid cells including the 36 off-board ones. The previous `_transform_state` initialized the output to zeros before writing the 85 on-board mappings, leaving the 36 off-board cells at 0 — `decode_state` takes `np.mean` over all 121 and decoded the wrong phase after every rotation. Fix: initialize `transformed = state.copy()` so off-board cells inherit the source value. Same mechanism protects any future spatially-uniform channel (turn number, score differential in the enhanced encoder).
- Perf: per-sample augmentation cost dropped from 27 ms to 5 ms for the original 12-transform expansion, now ~1.5 ms for the 4-transform D2 expansion. Decode_state + valid_moves factored out of the per-transform loop. ~0.3% of iteration wall-clock.
- **Encoder move-hash problem is separate and bigger.** `((src*31 + dst) % 5848)` for 85-position src/dst yields only 2687 distinct values for 7140 legal (src,dst) pairs — 62% collision rate, 3161 slots structurally unreachable because `max(src*31 + dst) = 2688`. Multiple distinct ring moves share policy indices in MCTS training targets; the network learns coupled predictions for aliased moves. Predates augmentation; affects MCTS targeting directly. Tracked as Track A #13 in NEXT_UP.

## Known limitations & open questions

- Discrimination target 0.15+ has never been reached (best: 0.104 with discrimination loss — 69% of target).
- High-confidence value predictions (|v| > 0.7) are <1% of positions; network is calibratedly cautious but may be leaving MCTS guidance on the table.
- Pure-neural play on hybrid-trained models (30% heuristic during training, 0% at eval) has a suspected distribution gap; curriculum schedule (1.0 → 0.3 → 0.0) is hypothesized but untested.
- Supervised warm-start with 240K Boardspace positions achieved 28.3% val-acc with the **old MSE value loss** (random = 0.014%); retrained 2026-04-13 with aligned CE value loss reached 29.8% val policy top-1 + 91.9% val value-class accuracy. **Resolved 2026-04-14**: 10-iter warm-start from CE iter-0 promoted iter 3 (1534, +34 ELO) and iter 6 (1561, +61 ELO), no OOM, peak MPS 5.1GB stable. The aligned warm-start *does* survive self-play.
- Whether iter 6's +61 ELO peak from a single 10-iter run is reproducible — and whether it extends with longer training — is the next open question. Run took 6.4h on Mac Mini.
