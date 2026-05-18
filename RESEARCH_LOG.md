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

## Tournament gate statistics

- The 55% Wilson gate was already in place at `supervisor.py::_wilson_lower_bound` long before this audit, but the log line only showed the lower bound. With `games_per_match: 50`, a 27/50 (54%) rejection has CI [40.4%, 67.0%] — wholly straddling the 55% threshold. The gate was making promote/reject calls on signal statistically indistinguishable from the threshold and the log gave no indication. Bumped to `games_per_match: 100` (CI half-width ~14% → ~10%, doubles arena wall-clock); still straddling near threshold but visibly tighter. Going further (200 games/match) is a sharper-CI vs iteration-throughput tradeoff — re-evaluate after the new logs are in hand.
- Promotion log now shows wins/total + win_rate + SE + CI95=[lower, upper] + an explicit "[CI straddles threshold — rejection may be statistical noise]" flag when a rejection's CI overlaps the gate. Same data goes to `_log_metric_safe` as `wilson_upper_bound` and `wilson_standard_error` for the experiment tracker.
- Per-pair Wilson 95% CI now logged for every round-robin matchup (and stored in `tournament_history.json` as `pair_cis`). Draws are excluded from the proportion's denominator so the rate is well-defined; total games (including draws) is reported alongside. Surfaces opponent-pool diversity gaps: if every pair has a wide CI overlapping 50%, expanding the sliding window matters more than tightening individual matchups.
- Wilson math extracted to `yinsh_ml/utils/stats.py` (single source of truth for `wilson_bounds` + `standard_error`); supervisor + tournament both delegate. Avoids the silent-divergence trap if anyone tweaks one formula and not the other.

## Tournament & evaluation

- A 35-44% training-loss improvement can still fail the tournament gate if the value-head form (classification vs regression) differs between training and play — the gate is an alignment test, not a loss test.
- Promotion threshold 55% win rate is load-bearing; plateau at ~48% is the signal that value discrimination is too weak for MCTS to exploit.
- "Training loss trending down + flat ELO" was a logging bug historically (loss reported as 0.0). Check the loss-tracking pipeline before blaming the model.
- The 55% promotion gate is doing real work, not just bookkeeping: in the CE-aligned warm-start, iter 1-2 candidates landed at 41-44% (well below threshold) so didn't promote — preventing buffer contamination that would otherwise compound the dip. Iter 3 (57.5%) and iter 6 (63.5%) both cleared cleanly. Without the gate, the regressed iter 1 weights would be the seed for iter 2, and recovery by iter 3 would not be observed.

## Heuristic evaluation (7-feature set)

- Linear heuristic: ~52% vs random; same 7 features under Random Forest reach 55.1% — non-linear headroom is ~3%, not more.
- Feature weight order: runs >> centrality ≈ spread > potential_runs >> chains ≈ mobility > edge_proximity. Adjust phase-specific weights, don't add more features speculatively.
- No positional-threat feature yet; manually engineered run-threat detection would be the most obvious addition if pushing past 55% linear.
- All 7 features are computed as differentials (`my_value - opponent_value`) — defense is mathematically captured by the existing set as long as search depth is ≥3. Adding "defensive features" buys nothing for the static-eval ceiling; it only matters as a speed/depth trade if running at depth 1-2 for bulk data generation. See `yinsh_ml/heuristics/features.py` and the audit runbook in `yinsh_ml/viz/README.md`.
- **Two distinct ways to form a 5-row, and the audit metric only catches one.** YINSH ring moves do two things atomically: (1) leave a marker of your color at the source, (2) flip every marker along the straight-line path. So a 5-row can form either via (a) gradual marker buildup over multiple turns — the 4-row state IS visible between turns and defendable by flipping one of its markers (depth-1 negamax catches this); or (b) a single ring move whose path-flip converts a 3-row or scattered markers directly into a 5-row of your color — no visible 4-row state, no defender warning. Audit metric `count(rows length == 4)` catches case (a) misses but is blind to case (b). For unambiguous post-hoc capture detection use score deltas between consecutive states (captured markers clear during RING_REMOVAL before row-length matching can fire).

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

## Track A polish sprint bake-off (2026-04-15 → 2026-04-19)

Twelve Tier 1-3 polish items shipped in 4 days: deterministic eval, EMA shadow, heuristic-weight curriculum, Wilson-CI tournament logging, subtree reuse (+ pre-existing MCTS expansion bug fix), FPU in PUCT, soft value targets, cosine+warmup LR, epsilon_mix taper, mixed-precision autocast (PyTorch 2.7.1 upgrade), value-head calibration+entropy logging, collision-free move encoder (7395→8390 slots). Also shipped: MCTS child-node state-copy elimination (3.5× search speedup) and `is_valid_position` frozenset cache.

### MCTS expansion bug — the biggest find

The `action is None or ...: continue` guard in `search()` / `search_batch()` misfired on every first simulation (root unexpanded → while-loop doesn't run → action legitimately None → skip fires). Effect: MCTS has been **silently falling back to uniform-over-valid-moves** for the entire training history — "160 sims" was zero sims. Fixed as part of subtree reuse. Side effect: per-game wall-clock jumped from ~1s to ~15 min because MCTS now does real work. This bug retroactively explains why prior runs showed limited improvement with increasing sim budgets.

### Bake-off methodology

Ablation-based: baseline config disables 9 Track A training features via config flags (subtree reuse, FPU, epsilon taper, soft targets, EMA, cosine LR, autocast, heuristic curriculum, deterministic eval). Move encoder rework + MCTS expansion bug fix stay in both sides — architectural, not recipe.

Three rounds of bake-offs revealed scale-sensitivity confounds:
- **v1 (5 iters, 8 sims):** 40-game first pass showed +88 ELO (noise); 100-game follow-up: 51-49, inconclusive. Models too weak at this scale for Track A features to compound.
- **v2 (10 iters, 16 sims, all features ablated):** Baseline won 60-40 (ΔElo ≈ -70). Root cause: cosine LR schedule (designed for 200 epochs) decayed to ~2% of base by epoch 18 of a 20-epoch run — 33× lower LR than StepLR baseline. Heuristic curriculum (1.0→0.0 over 5 iters) also dropped guidance too early at this short horizon.
- **v3 (10 iters, 16 sims, scale-sensitive features matched):** After fixing LR schedule (both StepLR) and heuristic curriculum (both constant 0.3), isolating only scale-neutral features: 96-102-2, ΔElo ≈ -10.5, CI [0.42, 0.55]. **Statistical tie.**

### Indirect training metrics — clearer signal than head-to-head

Even when head-to-head play is inconclusive, training diagnostics show Track A's value:
- **ECE** (calibration): challenger 0.110 vs baseline 0.142 — **23% better** (soft value targets working as designed).
- **Value loss**: baseline 1.61 vs challenger 1.81 — challenger's is higher, but expected: soft-target label distributions have higher entropy than one-hot, raising the CE floor. The gap doesn't mean the challenger learned *less* — it learned a *softer* distribution.
- **Policy loss/entropy**: ~tied, confirming the effect is value-head-specific.

### Lessons

1. **Cosine LR and heuristic curriculum are scale-sensitive.** They need training horizons long enough for their decay curves to be well-past the "model is still learning fast" phase. At <50 epochs, StepLR is safer. At 200+ epochs, cosine should dominate.
2. **Track A's scale-neutral features (subtree reuse, FPU, soft targets, epsilon taper, EMA) improve calibration measurably but don't move head-to-head ELO at 10-iter toy scale.** Expected: these are rate-of-improvement features that compound over many iterations of the network strengthening.
3. **The MCTS expansion bug fix is by far the largest single contribution** — but it's baked into both bake-off sides (it's a correctness fix, not a recipe choice), so its impact doesn't show up in the A/B comparison. It will manifest as a step-change in absolute playing strength vs any historical checkpoint.
4. **The proper full-scale bake-off (50 iters, 64+ sims) remains the gold-standard test.** The infrastructure (`scripts/run_bakeoff.py`, ablation configs, Wilson CI reporting) is proven and ready. Compute budget is the constraint (~4-8 days on Mac Mini MPS).

## Track B probes (2026-04-19 → 2026-04-20)

Two diagnostic probes landed to answer whether the 0.104 discrimination plateau is training-signal or representational.

### §5 Search-consistency regularization — shipped

Distills long-search MCTS visit distributions + root values into the network as an auxiliary loss every K training batches. `YinshTrainer._search_consistency_step` samples replay-buffer positions, runs long-search MCTS as a teacher, computes `λ_π · CE(π_long, softmax(logits)) + λ_v · MSE(v_long, v_pred)`. Off by default (`trainer.search_consistency.enabled: false`); smoke flips it on so the code path is always exercised. 16 tests; smoke-validated through 8 iterations of debugging.

### §5 bugs & gotchas uncovered

- **Serial `MCTS.search()` didn't expose `last_root_value`.** Only `search_batch()` set it — a silent asymmetry. Fixed by mirroring the assignment at end of `search()`. Affects any future consumer of the root value from the serial path.
- **`BatchNorm1d` batch-size-1 train-mode trap.** The value head contains `nn.BatchNorm1d(512)` at `value_head[8]`. Calling `predict_from_state(state)` (batch_size=1) from MCTS raised `ValueError: Expected more than 1 value per channel when training` from iter 1 onwards in training runs — specifically after a `train_step` has run on the same network on MPS with bf16 autocast. Even explicit `eval()` calls on the module + individual BN layers didn't fix it (logs confirmed `.training=False` at the time of forward, yet BN still fired the check). Root cause unproven but the symptom is reproducible. **Fix**: route the probe's MCTS through `search_batch` (which evaluates leaves via `predict_batch(states)` with batch_size>1), bypassing the single-sample BN path entirely.
- **`_reset_network_objects` network-swap hazard.** The supervisor's every-3-iters cleanup swaps `trainer.network` for a freshly-loaded wrapper (`supervisor.py:1250-1252`). A cached `_sc_mcts.network` would hold a stale reference post-swap. `_search_consistency_step` now rebinds `mcts.network = self.network` on every call. Initially suspected to be the BN1d bug but wasn't — the swap only fires at `iteration_counter % 3 == 0` and smoke (2 iters) hit the bug without triggering a swap. Still a real latent bug; rebind stays.

### §8 SAE probe — shipped

Trains a 256→2048 sparse autoencoder on the value head's penultimate ReLU activations (model.py:153). Identifies what concepts the network has learned vs. missed. New package `yinsh_ml/interpretability/` (`activation_capture`, `sae`, `feature_analysis`, `position_generator`); end-to-end via `scripts/run_sae_probe.py`. 13 tests.

### §8 — the discrimination plateau, quantified

First run landed on `runs/bakeoff_challenger/20260419_031237/iteration_9/checkpoint_iteration_9_ema.pt` (5000 positions, L1=1e-5, 50 epochs): **value range `[-0.06, +0.10]` — max |v| < 0.10.** This is the discrimination plateau showing up as a raw distribution-of-outputs statistic, independent of the classification accuracy metric that produced the 0.104 number in Track A's training diagnostics. The network is operating in ~10% of the `[-1, +1]` value space. Practical implication: the default `find_confident_errors` threshold (0.7) returns zero matches because no prediction crosses it — added `--confidence-threshold` CLI knob; at 0.05 the probe surfaces 53 confident-error positions (1.1% of buffer) for the "missing concepts" analysis.

### §8 confident-error analysis (53 positions, threshold 0.05)

- **Asymmetric over-optimism**: 46/53 (87%) of confident errors are "pred > 0, actual < 0" — the network thinks the position is winning but the player loses. Only 7/53 are the reverse. The network has a positive-bias failure mode, not symmetric noise.
- **Phase distribution**: 49/53 (92%) in MAIN_GAME, 4/53 in RING_REMOVAL, 0 in RING_PLACEMENT. Placement errors don't happen because the value head defaults to ~0 there; errors appear as soon as markers start placing.
- **Ring-count signature**: 47/53 errors have 5 rings per side (no captures yet) — the network mispredicts position quality *before any captures*, not after. This rules out "can't count rings" as the bug and points to missing *threat detection* as the likely gap.
- **Marker diff has no signal**: range [-6, +6], mean -0.68. Being up/down on markers doesn't predict which positions get misjudged — another argument that it's not a simple feature the network is missing, but an emergent one (threat structure).
- **Move-number range [10, 86]**: errors span the whole main game. Not a beginning-only or endgame-only failure.

Synthesis: the 0.104 discrimination ceiling is not "network is appropriately uncertain" — it's **"network has learned a slightly-positive-biased value function that cannot distinguish winning from losing positions pre-capture"**. Matches the Track A finding that "completed_runs_differential" is the dominant heuristic feature (weight 0.239): the network relies on something that only exists *after* captures. §5 search-consistency is the right first probe (distillation should transfer MCTS's post-search value estimate into the pre-search network output). Branch B fallback if §5 fails: targeted threat-detection auxiliary head.

### §8 SAE hyperparameter notes

- `l1_coefficient=1e-3` is too aggressive for this network's low-dynamic-range activations: 1852/2048 features go dead, final sparsity 0.3% (too sparse to interpret).
- `l1_coefficient=1e-5` → 391 active features (256 dense firing on >50% of positions, 135 sparse-but-firing), final sparsity 12.3%, reconstruction MSE 1.9e-4. Default bumped implicitly via the hyperparameter writeup; configs to be tuned once findings land.
- Top features by "distinctiveness" all collapse on the same board archetype (5 rings each, move 10-16, MAIN_GAME, 0-2 markers) — the position-generator samples via network-policy rollouts without MCTS and almost every game passes through the early-main-game, so that regime dominates top-K activations. For genuine concept diversity in a follow-up run, the probe needs either (a) MCTS-driven generation matching the training distribution, or (b) sampling from a real training run's replay buffer.

### Incidental finding

- **Pre-move-encoder-rework checkpoints are unusable.** `runs/supervised_warmstart_v2/iteration_6/checkpoint_iteration_6.pt` (the +61 ELO peak) has a 7395-slot policy head; current code emits 8390 slots and `NetworkWrapper.load_model` hard-fails on the mismatch (by design, from the Track A move-encoder rework). The strongest available SAE/bake-off target on current code is the bake-off challenger's iter 9 EMA checkpoint — whose own bake-off result was a statistical tie with baseline at the toy scale.

## Known limitations & open questions

- Discrimination target 0.15+ has never been reached (best: 0.104 with discrimination loss — 69% of target). §5 probe is now in place to test whether long-search distillation breaks the plateau; pending full-scale run.
- High-confidence value predictions (|v| > 0.7) are <1% of positions; network is calibratedly cautious but may be leaving MCTS guidance on the table. **Quantified 2026-04-19**: bake-off challenger iter 9 EMA produces `|v| ≤ 0.10` across 5000 sampled positions — the "calibratedly cautious" framing was understating it by ~7×.
- Pure-neural play on hybrid-trained models (30% heuristic during training, 0% at eval) has a suspected distribution gap; curriculum schedule (1.0 → 0.3 → 0.0) is hypothesized but untested.
- Supervised warm-start with 240K Boardspace positions achieved 28.3% val-acc with the **old MSE value loss** (random = 0.014%); retrained 2026-04-13 with aligned CE value loss reached 29.8% val policy top-1 + 91.9% val value-class accuracy. **Resolved 2026-04-14**: 10-iter warm-start from CE iter-0 promoted iter 3 (1534, +34 ELO) and iter 6 (1561, +61 ELO), no OOM, peak MPS 5.1GB stable. The aligned warm-start *does* survive self-play.
- Whether iter 6's +61 ELO peak from a single 10-iter run is reproducible — and whether it extends with longer training — is the next open question. Run took 6.4h on Mac Mini.
