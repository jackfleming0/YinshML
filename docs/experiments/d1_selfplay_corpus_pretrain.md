# D1 — Self-play data corpus for pretrain

**Status:** QUEUED
**Cost:** Game generation ~10-15h, ~$20; Corpus conversion ~1h, ~$1; Pretrain ~5h, ~$8; SPRT ~1-4h, ~$2-5. **Total: ~17-25h, ~$30**.
**Stack-rank:** Likely+ 3 / Unblocks 4 / Info-gain 4 / Cost 2 / Impl-risk 3 / Sum 16
**Dependencies / blocks:** Unblocks: a meaningful answer on theory III (data ceiling). If self-play-pretrained warm-start dramatically beats yngine-pretrained, the corpus was the limit.

## Description
**Goal:** generate ~100K self-play games using the best D.2 model (probably iter_0), capture MCTS visit distributions + rolled-out values as targets, build a 15-channel corpus from them, re-pretrain a new init from this corpus.

**Mechanism:** the yngine corpus uses raw outcomes (W/L/D) as value targets. Many positions have ambiguous outcomes — the result depends on subsequent play decisions. MCTS-rolled-out values use search depth to give a position-conditioned value estimate that's plausibly less noisy than the raw outcome. Better targets → higher achievable VAcc → stronger warm-start.

## Outcome
Pending — new corpus + checkpoint + SPRT verdict logged. Decisive read: if self-play-pretrained warm-start dramatically beats yngine-pretrained, the corpus (raw-outcome targets) was the data ceiling (theory III).

## Details

**Supporting evidence:**
- The D.2 value head plateaued at VAcc 0.636 — possibly the Bayes error on yngine raw outcomes.
- AlphaZero and successors use MCTS-rolled-out values as targets precisely because they encode search-informed value, not just raw rollout outcomes.
- The codebase already has self-play data collection infrastructure.

**Reasons to not believe:**
- **Self-play targets aren't free of bias.** They're biased toward whatever the *current* policy + value head predicts. A pretrain on self-play data may just reinforce existing model knowledge rather than add new signal — the "self-play collusion" failure mode the doc flagged elsewhere.
- **Generation cost is real.** ~100K games at MCTS-200 with batched eval = ~10-15h on 5090. Plus the pretrain on top.
- **The strength gap matters.** Our best self-play teacher (D.2 iter_0) is ~50 Elo below the previous champion `best_iter_4`. Using `best_iter_4` itself as the teacher might be better — but it's 6-ch and would produce a 6-ch corpus.

**Methodology:**
1. Pick teacher: `models/yngine_volume_15ch_pretrain/best_supervised.pt` (refrozen anchor, A1 verified STRONGER vs the prior Branch C champion — A1 result re-points D1 here away from `best_iter_4`). Genuinely 15-ch, no re-encoding problem.
2. **Generator script does NOT exist** — D1-sketch finding 2026-05-25. The closest is `scripts/run_selfplay_worker.py` which calls `SelfPlay.generate_games()` and prints the count but throws the results away. The data tuple from `generate_games` IS the right shape: per-game `(states, policies, values, history)` where `policies` are MCTS visit distributions and `values` are MCTS root values per position (Fix #1 in self_play.py:1647). Connector script needed (~2-3h coding):
   - Load teacher with auto-detect: `NetworkWrapper(model_path=...)`
   - Loop `SelfPlay.generate_games(batch)` in chunks of ~1000 games
   - **Scatter** per-position `policies` (over valid moves) into the full 7433-slot move-encoder space before saving — pretrain expects shape `(N, total_moves)` for soft targets, or `(N,)` argmax indices for hard targets. Soft preferred (more signal).
   - Save as npz with `states.npy`, `policies.npy` (soft, `(N, 7433)`) or `policy_indices.npy` (argmax, `(N,)`), and `values.npy` (`(N,)`). Schema matches `run_supervised_pretraining.py:97-114`.
   - Resume support — persist a "games-done" counter so crashes don't restart from zero (100K games is ~10-15h on 5090).
3. Convert npz → mmap shards (`scripts/convert_npz_to_mmap_shards.py`, already exists, schema-agnostic).
4. Pretrain on the mmap shards via existing `scripts/run_supervised_pretraining.py --data-dir <mmap_dir> --use-enhanced-encoding --value-head-type spatial --epochs 6`.
5. SPRT the resulting init vs `best_supervised.pt` (new anchor).

**Partial corpus already available:** the B1+B2+B3 run saved a replay buffer (12 MB compressed, ~2.2 GB uncompressed, ~50K MCTS-target samples across 5 self-play iters with the warm-start teacher) at `experiments/branchB1B2B3_run_2026-05-26/full_run_dir/20260525_233508/replay_buffer.pkl.gz`. Generated via the supervisor's proper batched MCTS path with subtree reuse. Lets D1 test "does pretrain on MCTS targets beat pretrain on yngine outcomes?" without the 10-15h generation cost up front.

**Open questions:**
- Use soft policy targets (full visit distributions) or argmax? The policy loss code already branches on shape. Soft targets carry more info.
- Use rolled-out root values or final-game outcomes as value targets? Doc notes the self-play trainer uses both (`trainer.value_loss_weights: [0.5, 0.5]`). Worth replicating that mix.

## Provenance & links
- Related: [[a4]] (regression value head — both target the value-head plateau / theory III; A4's "MCTS-rolled-out value targets aren't in the yngine corpus" is what D1 produces), [[a1]] (A1 result re-points D1's teacher).
- Source: `EXPERIMENT_BACKLOG.md` "Detailed write-ups" section. D1-sketch finding 2026-05-25; replay buffer noted in the B1+B2+B3 run log.
