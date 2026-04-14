# YinshML — NEXT_UP

Dynamic sequencer. What's active, what's queued, and the decision gates between them. Menus live in `TODO_baseline.md` (AZ polish pool) and `TODO_frontier.md` (research directions). Durable lessons live in `RESEARCH_LOG.md`.

Rule: this file is the only one that reorders. The menus are append-or-delete only.

---

## Active

- [ ] **BGA scrape/parse separation + cap-aware bulk crawl.** Design note written; full detail in git history.
  - Split `BGAScraper.scrape_game()` into `fetch_raw(tid)` (network) + `parse_raw(raw, tid)` (pure).
  - `scripts/bga_fetch.py` writes `expert_games/bga/raw/{tid}.json`, maintains `seen.json`, exits clean on cap-hit.
  - `scripts/bga_parse.py` reads `raw/*.json`, writes `parsed/{tid}.json`, always overwrites — safe to re-run after parser fixes.
  - Bump default `BGAScraper.delay` from 3s → 8s. Target ≤200 fetches/session.
  - Bonus: check in a couple of `raw/*.json` fixtures and unit-test `parse_raw`.

## Blocked on manual browser work

- [ ] **CodinGame — capture leaderboard API.** Old `LeaderboardRemoteService/getFilteredPuzzleLeaderboard` returns `"Service not found"` at every arity. Main JS bundle builds endpoint URLs dynamically; couldn't grep the replacement name.
  - **What to grab**: Chrome → `https://www.codingame.com/multiplayer/bot-programming/yinsh` → DevTools → Network tab → load the leaderboard → right-click the JSON request → "Copy as cURL". Paste here.
  - Once the new endpoint + payload shape is known, update `yinsh_ml/data/scrapers/codingame.py`. `findByGameId` (replay fetch) may still work unchanged.

---

## Next phase — Polish sprint + diagnostic probes (1-2 weeks)

Parallel track, not sequential. Goal by end of phase: an AZ baseline whose residual weakness is architectural, plus three probe results that say whether the expensive frontier items are worth committing GPU hours to later.

### Track A — polish sprint (from `TODO_baseline.md`)

Order updated 2026-04-14 after the 10-iter warm-start results came in. The iter 1-2 regression (-57, -24 ELO) directly evidences the curriculum item; the iter 6 → iter 9 fall-back (-64 ELO from peak) directly evidences EMA + tournament CI/SE. Cheap, evidence-backed items pulled forward; engineering-heavier items deferred until the cheap ones validate.

1. **Turn on D6 symmetry augmentation by default.** *Ungated 2026-04-14. Augmentation already on during the warm-start rerun (`augmentation.enabled: true` in `configs/training.yaml`) and ran cleanly across 10 iters at `num_workers=3`. Just verify it's wired in every other live training config — 5-minute audit. Free 12× data multiplier; nothing to slow-roll.*
2. **Heuristic-weight curriculum (1.0 → 0.3 → 0.0).** *Promoted from #4 — directly evidenced.* Iter 1-2 of the warm-start regress because the value head is readjusting from supervised distribution to heuristic-mixed self-play distribution. Curriculum annealing is exactly what dampens that handoff. Cheap trainer/supervisor change.
3. **EMA checkpoint for tournament eval.** *Was #2, kept high.* The yes-no-no-no-yes-no-no-no oscillation around iter 6 means latest-checkpoint-as-eval-target is too noisy. EMA smooths it. Independent of everything else; low risk.
4. **Tournament CI / SE reporting + opponent pool expansion.** *Promoted from Tier 3 menu — prereq for trusting any future A/B.* At 200 games per matchup, 49.5% has ~7% CI; the gate's calling rejections it can't statistically distinguish from coin flips. Log Wilson 95% CI per matchup, expand opponent pool to rolling top-K.
5. **Deterministic eval mode.** *Promoted from #11, paired with #4.* No fixed-seed path in `_play_match`. Together with #4, this is the trustworthy-comparisons foundation that everything downstream depends on.
6. **Subtree reuse across moves.** *Was #3, demoted.* Still the biggest single ELO lever in the menu, but it's the hardest engineering lift in Tier 1 — pull it after the cheap evidence-backed items land so the trustworthy-comparisons infrastructure is there to measure its impact.
7. **FPU in PUCT.** Tiny patch; near-zero cost.
8. **Soft value targets (Gaussian around class).** Small trainer change; hits the discrimination ceiling.
9. **Cosine + warmup LR schedule.**
10. **Promote `epsilon_mix` to config + taper.**
11. **Mixed-precision on MPS.** Less urgent now that the OOM is resolved; still a 20-30% wall-clock win.
12. **Value-head calibration + entropy logging.** Needed to *see* §5 probe results below.

Tier 4-5 items (threat feature, history planes, test hygiene) are held back until after the decision gate below — they're either GPU-dependent or low enough ROI to deprioritize.

### Track B — diagnostic probes (from `TODO_frontier.md`, Mac-Mini-feasible)

Fire in parallel with Track A. Each is 1-5 days on MPS. Order updated 2026-04-14 — §4 demoted because the warm-start question is mostly resolved.

1. **§5 Search-consistency regularization.** *Now the highest-information probe on the entire menu.* The warm-start regression question got answered by the 10-iter run — the open question is whether the discrimination plateau (0.104 ceiling per `RESEARCH_LOG.md`) is architectural or training-signal. §5 is the test. If it breaks the plateau, AZ recipe is sufficient and most of `TODO_frontier.md` is moot. 1-2 day trainer patch.
2. **§8 Value-head interpretability / SAE probe.** *Cheapest.* 1-2 days on the iter-6 promoted checkpoint (peak ELO). What concepts has the value head learned vs. failed to learn? Independent of §5; complementary diagnostic.

**Demoted to "later, only if needed":**
- **§4 Offline RL (IQL) pretraining.** Was sized as the answer to the warm-start regression. That regression mostly resolved with the simpler CE-form fix — iter 1-2 still dips but the system recovers and peaks at +61 ELO. IQL might still help eliminate the iter 1-2 dip but Track A #2 (heuristic-weight curriculum) is a much cheaper test of the same hypothesis. Run IQL only if curriculum *doesn't* fix iter 1-2.

---

## Decision gate #1 — after polish sprint + probes

Outcome determines the next phase. Three branches:

- **Branch A: §5 breaks the plateau (discrimination 0.104 → >0.13).** The plateau was a training-signal problem. AZ recipe is correct. Push `TODO_baseline.md` Tier 3-4 items and frontier items are mostly deferred. When GPU lands, scale up the polished AZ rather than pivoting.

- **Branch B: §5 fails to move it; §8 shows missing concepts.** Plateau is representational, not architectural. Add targeted auxiliary heads / features informed by SAE findings (positional threat, history planes, phase-conditioned heads). Still within AZ recipe but data-augmented. GPU planning is about capacity/depth, not architecture replacement.

- **Branch C: §5 fails; §8 shows rich internal representations.** Plateau is architectural. Commit to MuZero (`TODO_frontier.md` §1) or Transformer backbone (§2) when GPU lands. *Don't start either on Mac Mini* — wait for hardware.

§4's result is informative in every branch: if IQL warm-start survives iter_1 where BC fails, the supervised-vs-self-play training-regime gap is the real story. That conclusion holds regardless of §5 / §8.

---

## Later — GPU-dependent

Wait for GPU before starting any of these. See `TODO_frontier.md` hardware table for full feasibility breakdown.

- **§2 Transformer backbone.** First thing to try if decision gate lands on Branch B or C.
- **§1 MuZero / EfficientZero.** Only if Branch C is clear-cut.
- **§3 League training.** Requires fleet, not just one GPU. Later still.
- **§9 Quantization-aware training.** Unlocks 4-8× sims-per-second when paired with CUDA.
- **§10 Scaling laws study.** Needs enough runs to fit the curve.
- **§6 Diffusion policy head, §7 multi-game co-training.** Speculative; revisit if earlier items settle.

---

## Completed

### 2026-04-14

- [x] **10-iter warm-start rerun completed (6.44h, no OOM).** From CE-aligned iter-0 checkpoint at `runs/supervised_warmstart_v2/iteration_0/checkpoint_iteration_0.pt`. Tournament: iter 1-2 regress (1443, 1476 — rejected by 55% gate), iter 3 promotes (1534, 57.5%, +34 ELO over seed), iter 6 promotes (1561, 63.5%, +61 ELO peak). Memory stable across all 10 iters: peak MPS driver 5.10GB, RSS no upward trend, no cross-iter leak. **The CE-aligned warm-start does survive self-play.** The per-game TensorPool refactor at `self_play.py:1165-1173` is no longer triggered (no growth observed) — deferred. Findings logged in `RESEARCH_LOG.md` (multi-iteration + memory + tournament sections).

### 2026-04-13

- [x] Root-cause + fix iter-3 OOM from the previous warm-start. Cause: `num_workers=4` MPS tensor-pool copies + ~3GB MPS driver-cache inflation during backprop; not the buffer, not augmentation. Fix: `num_workers: 4 → 3` in `configs/training.yaml`, unified `[Memory]` logging at every transition in `supervisor.py::_log_mps_memory` incl. pre-training-epoch, `torch.mps.synchronize()` before `empty_cache()` confirmed in place. Verified via 1-iter smoke + 3-iter dry run (MPS driver stable at 3-4GB across iters). Full 10-iter warm-start rerun launched.
- [x] Fix supervised value-loss form mismatch — `scripts/run_supervised_pretraining.py` now uses CE on `_value_logits` against 7-class discretized targets (matches `trainer.py:624-636`); adds val value-class accuracy metric.
- [x] Retrain supervised checkpoint with CE loss — 10 epochs, 3h 46m on MPS, val PAcc=29.8% / VAcc=91.9%. `models/supervised/best_supervised.pt`.
- [x] Audit + fix two `NetworkWrapper` value-head bugs: (a) `predict_batch` double-tanh (`wrapper.py:439`), (b) `load_model` silently accepted `value_mode` / `num_value_classes` mismatches — now hard-fails like the channel-mismatch check.
- [x] Rewrite BGA parser from real notification schema — correct nesting (`data.logs[i].data`), single `type='move'` with log-template dispatch, `restartUndo` pop-by-player_id, `REMOVE_RING` uses `locationFrom`, regex word boundaries, inverted color map fixed. Probe game validates cleanly (85 moves). Scripts: `scripts/probe_bga_replay.py`, `scripts/probe_bga_no_warmup.py`.
- [x] BGA cookie set expanded — all 7 cookies required (added persistent `TournoiEnLigneid` + `TournoiEnLignetk` pair); session promotion now completes, `/archive` accepts.
- [x] Stage CE-trained checkpoint into `runs/supervised_warmstart_v2/iteration_0/checkpoint_iteration_0.pt` for the warm-start rerun.
- [x] Split TODO into `TODO_baseline.md` (AZ polish menu), `TODO_frontier.md` (frontier research), `NEXT_UP.md` (dynamic sequencer); add hardware-feasibility table to frontier file.

### 2026-04-12

- [x] Fix SSL cert verification in all scrapers (certifi context)
- [x] Full Boardspace scrape: 31,896 games in ~25 min
- [x] Fix old-format SGF parser (commands without numeric indices were dropped); re-parse all 31,896 raw SGFs
- [x] Fix validator to derive game result from final state when RE property missing (only 27% of Boardspace SGFs carry RE)
- [x] Convert 3,906 validated games → 240,882 training positions in `expert_games/training_data.npz`
- [x] Confirm CodinGame leaderboard API is dead (service not found)
- [x] Confirm BGA login moved to JS-rendered OIDC-like flow
- [x] Wire cookie-based BGA auth + CSRF token handling; verified against live hall-of-fame API
- [x] Initial supervised pretraining (old MSE loss): 3h 48m, val acc 28.3%
- [x] First warm-start run — died mid-iter 3, regression surfaced via tournament (root-caused 2026-04-13 to value-loss mismatch)
- [x] Large-scale pruning + consolidation — 6 commits on branch `clean-slate`, PR #8. Deleted dead implementation branches (`optimized_trainer`, `enhanced_mcts`, `enhanced_self_play`), 17 orphan tests, 70 stale `.md` files, `experiments/` (1.1 GB), old `runs/` subdirs (3.5→0.5 GB), 45 root-level ad-hoc scripts. Unified `use_enhanced_encoding` flag end-to-end. 716 tests collect clean. Findings distilled into `RESEARCH_LOG.md`.
