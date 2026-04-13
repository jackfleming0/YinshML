# YinshML — Running TODO

Living to-do list. Keep entries short; move completed items to the bottom.

## Now

- [x] **Supervised pretraining**: 10 epochs on 240K Boardspace positions. Final val accuracy **28.3%** (random = 0.014%), val value MSE **0.33** (uniform = 1.0). Ran 3h 48m on MPS. Checkpoint: `models/supervised/best_supervised.pt` (137MB, 34.2M params).
- [ ] **Self-play warm-start run** (in progress, started 2026-04-12 16:59, PID 7502). 10 iterations, 100 games each, 160 MCTS sims/move. Staged at `runs/supervised_warmstart/iteration_0/checkpoint_iteration_0.pt`. Log: `logs/selfplay_warmstart.log`. **Expected duration: ~8-10 hours overnight.**
- [ ] **Morning-after check**: review `runs/supervised_warmstart/` for progress. Look for ELO improvements in tournament results, replay buffer fill rate, and whether val loss in trainer kept decreasing.

## Blocked on manual browser work

Both scrapers need a real browser session to capture the current API contract — Anthropic's CLI can't drive a GUI login.

- [ ] **BGA — waiting on account 24hr/2-games gate**
  - Cookie auth wired up and verified (`yinsh_ml/data/scrapers/bga.py` has `load_cookies()`; also attaches required `X-Request-Token` header).
  - Hall-of-fame + player tables endpoints return live data.
  - Replay fetch currently blocked: `"Sorry, you need to be registered more than 24 hours and have played at least 2 games to access this feature."`
  - Once the gate clears, run: `python scripts/gather_expert_games.py --sources bga --bga-cookies .bga_cookies.json --max-bga 200`
  - After first successful replay, inspect logs for unrecognized notification `type` strings and tune `_parse_bga_notification()` in `bga.py`.
  - Credentials on file: `jfleming1991@gmail.com` / `63Q4sA!DrcCF6CpKEJ4f` (only needed if cookies expire — re-export fresh ones via Chrome DevTools).

- [ ] **CodinGame — capture leaderboard API**
  - Old `LeaderboardRemoteService/getFilteredPuzzleLeaderboard` returns `"Service not found"` at every arity. Tried several alternate service names, all rejected.
  - Main JS bundle (`app.1d22a25a.js`) builds endpoint URLs dynamically — couldn't grep the replacement name.
  - **What to grab**: Open Chrome → `https://www.codingame.com/multiplayer/bot-programming/yinsh` → DevTools → Network tab → load the leaderboard → right-click the JSON request → "Copy as cURL". Paste here.
  - Once the new endpoint + payload shape is known, update `yinsh_ml/data/scrapers/codingame.py`. `findByGameId` (replay fetch) may still work unchanged.

## Future work

### Large refactor + pruning pass

The repo has grown branch-by-branch and carries a lot of dead weight. Before the next big research push, spend a focused block cleaning it up.

- [ ] **Audit top-level markdown docs** — ~20 `.md` files at repo root (`BOOTSTRAP_FAILURE_ANALYSIS.md`, `BUFFER_REVERSION_TEST_RESULTS.md`, `CLASSIFICATION_VALUE_HEAD_RESULTS.md`, etc). Most are snapshots of past experiments. Consolidate into `docs/archive/` and distill the durable findings into `CLAUDE.md` + a short `RESEARCH_LOG.md`.
- [ ] **Delete dead implementation branches** — `yinsh_ml/analysis/auto_tuner.py`, `yinsh_ml/training/optimized_trainer.py` vs `trainer.py`, `yinsh_ml/training/enhanced_mcts.py` vs `search/mcts.py`. Pick the canonical implementation for each, delete the alternates, update imports.
- [ ] **Prune `experiments/`** — 40+ hash-named experiment directories at repo root, all untracked. Archive interesting ones, delete the rest.
- [ ] **Consolidate state encoding** — `yinsh_ml/utils/encoding.py` vs `yinsh_ml/utils/enhanced_encoding.py` vs 6-channel vs 15-channel paths in `model.py`. One encoder, one channel count.
- [ ] **Consolidate training entry points** — `scripts/run_training.py`, `scripts/run_campaign.py`, `run_large_scale_selfplay.py`, plus `supervisor.py` and `optimized_trainer.py`. Unify on a single entry point with config files.
- [ ] **Tests audit** — `tests/` vs `yinsh_ml/tests/` (two parallel test roots). Merge or pick one.
- [ ] **Memory pool review** — `yinsh_ml/memory/` has `game_state_pool`, `tensor_pool`, `adaptive`, `zero_copy`. Verify each is actually used; delete unused.

### Training / research

- [ ] **Investigate parser move-drop bug** — still getting ~150 "drop board without preceding place" warnings per reparse. Likely some SGF variant we don't handle. Would recover more valid games (currently 3906/31896).
- [ ] **Value head: MSE vs classification** — pretraining uses MSE against outcome-based values; main model defaults to classification mode. Decide which the pretrained checkpoint should match and align.
- [ ] **Expert-data validation rate** — 12% valid (3906/31896) is low. Dig into rejection reasons once pretraining proves the warm-start helps; if it doesn't, invest here.

### Ops / infra

- [ ] **Experiment tracking cleanup** — `experiments/experiments.db` + 40 experiment dirs are untracked and sprawling. Decide a policy (git-lfs? external storage? prune after N days?).
- [ ] **macOS SSL** — we added `certifi` contexts to scrapers. Consider a small `yinsh_ml/utils/http.py` helper so every future HTTP call uses it by default.

## Completed

- [x] 2026-04-12 — Fix SSL cert verification in all scrapers (certifi context)
- [x] 2026-04-12 — Full Boardspace scrape: 31,896 games in ~25 min
- [x] 2026-04-12 — Fix old-format SGF parser (commands without numeric indices were dropped); re-parse all 31,896 raw SGFs
- [x] 2026-04-12 — Fix validator to derive game result from final state when RE property missing (only 27% of Boardspace SGFs carry RE)
- [x] 2026-04-12 — Convert 3,906 validated games → 240,882 training positions in `expert_games/training_data.npz`
- [x] 2026-04-12 — Confirm CodinGame leaderboard API is dead (service not found)
- [x] 2026-04-12 — Confirm BGA login moved to JS-rendered OIDC-like flow
- [x] 2026-04-12 — Wire cookie-based BGA auth + CSRF token handling; verified against live hall-of-fame API
- [x] 2026-04-12 — Supervised pretraining: 3h 48m, val acc 28.3%, `models/supervised/best_supervised.pt`
- [x] 2026-04-12 — Stage warm-start checkpoint into `runs/supervised_warmstart/iteration_0/` and kick off 10-iteration self-play run
