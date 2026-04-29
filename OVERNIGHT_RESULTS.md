# Overnight phase results — warm-start eval & path forward

**Date:** 2026-04-29
**Branch:** `alphazero-comparison` (companion to `clean-slate`)
**Spend so far:** ~$22 of $200 budget. **$178 remaining.**

---

## TL;DR

**Your stated goal — "play competitively with an intermediate player" — is already met by the supervised seed alone.** No self-play training needed for this.

```
iter_0_warmstart vs HeuristicAgent(depth=2) @ 100 MCTS sims:
   30/30 wins  (100%, CI95 [0.886, 1.000])
   avg game length 65.5 moves   (decisive, not stalling)
   Verdict: STRONG — clears 'intermediate player' bar
```

The 12-iter self-play run produced models *strictly weaker* than the supervised warm-start (head-to-head: iter_0 sweeps iter_3, iter_5, iter_9 all 40-0). Self-play training is moving you backward, not forward, because of an unfixed gating-loop bug in the supervisor.

---

## Three concrete numbers

1. **Supervised warm-start strength** (10 epochs, 240k expert positions, ~9 min on a 4090):
   - val PAcc=0.291 (top-1 expert-move match)
   - val VAcc=0.892 (value class accuracy)
   - vs HeuristicAgent(depth=2) @ 100 sims: **30/30** ✓
2. **Post-self-play strength** (12-iter de-risk run from warm-start):
   - iter_3 vs iter_0: **0/40** (completely dominated)
   - iter_5 vs iter_0: **0/40**
   - iter_9 vs iter_0: **0/40**
   - Internal trajectory IS improving (iter_9 sweeps iter_3 and iter_5 40-0), but the whole regime is below iter_0.
3. **Per-iter cost** (with anchor disabled, sliding window 2): flat ~21 min/iter on a 4090. So the recipe scales fine, it just optimizes in a strictly weaker basin.

---

## Why self-play degraded the model — the gating bug

In `yinsh_ml/training/supervisor.py`, the AlphaZero gating loop is incomplete:

| Canonical AZ step | YinshML behavior |
|---|---|
| 1. Train candidate. | ✅ |
| 2. Match candidate vs incumbent. | ✅ |
| 3. If candidate wins ≥55%: promote, candidate becomes data generator. | ✅ |
| **4. If candidate loses: discard candidate weights, keep using incumbent for self-play.** | ❌ **MISSING** |

Without step 4, the trained network is always used for self-play, even when it's strictly worse than its predecessor. After iter_1's catastrophic regression, every subsequent iter is generating data downstream of a damaged network. iter_0's strength can never re-enter the loop.

**Important context:** a friend reportedly worked overnight on this exact issue and may have landed an opt-in fix (`arena.revert_self_play_on_gate_failure: true`) on a different branch. If that's real and tested, it's the proper fix for warm-start training. **Have not yet seen the code; needs evaluation.**

---

## Paths forward (pick by goal)

### Path A — ship the intermediate-level model (TODAY)

**Cost: $0.** You already have it.
- The supervised seed `models/supervised_seed/best_supervised.pt` (10-epoch, on cloud) is the deliverable.
- Pull it locally:
  ```bash
  rsync -avz -e "ssh -p <PORT>" \
    root@<HOST>:/workspace/YinshML/models/supervised_seed/best_supervised.pt \
    models/supervised_seed/best_supervised_cloud.pt
  ```
- Use `scripts/play_vs_model_mcts.py` to play against it yourself for sanity-check.
- Destroy the cloud instance.
- **Done.**

### Path B — confirm at deployment-realistic settings (TODAY, ~5h, ~$2)

If you want a more rigorous "intermediate" claim:
```bash
python scripts/eval_vs_heuristic.py \
  --checkpoint models/supervised_seed/best_supervised.pt \
  --num-games 60 \
  --depth 3 \
  --mcts-simulations 200 \
  --device cuda
```
This is depth=3 (one ply harder than the win we already have) at 200 MCTS sims (deployment-realistic but tractable, unlike the 400-sim run that ran 9+ hours). At ~3 min/game on a 4090, this finishes in ~3h.

If verdict is STRONG/PROMISING here too: rock-solid claim of intermediate-level. If WEAK: iter_0 plateaus at depth=2, and beyond-intermediate needs Path C.

### Path C — beyond intermediate, requires the gating fix (TOMORROW+)

Toward expert-level. Two prerequisites:
1. **Evaluate the friend's rollback fix** (`overnight_findings.md` on their branch). Need to verify:
   - Does it actually preserve warm-start strength when promotion fails?
   - Does it allow gradual improvement above warm-start, or does it just freeze?
   - Any other knob changes (sliding window, gate threshold)?
2. **Then run a 30-iter long-run experiment** with the fix enabled, warm-starting from `best_supervised.pt`. ~10h, ~$4. If iter_15+ surpasses iter_0 in head-to-head, scale further.

If the rollback fix doesn't pan out: implement it ourselves in supervisor.py and test.

### Path D — speculative pinned-heuristic (cheap one-off)

Keep `heuristic_weight: 0.3` throughout (no decay). The heuristic acts as an external anchor that limits how far self-play data can drift from a sane policy distribution. ~$4, 10h. If it works, it's a band-aid for the missing rollback. **Skip if Path C is available** — Path C is the principled answer.

---

## Recommendation

1. **Today:** Path A (ship). You have the deliverable. Optional: Path B for the rigorous claim.
2. **Tomorrow:** Read `overnight_findings.md`, evaluate the rollback fix, decide Path C.

Don't burn budget on Path D unless the rollback fix is unavailable or unconvincing.

---

## Open items / TODOs

- [ ] Pull the 10-epoch supervised checkpoint from cloud to local.
- [ ] Play 5-10 games against iter_0 yourself via `scripts/play_vs_model_mcts.py` to validate "intermediate-level" qualitatively.
- [ ] Read `overnight_findings.md` from friend's branch.
- [ ] Decide: Path A only, Path A+B, or push to Path C.
- [ ] Destroy cloud instance once you've decided.
- [ ] (If Path C) write a focused validation plan for the rollback fix.
- [ ] (If validated) write a "long-run with rollback" config and budget the ~$4 run.
- [ ] (Cleanup) the CoreML export error on iter 4 / iter 8 / iter 12 — non-fatal but noisy. Probably BatchNorm-with-batch=1 in eval-mode export. Defer.
- [ ] (Cleanup) the misleading `Anchor eval disabled via mode_settings['anchor_enabled']=False` log message that fires when anchor is just being skipped (iter 0). Cosmetic.

---

## Spend tracker

| Phase | Cost | Cumulative | Notes |
|---|---|---|---|
| Pre-warm-start cloud experiments | ~$10 | $10 | `REMOTE_TRAINING_INVESTIGATION.md` era |
| Mac smokes v1-v3 | $0 | $10 | Mac compute |
| Cloud supervised pretraining (3× attempts; 10-epoch retained) | ~$0.30 | $10.30 | |
| Cloud Phase 1 first attempt (depth=3 anchor hung at iter_1) | ~$1.50 | $11.80 | Killed |
| Cloud iter-1-5 attempt (preempted) | ~$2.50 | $14.30 | Lost run dir |
| De-risk 12-iter run (anchor disabled) | ~$2 | $16.30 | Saved + analyzed |
| Cross-iter head-to-head (iter_0/3/5/9) | ~$0.10 | $16.40 | Found the gating gap |
| Heuristic eval iter_0 (overshot, mis-budget) | ~$5 | $21.40 | 9-13h grind, but produced verdict |
| Heuristic eval iter_0 (tighter, depth=2 + 100 sims) | ~$0.20 | $21.60 | **30/30 STRONG** |
| **— budget remaining —** | **— $178 —** | $200 cap | |
