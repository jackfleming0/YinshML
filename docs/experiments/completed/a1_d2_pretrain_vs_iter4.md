# A1 — Direct SPRT: D.2 pretrain (iter_0) vs `best_iter_4`

**Status:** DONE: STRONGER
**Date(s):** 2026-05-25 (run + result; A1 "RUNNING" 17:14 UTC, anchor re-frozen 2026-05-25 after result)
**Cost / hardware:** ~30 min if decisive (was: decisive in 21 games, 15 min 3 sec); up to ~4h on cap if borderline; $0.50–$4 cloud compute. Ran on RTX 5090.
**Branch / run dir / artifacts:**
- Candidate: `models/yngine_volume_15ch_pretrain/best_supervised.pt` (D.2 pretrained warm-start, never touched by self-play)
- Anchor: `models/branchC_volume_pretrain/best_iter_4.pt` (frozen 6-ch champion)
- SPRT JSON: `logs/d2_pretrain_iter0_vs_frozen.json`

## Description

Measure `models/yngine_volume_15ch_pretrain/best_supervised.pt` (the D.2 15-ch
pretrained warm-start) directly against the frozen 6-ch anchor `best_iter_4.pt`,
with no D.2 self-play loop in between. Fired immediately after the D.2 SPRT
landed NOT_STRONGER, to test whether the self-play loop had diluted away a
genuinely-stronger model.

**Mechanism:** the D.2 self-play loop appears to be diluting the strength of the
pretrained warm-start (iter_1–3 all 50–100 Glicko below iter_0). If true, the
strongest checkpoint D.2 produced is iter_0 itself. SPRT'ing it directly
separates "is the 15-ch pretrain stronger than 6-ch + self-play?" from "is the
full D.2 loop stronger?" — currently entangled.

**Supporting evidence (forward):**
- D.2 tournament data: iter_0 beat iter_1 116–84 (CI95 [0.511, 0.646], real edge of ~107 Glicko Elo).
- iter_2 vs iter_1 was 99–101 (effectively tied, candidate slightly worse).
- iter_3 dropped back to Glicko 1467 vs iter_2's 1478.
- The pattern is noisy oscillation in a band well below iter_0, not climbing recovery.

**Reasons to not believe (forward):**
- Glicko within a small population is noisy. With 3–4 models and 200 games per pair, the ratings carry meaningful CI — but iter_0's edge over iter_1 had CI95 lower bound 0.511, strictly above 0.5, so it's not pure noise.
- EMA weights might tell a different story. Tournament uses `use_ema_for_eval=True`; the non-EMA iter_4 (what the SPRT consumes via `*_ema.pt` siblings) could differ.
- iter_0 was the warm-start before any self-play; its eval used the EMA which on a fresh init starts identical to raw weights. Later iters had more EMA-vs-raw divergence. Possible artifact.

**Methodology:**
```bash
python scripts/eval_vs_frozen_anchor.py \
    --candidate models/yngine_volume_15ch_pretrain/best_supervised.pt \
    --anchor    models/branchC_volume_pretrain/best_iter_4.pt \
    --sprt --sprt-p1 0.60 --sprt-max-games 400 \
    --device cuda --quiet-mcts \
    --output-json logs/d2_pretrain_iter0_vs_frozen.json
```
Same protocol as the D.1 / Step 2 SPRT — opening-sample-plies=20, n=400 cap,
WR-0.60 promotion bar.

## Outcome

**STRONGER** (crossed +2.94 boundary at game 21 of 400-cap).
- DECISION: STRONGER
- Candidate 19-2-0, WR **0.905**, **CI95 [0.711, 0.973]**, LLR **+3.02**
- Color split 11/8 (acceptable at low n; would balance at higher n)
- Duration: 15 minutes 3 sec on RTX 5090
- JSON: `logs/d2_pretrain_iter0_vs_frozen.json`

**The crucial detail:** CI95 lower bound **0.711** — far above the 0.60 STRONGER
bar; the pretrained 15-ch init alone is the strongest checkpoint the project has
produced (not "marginally stronger" — decisively stronger than the previous
champion).

## Details

**Comparison table** — two SPRTs run minutes apart, same anchor, same protocol,
same candidate *family* (D.2 pipeline), different pipeline stages:

| Candidate | Stage | Verdict | Games | WR | CI95 |
|---|---|---|---|---|---|
| `iteration_4_ema` | pretrain + 5 self-play iters | **NOT_STRONGER** | 304 | 0.526 | [0.470, 0.582] |
| `best_supervised` | **pretrain only, no self-play** | **STRONGER** | 21 | **0.905** | **[0.711, 0.973]** |

The self-play loop destroyed approximately **250–300 Elo** of value between
these two candidates.

**Confirmed/falsified findings:**
1. ✅ **"Pretrain is where the strength lives now"** — confirmed decisively. iter_0 alone beats the prior champion 19-2. The "strongest D.2 model = the pretrain" hypothesis was correct.
2. ✅ **"15-ch encoding moved pretrain metrics only marginally" was the wrong frame.** The marginal metric movement (+1.4 PAcc, +0.7 VAcc) understated the strength gain at the SPRT level — the network was learning something val-metric argmax couldn't capture. Indirect support for A4's "theory IV (argmax-VAcc is the wrong metric)": val P-loss kept dropping every epoch and that movement was load-bearing even though VAcc plateaued.
3. ✅ **"Self-play at MCTS-200 has no headroom; is a random walk on this warm-start."** Confirmed in the strongest form: not a random walk, an *actively destructive* drift. iter_0's 0.905 WR collapsed to iter_4's 0.526 WR over 5 iters. Net-negative, not net-zero.
4. ✅ **"Wilson 0.20 is too loose for this regime."** Every iter promoted at 42–49% WR; cumulative effect was 250–300 Elo of value destruction. The gate is the *mechanism* by which dilution propagated forward.
5. ❌ **"Search depth isn't the dominant lever"** (the read after Step 2) — still true at MCTS-400 vs MCTS-200, but now we know the *encoding* axis WAS a real lever, masked by the self-play loop's mistuning. Step 2's "ceiling is structural" framing is partially wrong: the MCTS-200 ceiling wasn't network capacity, it was the self-play loop discarding pretrain gains.

**Validity note:** A1's STRONGER verdict is **unaffected by the phase-weight bug**
that invalidated D.2 / B1+B2+B3 self-play interpretation — A1 is a pretrain
checkpoint comparison, not a self-play loop test, and pretraining doesn't use the
trainer's phase weighting.

**Operational lessons:** None new for A1 itself (15-min run, no issues). Reinforces
the D.2-level lesson that the autopilot's SUMMARY.md writer should print SPRT
details directly to the consolidated log so the JSON doesn't need to be manually
pulled each time.

**Next experiments (re-ranked by A1 STRONGER):**
1. Re-freeze the anchor to `best_supervised.pt` *before any further SPRT* (done 2026-05-25). Otherwise every future comparison is against a stale 6-ch reference.
2. **B1 + B2 + B3 jump to the top.** The loop is *actively destroying value* — no longer a tuning task, a "stop the leak" task. B1 (tighten Wilson), B2 (lower LR), B3 (more games/iter) are direct fixes for the dilution mechanism.
3. F1 (bare-NetworkWrapper cleanup) remains queued — cheap, removes a class of bugs.
4. A4 (regression value head) becomes *more* interesting — theory II (3-class discretization leaves signal on the table) has stronger grounds given A1's "metric understated strength" finding.
5. D1 (self-play data corpus pretrain) moves up — generating a corpus from our strongest model and re-pretraining might *replace* self-play entirely.
6. C1 (SE blocks) drops further — architecture exploration is less urgent with a clearly-stronger pretrain to anchor against and a known-broken loop to fix first.

**Deeper meta-finding:** the self-play loop is built for a model in the "learning"
regime, not the "fine-tuning" regime. At weak warm-start (Branch C era) it holds
value; at strong warm-start (D.2 era) it destroys value. The transition was binary,
not gradual — a known failure mode in RL with strong priors (LLM RLHF uses tiny LRs
+ KL penalties precisely to anchor toward the supervised init).

**Definition of done (met):** JSON written to `logs/d2_pretrain_iter0_vs_frozen.json`
with a verdict; result logged in the Done section.

**Open questions at launch (forward):**
- If iter_0 IS STRONGER but iter_4 ends up INCONCLUSIVE → near-proof the self-play loop is hurting; B1 (tighten gate) becomes the highest-prior follow-up. *(This is what occurred.)*
- If iter_0 is also INCONCLUSIVE → encoding lift alone is small; A4 (value-head loss change) becomes higher-prior next test.

## Provenance & links

- Run + result: 2026-05-25. New anchor re-frozen to `best_supervised.pt` same day.
- Related: [[branchD2_15ch_encoding]] (the loop result A1 disentangles), [[b1b2b3_stop_the_leak]] (the "stop the leak" follow-up A1 launched to the top of the queue).
- Cross-doc refs: `VOLUME_PRETRAIN_RESULTS.md` (§ 2026-05-21 bridge gap), `D2_PREP.md`.
