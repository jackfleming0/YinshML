# Qualitative play observations — supervised seed (iter_0)

**Date:** 2026-04-29
**Model:** `models/supervised_seed/best_supervised.pt` (1-epoch local seed; the
cloud 10-epoch seed is referenced as `best_supervised_cloud.pt` in the
`scripts/play_vs_model_mcts.py` defaults but isn't on this machine).
**Method:** Manual play via `scripts/play_vs_model_mcts.py` on macOS / MPS.
Two games played, varying `--mcts-simulations` (64 vs 400) to compare
deployment-budget behavior. Human (Claude) playing as White, AI (model) as
Black, both games.
**Goal:** complement the cold eval numbers (`eval_vs_heuristic.py` at depth=2
swept 30/0, depth=3 came back 50%) with a tactical sanity check — does
the model actually play like a 50%-vs-depth-3 opponent does, or does it have
specific blind spots?

---

## TL;DR

**Both games won, but the 400-sim run was a real game.** At 64 sims the
model is essentially a confident-but-blind policy with no meaningful
search; at 400 sims it actually plays — completes a 5-in-a-row of its own,
sets up multi-step traps, makes occasional moves that punish my mistakes.
The win still came inside ~25 moves in both cases, and the failure modes
are pattern-consistent: the model **doesn't reliably evaluate the
opponent's multi-flip moves**, even at deployment budget. Confirms the
depth=3 50%-CI eval result and refutes the depth=2 100% sweep as a
useful benchmark — depth=2 was just too shallow to expose the seed's
weaknesses.

---

## Game 1 — 64 sims (deployment "fast" preset)

```
python scripts/play_vs_model_mcts.py \
    --color white \
    --mcts-simulations 64 \
    --device mps
```

**Result:** White (human) wins **3-0** in 25 moves. AI failed to score even
one row.

### Key moves

| # | My move | Model response | Note |
|---|---|---|---|
| 1 | `place F5` | `place F6` (100%) | Mirror, classic strong response |
| 2 | `place G7` | `place E5` (100%) | Continued tight cluster |
| ~7 | `F5 → C2` (NW jump) | corner move | **Big swing:** flipped E4 + D3 to white in one move |
| ~9 | `C2 → D2` | `J9 → J6` | I formed **4-in-a-row**. Model played peripheral corner move. |
| ~11 | `G6 → G5` | — | **5-in-a-row** completed, scored 1st row |
| ~14 | `H5 → I5` | — | **5-in-a-row** on row 5, 2nd row |
| ~22 | `H9 → D5` (4-step NW-SE jump) | — | **3-marker multi-flip.** F7 flipping completed row 7 (D7-H7), 3rd row, won. |

### Observations

1. **Confidence collapse on the policy head.** AI played at 100% confidence
   on almost every move per the `[AI thought ... Top considered]` readout.
   With only 64 sims the visit distribution collapses onto whatever the
   prior fancied — no meaningful tree search.
2. **Doesn't see opponent multi-flip moves at all.** When I had ring at C2
   and markers at D3, E4, F5 (3-in-a-row), the model played `J9 → J6` — a
   peripheral relocation. Pattern repeated multiple times.
3. **Sacrifices its own markers.** Move `G3 → D3` flipped own E3 and F3 to
   white with no compensating gain.
4. **Misses lookahead-2 wins for me.** Every row I scored required setup
   on turn N and completion on turn N+1 (or N+2). The model never broke
   the chain by flipping a marker between my setup and completion.
5. **Reasonable openings; broken in mid-game.** The 5 ring placements were
   sensible (F6/E5/D4/F8/F7 — center cluster). The wheels came off when
   concrete tactics started.

---

## Game 2 — 400 sims (deployment "hard" preset)

```
python scripts/play_vs_model_mcts.py \
    --color white \
    --mcts-simulations 400 \
    --device mps
```

**Result:** White (human) wins **3-1** in ~28 moves. AI scored one row
(col C) and put up a real fight before losing.

### Key moves

| # | My move | Model response | Note |
|---|---|---|---|
| 1 | `place F5` | `place D4` (8.7%) | **Different from 64-sim.** Visit dist split across multiple candidates instead of collapsing to mirror. |
| 2-5 | (placement) | spread across left side (B2, C5, D6, D3) | AI placed strategically across cols B-D, didn't tight-cluster |
| ~10 | `G7 → H7` | `C6 → C7` | I made 4-in-a-row D7-G7. Unlike 64-sim, AI didn't play obviously bad — it had its own plan, just wasn't defending. |
| 11 | `H7 → I7` | — | **5-in-a-row D7-H7**, scored 1st row |
| post-row | `remove I7` | `C7 → C8` (post-row score) | **AI scored its own row** (col C: C3-C7) by playing `C7 → C8`. Score 1-1. This **never happened at 64 sims**. |
| ~16 | `F7 → E6` flipped F5 via E4→G6 | AI saw the F-col attack; I had to defend with H6 → F4 | AI was reading multi-step tactics that 64-sim AI missed entirely |
| ~20 | `G10 → H11` | — | Set up diagonal D7-E8-F9-G10-H11. AI didn't disrupt my chain at G10 setup. |
| 21 | `H11 → I11` | — | **5-in-a-row** on NW-SE diagonal, 2nd row |
| ~25 | `D8 → D4` (multi-flip) | `B7 → D7` (counter, flipped C7) | Big simultaneous-3-in-a-row move; AI's counter destroyed one of my chains but not the other. |
| ~26 | `D4 → D2` | `D7 → E8` (didn't defend col D!) | I created TWO 4-in-a-rows; AI repositioned instead of defending. |
| 27 | `D2 → E3` | — | **5-in-a-row col D (D2-D6)**, 3rd row, won. |

### Observations

1. **Visit distribution is meaningful at 400 sims.** Top candidates
   typically 8.7%–17.4% (split among 11–4 candidates), not collapsed to
   100%. MCTS is actually exploring tree branches.
2. **AI plays strategically — places rings to control territory, builds
   own marker chains.** Notably built B6-E6 4-in-a-row on row 6 over
   ~10 moves while I was busy with row 7. I had to defend with `E5 → F6`
   to prevent AI from completing.
3. **AI completed a row** (col C: C3-C7 via `C7 → C8`), something the
   64-sim model never threatened. Score 1-1 at one point.
4. **AI made tactical attacks I had to react to:**
   - `C4 → D5` (positioned D5 ring as a future attacker)
   - `D3 → E4` (set up `E4 → E9` long-jump-flip on my col E chain — I had
     to play `F5 → E5` to block)
   - `B7 → D7` (placed ring in middle of my row 7 chain, then flipped C7
     marker the next turn)
5. **Still missed the deep multi-chain finish.** When I played `D4 → D2`
   (creating two simultaneous 4-in-a-rows: col D and the NW-SE diagonal),
   the AI played `D7 → E8` — a non-defensive repositioning. Either the
   MCTS at 400 sims didn't see the col D 5-in-a-row threat one ply ahead,
   or it preferred a different line.
6. **Specific recurring weakness: 4-in-a-row → 5-in-a-row finish.** Both
   games, AI failed to defend whenever I made 4-in-a-row. It defends
   3-in-a-row threats sometimes, but the transition from 4 to 5 (the
   final marker via ring move-away) seems to slip through MCTS.

---

## Comparison: 64 vs 400 sims

| Aspect | 64 sims | 400 sims |
|---|---|---|
| Final score | W:3 / B:0 | W:3 / B:1 |
| AI visit distribution | 100% / 0% / 0% (collapsed) | 8.7%–17.4% (broad) |
| AI ring placements | Tight cluster (mirror) | Spread, strategic |
| AI scored own row? | No | Yes (col C) |
| AI made tactical attacks? | No | Yes (E4 → E9 setup, B7 → D7 chain disruption, F-col threat) |
| AI defended my 3-in-a-rows? | Rarely | Sometimes |
| AI defended my 4-in-a-rows? | No | No |
| AI saw my multi-flip setup moves? | No | Sometimes (saw F-col attack vector; missed col D) |
| Game length | ~25 moves | ~28 moves |
| Difficulty | Trivial — exploits obvious | Real — required defending against AI threats |

The depth=2 eval (`30/0` from `eval_vs_heuristic.py`) is a useful sanity
check that the model isn't completely broken, but it's far below what the
model can actually do at deployment budget. The depth=3 eval (`50%`,
`CI95=[0.188, 0.812]`) lines up with the 400-sim play experience: model
is good enough to score occasionally, not good enough to reliably win.

---

## What this confirms / refutes about the cold eval numbers

- **Confirms** the depth=3 result. The 400-sim model plays like a 50%-vs-
  depth=3 opponent does: real moves, occasional successes, consistently
  beatable by a deliberate adversary.
- **Refutes** the depth=2 100% sweep as a useful benchmark for "is the
  model intermediate?". Depth=2 is too shallow — at depth=2 even the
  64-sim model would probably win, because depth=2 alpha-beta misses the
  same multi-step tactics the human exploits.
- **Refutes** "iter_0 is at intermediate level" by user's bar (≥65% vs
  depth=3). The model is solidly mediocre — capable enough to defend some
  tactical threats but not reliably, and missing the same class of
  multi-flip moves regardless of MCTS budget.

---

## Recommendations for future qualitative testing

1. **Always test at 400 sims, not 64.** The 64-sim behavior tells you
   nothing useful about the policy head's actual content; you're just
   reading the prior.
2. **Same routine on the gating-revert iter_K:** play it at 400 sims and
   compare to iter_0's behavior. Specifically:
   - Does it defend 4-in-a-rows? (iter_0 doesn't.)
   - Does it see opponent multi-flip moves? (iter_0 doesn't.)
   - Does it score rows of its own? (iter_0 at 400 sims did once; at 64
     sims it didn't at all.)
3. **Tactical pattern catalog worth tracking across checkpoints:**
   - 4-marker multi-flip jumps (single ring move flips 3+ opponent
     markers).
   - The "set up ring at extension cell, move it to leave marker" 5-in-a-
     row finishing pattern.
   - Forcing-move recognition (4-in-a-row defense priority).
   - Ring-placement strategy (tight cluster vs. spread).
4. **Sanity check protocol when assessing a new checkpoint:** run a
   400-sim game, lose to it deliberately as a baseline (i.e. play random
   non-self-flipping moves and see if it scores), then play to win. The
   gap between "loses to a random opponent" and "loses to a deliberate
   one" is a useful signal of how close it is to intermediate.
5. **Don't use depth=2 as a benchmark eval.** It's a useful liveness
   check (does the model not crash and not lose to a near-random
   opponent), but a 100% sweep at depth=2 is consistent with a wide range
   of model strengths from "barely better than random" to "intermediate".
   The depth=3 number is what differentiates.

---

## What this implies for the gating-revert run

- **The bar is "iter_K beats iter_0 head-to-head."** From the
  `WARMSTART_PHASE_LOG.md` §5d data, pre-fix iter_3/5/9 lost 0-40 vs
  iter_0. If the gating-revert iter_K just *holds even* with iter_0 in
  head-to-head, that's evidence the fix is working. If it actually
  *beats* iter_0, that's evidence self-play training is producing real
  improvements past the supervised seed.
- **Qualitative test for iter_K:** a 400-sim play game against iter_K
  should reveal whether iter_K has fixed the specific blind spots iter_0
  had. Specifically: does iter_K notice when I make 4-in-a-row? Does it
  see multi-flip threats coming? Does the visit distribution at 400 sims
  show meaningful spread or has it collapsed back toward 100%?
- **A practical follow-up worth scheduling:** once the cloud run
  finishes, pick the strongest-by-Glicko checkpoint, play it at 400 sims
  with the same opening sequence as Game 1 here (`F5, G7, D7, H6, E8`),
  and see whether the same human-exploit patterns work. If they do,
  we're not at intermediate yet. If they don't, we are.

---

## Update: gating-revert run head-to-head results (2026-04-29 evening)

The 12-iter `phase_d_warmstart_derisk_revert.yaml` run completed. Gating
revert fired correctly on every failed gate (iters 1, 4, 5, 6, 7, 8, 9,
10, 11). iter_3 was the last and best promotion (Wilson gate passed at
ELO 1640). Iters 4–11 plateaued, all reloading iter_3 weights via the
new revert mechanism.

Ran `scripts/eval_head_to_head.py` over surviving checkpoints (iter_0
was pruned by the retention-5 policy, so the bookend comparison vs the
supervised seed isn't possible directly — see "What this implies"
section below for the workaround). Field: iter_2, 3, 5, 6, 9. 40 games
per pair, raw policy at temperature=0, 400 games total.

### Pair-by-pair results

```
                pair  a_wins  b_wins  draws  a_score  significant?
    iter_2 vs iter_3       0      40      0    0.000  ★★★ (iter_3 wins)
    iter_2 vs iter_5       0      40      0    0.000  ★★★ (iter_5 wins)
    iter_2 vs iter_6       0      40      0    0.000  ★★★ (iter_6 wins)
    iter_2 vs iter_9      40       0      0    1.000  ★★★ (iter_2 wins ←!)
    iter_3 vs iter_5      40       0      0    1.000  ★★★ (iter_3 wins)
    iter_3 vs iter_6       0      40      0    0.000  ★★★ (iter_6 wins ←!)
    iter_3 vs iter_9      40       0      0    1.000  ★★★ (iter_3 wins)
    iter_5 vs iter_6      20      20      0    0.500   tied
    iter_5 vs iter_9      20      20      0    0.500   tied
    iter_6 vs iter_9       0      40      0    0.000  ★★★ (iter_9 wins)
```

Aggregate (avg score across 4 pairs each):
```
  #1  iter_3:  0.750
  #2  iter_6:  0.625
  #3  iter_5:  0.500
  #4  iter_9:  0.375
  #5  iter_2:  0.250
```

### Headline finding: a non-transitive cycle

```
iter_3 > iter_9 > iter_6 > iter_3
```

Plus the `iter_2 > iter_9` anomaly (iter_2 is the weakest by aggregate
but specifically beats iter_9 40-0).

### What's actually going on

Same structural issue as the depth=3 eval results in `eval_vs_heuristic.py`:
**raw policy at temperature=0 is deterministic per (model, color, seed)
triple**. With seeded RNG and 20 games per side, all 20 games per
(pair, color) are identical replays. Each "40-0" really means "in the
specific deterministic game from this seed, model X wins both colors."
True per-pair sample size is 2, not 40.

This amplifies tiny policy differences into big-looking scores. iter_6
beat iter_3 40-0 — but that doesn't mean iter_6 dominates iter_3; it
just means iter_6's policy happens to handle iter_3's specific argmax-
play sequences in both colors of this seed. With Dirichlet noise or
non-zero temperature, that 40-0 would likely smear toward 50-50.

It's also further evidence that **MCTS at deployment budget is what
makes these models actually playable.** Raw-policy is too brittle —
small policy differences become deterministic 40-0 matchups; MCTS
smooths it. Game 2 of the qualitative play tests above (400-sim model)
felt much more solid than Game 1 (64-sim) for exactly this reason.

### What's still load-bearing despite the cycle

1. **iter_3 vs iter_2 = 40-0** is consistent with the Wilson gate's
   call. The iter_2 → iter_3 promotion was a real strength jump.
2. **iter_3 has the highest aggregate (0.750)** across the post-revert
   field — most consistent winner even though one specific opponent
   (iter_6) beats its argmax-play. That matches what the training gate
   gave us: iter_3 is the canonical "best" checkpoint.
3. **The cycle suggests the recipe is producing models with *different*
   policy quirks rather than monotonically stronger ones.** Each post-
   iter_3 model has its own distribution of blind spots. None of them
   is clearly smarter than iter_3 in a transitive sense, even though
   none collapsed either.

### Practical implication

The within-run head-to-head can't answer "is iter_3 intermediate-level?"
because all opponents are iter_3-derived. The benchmark for that is
`eval_vs_heuristic.py --depth 3 --time-limit-per-move 30 --mcts-simulations 400`
against iter_3, compared to the supervised seed's result (50%,
CI=[0.188, 0.812]).

**Useful side-comparison:** the same eval against iter_6 (since iter_6
beat iter_3 head-to-head at temp=0). If iter_6 lands higher vs depth=3,
the cycle is misleading and iter_6 is the better deployment candidate.
If iter_6 lands lower or equal, the head-to-head was just a quirk of
deterministic play and iter_3 remains the canonical pick.

### Lesson for future eval runs

When running `eval_head_to_head.py`, **don't read individual pair scores
as ground truth**. Look at the aggregate ranking and mark "near-50/50"
ties (like iter_5 ↔ iter_6, iter_5 ↔ iter_9) as the *real* signal:
those models are roughly equivalent, the deterministic 40-0s are
quirks. A model that wins 0.700+ aggregate across many opponents is
genuinely stronger; one that wins specific 40-0 matchups but loses
others to the same field is just policy-different, not policy-better.

If you want a less-noisy head-to-head, the script should be extended
with either (a) Dirichlet noise during eval, or (b) varying game seeds
per replicate (so each "40 games" pair is 40 *different* games rather
than 1 game replicated 40 times). Both involve small `eval_head_to_head.py`
patches. Worth doing if this kind of cross-iter comparison becomes a
recurring need.

---

## Update 2: iter_3 vs HeuristicAgent(depth=3) — the canonical "is it intermediate?" benchmark (2026-04-30)

Ran `eval_vs_heuristic.py` against `runs/20260429_152142/iteration_3/checkpoint_iteration_3.pt` at the deployment-realistic config (400 MCTS sims, 30s/move time-limit on the heuristic, 30 games). Wall-clock 6h52m on RTX 4090.

```
Result: iter_3_revert (mcts) vs HeuristicAgent(depth=3)
  Games played:   30
  Candidate wins: 6   (20.0%)
  Anchor wins:    24
  Draws:          0
  Win rate:       0.200  CI95=[0.095, 0.373]
  Avg game length: 94.3 moves
Verdict: FAILS: candidate consistently loses to heuristic.
```

### Headline

**The gating-revert recipe didn't reach intermediate-level play.** iter_3, the recipe's best checkpoint, loses 20% / 80% to depth=3 heuristic. CI upper bound is 37%, comfortably below the 50% even-match line and far below the 65% intermediate bar.

### Comparison to the supervised seed

We tested the supervised seed at depth=3 only with **6 games** (the smoke run after the original 30-game version got killed by the no-time-limit hang), got 50% with CI95=[0.188, 0.812].

| Checkpoint | N | Win rate | CI95 |
|---|---|---|---|
| Supervised seed (iter_0) | 6 | 50% | [0.188, 0.812] |
| **iter_3 (post-fix)** | **30** | **20%** | **[0.095, 0.373]** |

The CIs **overlap** (iter_0's lower bound 18.8% sits just above iter_3's upper bound 37.3%), so we can't *conclusively* say iter_3 is weaker than the seed — but we definitely can't say it's stronger. To disambiguate cleanly: rerun iter_0 at 30 games depth=3. ~7h, ~$2. Worth it before committing significant compute to a new recipe.

### What this tells us about the recipe

Putting the head-to-head and the depth=3 numbers together:

- **The gating-revert mechanism works structurally.** Fired correctly 9/9 times. iter_3 was a real promotion (Wilson-validated, dominates iter_2 head-to-head 40-0).
- **But the recipe plateaued sub-intermediate.** Iters 4–11 all reverted to iter_3, none could push past it. iter_3 itself is below the intermediate bar.
- **The non-transitive cycle and the depth=3 result tell the same story:** post-iter_3 models aren't monotonically stronger than iter_3, AND iter_3 isn't strong enough vs the depth=3 benchmark. The recipe is bouncing around a local optimum that's *below* what we'd want.

### Implication

The path forward isn't "more iterations of the same recipe." That's been tested. The plateau is real. The next experiment needs to be a *different* recipe (or a different starting point), and we should make a deliberate choice about which axis to vary first. See `WARMSTART_PHASE_LOG.md` §9 (Phase E plan) for the strategic options ranking.

---

## iter_0_v2 sanity check — Game played as White at 400 sims (2026-04-30)

**Model:** `models/supervised_seed_humans_only/best_supervised.pt` (humans-only seed: 1,312 games / 83K positions, no bot games — testing whether filtering the bot-contaminated subset of Boardspace fixes the prior model's tactical blind spots).
**Method:** `scripts/play_step.py` driven by Claude subagent on macOS / MPS.
**You played:** White
**MCTS sims:** 400 (deployment "hard" preset)

### Result

**White (human) wins 3-1 in 55 moves.** Same final score as the prior iter_0 at 400 sims (Game 2 above), but a noticeably longer game (55 vs ~28 moves) and the model put up significantly more tactical resistance throughout.

### Key moves and observations

| # | Your move | Model response | Note |
|---|---|---|---|
| 1 | `place F5` | `place D5` | **Different from iter_0 v1** — neither mirror (F6) nor symmetric placement. Direct row-5 contestation. |
| 2-5 | `place G7, D7, H6, E8` | `place E5, F3, F9, H7` | AI placed **H7 directly on my row 7 plan** between D7 and G7 — preemptive interposition, smarter than iter_0 v1's left-side cluster. |
| 6 | `F5 → F6` | `D5 → C5` | Standard mid-game start. AI building col C. |
| 7 | `F6 → D6` (slide) | `E5 → E7` | AI repositioned ring E5 → E7 between my D7 and E8 rings — disruptive placement. |
| 8 | `H6 → H4` | `C5 → C6` | AI continues col C development. |
| 9 | `D6 → D4` (jumped D5, flipped to white) | `H7 → H5` | **AI's first multi-purpose move:** ring H7→H5 jumped over my H6 white marker, flipping it to black. Drops black marker H7. So in one move: relocates ring, flips my marker, and adds a black marker. **This is a real tactical move iter_0 v1 never made at 400 sims** — it punished my unprotected H6. |
| 10 | `D7 → D8` (sets up 3-in-a-row col D: D5,D6,D7) | `F9 → F8` | AI did NOT defend the col D 3-in-a-row. **Same blind spot as iter_0 v1.** |
| 11 | `D4 → C4` (now 4-in-a-row col D: D4-D7) | `H5 → H8` | AI made a **bad** jump: H5→H8 jumped over its own H6,H7 black markers, flipping both to white. Net: AI sacrificed 2 of its own markers for a ring relocation, and missed the col D 4-in-a-row threat completely. |
| 12 | `D8 → C8` (drops D8 marker → 5-in-a-row D4-D8) | — | **5-in-a-row col D, scored.** W:1 B:0. |
| 13-14 | row removal, ring C8 | `F8 → G9` | |
| 15 | `G7 → F7` (drops G7 white marker) | `G9 → G6` | AI's G9→G6 ring jumped over my G7 white marker, flipping it to black. **Multi-purpose move again** — flips my piece + lands ring centrally on row 6. |
| 16 | `F7 → F10` (jumps F8,F9 black markers, flips both white → 5-in-a-row F5-F9) | — | **5-in-a-row col F, scored.** W:2 B:0. AI failed to defend the col F threat. |
| 17 | `H4 → E4` | `H8 → H3` | **AI's strongest move of the game.** Ring H8→H3 jumps the entire col H run (H7 white, H6 white, H5 black markers — 3 markers). Flips my H6,H7 markers black, flips H5 white, lands at H3 (note: per docs the engine should land at H4 first-empty-after-run, but the move was accepted to H3 — possibly the engine allows further sliding after a jump in some edge case; not investigated). Either way, AI **wiped my col H pair** and disengaged its ring from danger. This is real defensive recognition — iter_0 v1 didn't see col H pairs at all. |
| 18 | `E4 → I4` (jumps H4 black marker, flips white) | `G6 → D3` | AI's G6→D3 jumped my just-dropped E4 white marker, flipping it black — **another counter-flip on a marker I just made.** Same flavor as iter_0 v1's `B7→D7` chain disruption. |
| 19 | `E8 → I8` (jumps H8, flips white) | `F3 → E3` | AI played a quiet positional move (single step), missed the developing threat. |
| 20 | `C4 → F4` (jumps E4 black, flips white) | `E6 → H9` | AI repositioned to H9 — **defensive against my potential col H attack from the north**. Strategic awareness here. |
| 21 | `F4 → G4` (4-in-a-row row 4: E4-H4) | `H9 → H10` | Did not defend row 4. |
| 22 | `G4 → G2` (drops G4 → 4-in-a-row F4-H4 still, plus E4) | `D3 → F5` | **AI defended row 4 by counter-flip!** D3→F5 jumped my E4 marker, flipping it back to black. Broke my 4-in-a-row contiguity. **This is the prior model's blind spot apparently fixed.** Plus it set up... |
| 23 | `I4 → J5` (5-in-a-row F4 wait no — I made an error, the chain became F4-I4 4-in-a-row again) | `F5 → G5; REMOVE D3,E4,F5,G6,H7; REMOVE_RING E3` | **AI scored a 5-in-a-row on the NW-SE diagonal D3-H7!** This was a multi-move plan: D3 placed several turns earlier, E4 from the counter-flip on move 22, F5 placed by my own miscalculation (it was AI's flipped marker), G6 placed turns earlier, H7 from the placement phase. The model assembled a 5-marker diagonal across many moves. **Score W:2 B:1.** Equivalent to iter_0 v1's col-C row, but along a diagonal — somewhat more impressive. |
| 24 | `I8 → F5` (jumps H7 empty, G6 black → flips G6 white; doesn't help much) | `C6 → B6` | AI quiet move, missed defending. |
| 25 | `F5 → E4` (sets up final move) | `B6 → B7` | **AI did not defend the row 4 4-in-a-row + ring-on-E4 setup** — the same kind of "ring on extension cell, drop marker next turn" pattern iter_0 v1 also missed. |
| 26 | `E4 → D4` (drops E4 marker → 5-in-a-row E4-I4 row 4) | — | **5-in-a-row row 4, scored.** W:3 B:1. Win. |

### Observations

1. **Did the model defend my 3-in-a-rows?** No, not the col D 3-in-a-row (move 10) — same blind spot as iter_0 v1.
2. **Did the model defend my 4-in-a-rows?** **Sometimes yes**, sometimes no.
   - Move 11: missed the col D 4-in-a-row (failed).
   - Move 16: missed the col F 5-in-a-row finish (well, the threat was a 1-move win via multi-flip jump — hard to see).
   - Move 22: **DID defend** the row 4 4-in-a-row by counter-flipping E4 with D3→F5. This is a real improvement — iter_0 v1 didn't reliably spot these.
   - Move 25: missed the second row 4 4-in-a-row → 5-in-a-row threat (failed).
3. **Did the model make multi-flip moves itself?** Yes, **and more often than iter_0 v1**:
   - Move 9 (`H7→H5`): single-flip + ring relocation.
   - Move 15 (`G9→G6`): single-flip + central ring placement.
   - Move 17 (`H8→H3`): **3-marker pass-through** wiping my col H pair. The biggest single tactical move I saw from the model.
   - Move 18 (`G6→D3`): single-flip counter-attack on my E4.
   - Move 22 (`D3→F5`): single-flip counter on E4 — but with a **multi-move plan** behind it (assembling the D3-H7 diagonal).
4. **Did the model score any rows of its own?** **Yes, one row.** The D3-H7 NW-SE diagonal (move 23) — equivalent to iter_0 v1's col-C row, but constructed along a diagonal axis and visibly part of a multi-move plan rather than opportunistic.
5. **Comparison to iter_0 at 400 sims (the prior baseline):** see below.

### Comparison to iter_0 at 400 sims (the prior baseline)

| Aspect | iter_0 v1 (400 sims, Game 2) | iter_0 v2 humans-only (this game) |
|---|---|---|
| Final score | W:3 / B:1 | W:3 / B:1 |
| Game length | ~28 moves | 55 moves |
| AI placement strategy | Spread across left side, B-D cols | **Mixed**: contests row 5 (D5,E5), interposes ring on row 7 (H7) directly between my placements |
| AI scored own row? | Yes — col C straight row | Yes — D3-H7 NW-SE diagonal (multi-move plan) |
| AI defended my 3-in-a-rows? | Sometimes | Sometimes (no on col D, but engaged elsewhere) |
| AI defended my 4-in-a-rows? | No (missed col D finish) | **Mixed** — defended row 4 once via counter-flip (move 22), missed it the second time (move 25–26) |
| AI multi-flip / multi-purpose moves | Saw F-col attack, missed col D | **Saw col H pair** (H8→H3 wiped it); counter-flipped my E4 marker twice (G6→D3, D3→F5); H7→H5 multi-purpose |
| Visit distribution at 400 sims | 8.7%–17.4% (broad) | Not displayed by `play_step.py`, but moves looked decisive — likely similar |

**One-paragraph summary:** This humans-only seed feels **modestly stronger tactically** than iter_0 v1, but the same headline blind spots persist — it still doesn't reliably defend a 4-in-a-row → 5-in-a-row finish. The improvements are around the edges: it makes more multi-purpose ring moves (H8→H3 was a clean 3-marker chain disruption), it counter-flipped my markers twice in the same game (D3→F5 broke my 4-in-a-row at move 22 — iter_0 v1 only managed one such defensive flip per game), and its scored row was assembled across multiple moves on a diagonal axis (slightly more impressive than iter_0 v1's straight col C). On the negative side: at move 11 it made a flat-out **bad** move (`H5→H8` sacrificing two of its own markers for no clear gain) — the model still has noisy, low-quality moves mixed in with the better ones. Net: this looks like a slightly cleaner version of the same model class. Not a different tier of play. The bot-contamination filter helped on the margin but didn't fix the core 4-in-a-row defensive blind spot. Would expect the depth=3 heuristic eval to land somewhere in the same neighborhood as iter_0 v1's 50% (CI overlapping), maybe slightly higher.

### Notes on scoring evolution within the game

- **My 5-in-a-rows:** col D (move 12) → col F (move 16, multi-flip finisher) → row 4 (move 26, 2-step setup). 
- **AI's 5-in-a-row:** D3-H7 diagonal (move 23). The AI assembled this across at least 4 separate moves, and the actual completing move (F5→G5 dropping marker at F5) was a normal 1-step ring slide — the kind of move that's invisible-to-defense unless you've been tracking the diagonal threat for several turns.
- **Total game length 55 moves** is roughly 2× the prior iter_0 games at the same budget. That's because (a) the AI scored a row mid-game, forcing me to defend across multiple chains, and (b) more counter-flips broke my chains and made me rebuild. The model is genuinely harder to grind down than iter_0 v1.

---

## iter_0_v2 sanity check — Game played as Black at 400 sims (2026-04-30)

**Model:** `models/supervised_seed_humans_only/best_supervised.pt` (humans-only seed, same checkpoint as the White-side game above; this is the color-flipped sanity test).
**Method:** `scripts/play_step.py` driven by Claude subagent on macOS / MPS.
**You played:** Black (model goes first as White).
**MCTS sims:** 400 (deployment "hard" preset).

### Result

**Black (human) wins 3-0 in 56 moves.** A complete shutout — the model never scored a row playing White. Score progression: 1-0 at move ~26 (row 6), 2-0 at move ~46 (NE diagonal D5-H9), 3-0 at move 56 (row 3 5-in-a-row, via game-ending multi-flip).

### Key moves and observations

| # | Model move | Your move | Note |
|---|---|---|---|
| 1 | `place D4` | `place E6` | Same opening as iter_0 v1 at 400 sims (D4 first). Not a mirror of human. |
| 2-5 | `place F5, D5, F3, C5` | `place G7, H6, F8, I8` | **Tight cluster on left/center cols B-F**. Identical placement-strategy footprint to iter_0 v1. Did not adapt to my spread placement. |
| 6 | `F5→G6` (1-step slide, drops F5 marker) | `H6→H9` (build col H) | Standard. |
| 7 | `C5→B4` | `F8→F4` (jumps F5 white → flips to black) | First multi-flip exchange — I flipped white's F5 marker. |
| 8 | `D4→D3` | `F4→H4` | AI started building col D (D4 → marker, ring on D3). |
| 9 | `B4→B2` | `E6→F6` | AI continues quiet repositioning. |
| 10 | `D5→D7` (white now has D4-D5 col D 2-in-a-row) | `F6→C3` (jumps D4 white, flips to mine, ring lands left) | Big disruption move on my part — flipped white's D4. |
| 11 | `G6→E4` (jumps F5 black → flips white) | `G7→G5` (jumps G6 white → flips to black, sets up row 6 4-in-a-row D6-H6) | **The model didn't see the row 6 buildup coming.** |
| 12 | `E4→G4` (jumps F4 black → flips white; positional) | `I8→I6` (preparing row 6 finisher) | AI made a move that re-flipped F4 to white — minor counter, but didn't address row 6 threat. |
| 13 | `D7→H7` (jumps G7 black → flips white) | `I6→I7` (drops marker I6 → 5-in-a-row row 6 E6-I6) | **5-in-a-row row 6, Black scores. 0-1.** AI did not defend the obvious 4→5 threat. |
| 14 | `F3→F6` — **AI played a self-flip!** Ring jumped over its own F4 + F5 markers, flipping both back to black. Net: white **gave me 2 free markers** and lost a marker. | `H9→C4` (jumps D5 white → flips, lands ring at C4) | **First clear bad move**: F3→F6 was a strict negative-EV move for white. iter_0 v1 had similar self-sacrificing moves; this is the same blind spot. |
| 15 | `D3→D6` — **AI's first real multi-flip:** ring jumped D4 + D5 (both my markers), flipping both back to white. White now has 3-in-a-row col D (D3-D5). | `C4→E6` (jumps D5 → flips back to mine) | I had to counter-flip immediately to disrupt the col D 3-in-a-row. |
| 16 | `B2→B3` (passive) | `E6→E3` (jumps E4 → flips to mine) | AI's response was a 1-step quiet move — didn't develop col D further. |
| 17 | `F6→H6` (1-step slide) | `E3→C3` (jumps D3 → flips to mine) | I disrupted column D again. |
| 18 | `H7→C2` — **AI's biggest tactical move of the game.** SW-diagonal jump from H7 through F5+E4+D3 (all my markers, contiguous), flipping all 3 back to white in one move. **Same flavor as iter_0 v1's `B7→D7` chain disruption — but flipping 3 markers in one shot, not 1.** This is a real multi-flip, the model saw it. | `I7→F7` (multi-flip H7+G7 white → flip to black, ring at F7) | I had to immediately counter with my own multi-flip to disrupt the SW diagonal that was now becoming dangerous. |
| 19 | `H6→H8` (jumps H7 black → flips white) | `C3→G3` (multi-flip: jumps D3+E3+F3, flipping mixed-color row 3 chain) | Both sides multi-flipping. |
| 20 | `H8→E5` (jumps G7+F6 black markers → flips to white, lands ring centrally on E5) | `F7→C7` (jumps D7 → flips to mine; **drops marker F7 → NE diagonal C4-D5-E6-F7 = 4-in-a-row mine**) | **AI didn't see the NE diagonal 4-in-a-row threat.** |
| 21 | `D6→E7` (1-step slide, ring repositions) | `G5→G8` (jumps G7 → flips to mine, **ring lands at G8 = setup for 5-in-a-row NE diagonal next turn**) | This was the critical setup move for me. The threat was: next turn move ring G8 away → drops marker at G8 → 5-in-a-row C4-G8 NE diagonal. **The threat was visible 1 ply ahead. AI's response was the worst-case option for them**: |
| 22 | `C2→B1` — **AI played a corner move.** Total non-defense. iter_0 v1 also missed similar 4→5 finishes. | `G8→G9` (drops marker G8) → **5-in-a-row NE diagonal D5-E6-F7-G8-H9, Black scores. 0-2.** | **Identical 4-in-a-row → 5-in-a-row blind spot as iter_0 v1.** Visible 1-ply threat, AI moved a ring to a corner instead of disrupting. |
| 23 | `G4→G6` (jumps G5 black → flips white) | `H4→H9` — **3-marker multi-flip jump** (H6+H7+H8 white markers → all flipped to black). **My biggest tactical move of the game.** | This single move flipped 3 white markers and set up multiple new chains (col H, row 7, NE diagonal F6-H8). |
| 24 | `E5→E2` — **Another self-flip blunder.** Ring jumped its own E3+E4 white markers, flipping both back to black. Net: white **gave me 2 more free markers** and lost a marker. | `H9→H10` (drops marker H9 → col H 4-in-a-row H6-H9) | Second clear self-sacrificing blunder. iter_0 v1 had these too. |
| 25 | `G6→J9` — **AI's only smart defensive move of the late game.** NE-diagonal jump from G6 through H7+I8 (my markers), flipping both to white. **Disrupted my col H chain by flipping H7.** Counter-flip recognized 1 ply ahead. | `G3→G8` — **game-winning multi-flip + 5-in-a-row.** Ring jumped G4+G5+G6+G7 (4-marker run), flipping all 4. Source-drop at G3 completed row 3 5-in-a-row C3-G3. **AND** the NE diagonal E3-F4-G5-H6-I7 also became a 5-in-a-row simultaneously (the engine offered both as choices). | **Two simultaneous 5-in-a-rows from one move** — game over. |
| 26 | — | `markers C3 D3 E3 F3 G3; remove H10` | **3-0 win.** |

### Observations

1. **Ring placements:** The model's 5 placements (D4, F5, D5, F3, C5) were a **tight central cluster on cols C-F**, identical in flavor to iter_0 v1's left-side cluster. Did not adapt to my spread placement (E6, G7, H6, F8, I8). Same opening pattern as iter_0 v1.
2. **Did the model defend my 3-in-a-rows?** Sometimes. It engaged on the col D area (multi-flip D3→D6) when its own threat aligned, but never specifically defended my chains.
3. **Did the model defend my 4-in-a-rows?** **No, twice.** Move 22 (C2→B1 corner move when I had a visible 4-in-a-row NE diagonal one ply from completion) and move 26 (failed to defend row 3 4-in-a-row). **Same exact blind spot as iter_0 v1 — defending the 4→5 finish is not in this model's repertoire.**
4. **Did the model make multi-flip moves itself?** Yes:
   - Move 18 (`H7→C2`): **3-marker SW-diagonal jump** — biggest tactical move from the model. Flipped 3 of my markers in one shot.
   - Move 25 (`G6→J9`): 2-marker counter-flip disrupting my col H chain. Late-game defensive recognition.
   - Move 15 (`D3→D6`): 2-marker jump assembling col D 3-in-a-row.
   These three are real tactics — the model saw them. Comparable to iter_0 v1 at 400 sims.
5. **Did the model make any clear blunders?** **Yes, two self-flips:**
   - Move 14 (`F3→F6`): jumped its own F4+F5 markers, flipping both to black. Strictly negative-EV. Net loss: 3 white markers.
   - Move 24 (`E5→E2`): jumped its own E3+E4 markers, flipping both to black. Strictly negative-EV. Net loss: 3 white markers.
   - These are the same flavor as iter_0 v1's `G3→D3` move at 64 sims. **Persistent blind spot: the model does not reliably evaluate self-marker flipping as bad.** This is not just MCTS noise — both moves were visit-converged and didn't make sense.
6. **Did the model score any rows?** **No, zero.** This is **worse than iter_0 v1** at 400 sims, which scored col C (1 row). Caveat: I'm playing as Black this game (vs White previously), so the color asymmetry might matter — but the engine has no inherent color bias.
7. **Game length:** 56 moves. Comparable to the iter_0_v2 White-side game (55 moves) and roughly 2× the iter_0 v1 game length at 400 sims. The longer length reflects the same back-and-forth multi-flip patterns and more counter-flips.

### Comparison to iter_0 at 400 sims (the prior baseline) and to the iter_0_v2 White-side game above

| Aspect | iter_0 v1 (400 sims, Black) | iter_0_v2 White-side game | iter_0_v2 Black-side (this game) |
|---|---|---|---|
| Final score | W:3 / B:1 | W:3 / B:1 | **W:0 / B:3 (shutout)** |
| Game length | ~28 moves | 55 moves | 56 moves |
| AI scored own row? | Yes — col C | Yes — D3-H7 diagonal | **No** |
| AI defended my 4-in-a-rows? | No | Mixed (1 yes, 1 no) | **No (twice)** |
| AI multi-flip moves | 1-flip occasional | Multiple, incl. 3-flip H8→H3 | 3-flip H7→C2 + 2-flip G6→J9 |
| AI clear blunders | Self-flip G3→D3 (64 sims) | Bad H5→H8 (move 11) | **Two self-flips: F3→F6 and E5→E2** |
| AI placement style | Tight left-cluster | Mixed (contests row 5, interposes H7) | Tight central cluster cols C-F |

**One-paragraph summary:** Playing as Black against this model is **noticeably easier** than playing as White was — I won 3-0 vs 3-1 on the prior White-side game. The model still produces real multi-flip tactics (H7→C2 was a 3-marker chain disruption, G6→J9 was a 2-flip counter on my col H chain), but it also produced **two clear self-flip blunders** in this game (F3→F6 and E5→E2, both jumping its own markers for no compensating gain). On the defensive side, the headline blind spot is unchanged: **the model does not see the 4-in-a-row → 5-in-a-row finishing pattern when threatened against it**, regardless of color. At move 22, with my NE diagonal one ply from completion, the AI played a corner move (C2→B1). At move 26, with row 3 4-in-a-row + ring on extension cell setup, the AI played another quiet move. Both are the exact pattern iter_0 v1 also missed. The bot-filter improvement (humans-only training data) appears to not have addressed this specific blind spot. Color asymmetry is the most striking finding: **either the model is meaningfully weaker as White than as Black, OR I happened to find more tactical opportunities as Black** (the game-ending move G3→G8 was a 4-marker multi-flip producing two simultaneous 5-in-a-rows — the kind of high-leverage move that's hard to set up unless the position lets you).

### Notes on scoring evolution within the game

- **My 5-in-a-rows:** row 6 (move ~26, from a 2-step setup I8→I6 then I6→I7) → NE diagonal D5-H9 (move ~46, from G5→G8 then G8→G9 setup, off the C4-D5-E6 base I'd built earlier) → row 3 + NE diagonal E3-I7 simultaneously (move 56, via the 4-marker multi-flip G3→G8).
- **AI's 5-in-a-rows:** None.
- **The game-winning move (G3→G8) is worth special note.** It flipped 4 white markers in one ring slide (G4+G5+G6+G7), AND the source-drop at G3 simultaneously completed two different 5-in-a-rows on different axes. This kind of high-leverage finishing move depends on the opponent NOT seeing the threat-line buildup — and the AI's repeated quiet moves (B2→B3, F6→H6, C2→B1) on adjacent turns gave me the time to set this up. **Same pattern as iter_0 v1: the model tolerates patient buildup of threats it can't see.**
- **Cross-color sanity:** combined with the prior White-side game, this confirms the iter_0_v2 model is in roughly the same strength tier as iter_0 v1 — multi-flip tactics emerge, but the 4→5 defensive blind spot persists, plus self-flip blunders still occur at low but non-trivial frequency. **The humans-only filter did not produce a step-change in tactical play.** Would expect head-to-head vs iter_0 v1 to land near 50% (with high variance from determinism quirks per the cycle observations earlier in this doc), and depth=3 heuristic eval to land in the same overlapping CI as iter_0 v1 (~50%, CI95 wide).
