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
