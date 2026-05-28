# Opening-theory experiment design

**Date:** 2026-05-28
**Status:** designed, not yet run
**Trigger:** friend-tester feedback (2026-05-28) — engine clusters 4 of 5 white
rings around D5/E5/F5 in its first game, described as "unheard of strat, gets
the first ring quick, hard to defend."
**Sibling docs:** `analysis_board/opening_map/DESIGN.md` (the model-strength
gradient sweep — bigger, slower); this doc is the *cheap* training-config
experiment, targeted at exploration during placement.

---

## 1. Diagnosis

### 1.1 What the expert datasets say about opening shape

Two corpora, deliberately split by player quality:

- **Boardspace** (mixed tier): n = 3,886 validated games
  (`expert_games/validated/all_games.json`).
- **BGA top-10** (high tier): n = 781 games from the daily top-10
  leaderboard scrape (`expert_games/bga/parsed/`).

**First white placement** — F6 dominates in both, but harder at the top:

| Position | Boardspace | BGA top-10 |
|---|---|---|
| F6  | 16.8% | **31.4%** |
| Next 3 | G2/E5/E6 ~2.9% each (center cluster) | J11/G2/E1 ~5% each (perimeter) |

Strong players hit F6 first ~2× more often than mixed-tier players — and
when they don't, they swing to walls/corners (J11, E1, K10, G11, G2), not
to E5/F5/F7. The "second mode" is bimodally separated from F6, not
adjacent to it.

> **Caveat:** F6 is the geometric center of the board. "Place first piece in
> the middle" is the default for anyone with no domain knowledge — it shows
> up in chess, Go, hex, Othello, basically any board game with a visual
> center. So a high F6 first-placement rate isn't strong evidence of expert
> knowledge; it's the prior any naive player or naive policy network
> converges to. The model's heuristic teacher (`ring_centrality`, weight
> 0.211, the #2 feature) literally rewards center-placement, so the model's
> F6 preference is also overdetermined. The *interesting* question is what
> happens for placements 2-5, given the first one was F6.

### 1.1a F6 is a "rich-get-richer" opening

Splitting the corpora by tier and computing white's win-rate conditional
on whether they opened F6:

| Corpus | White WR overall | WR \| F6 first | WR \| not F6 | Δ |
|---|---|---|---|---|
| BS validated mixed (n=3,876) | 43.0% | 41.9% | 43.3% | **−1.4 pp** |
| BS human-train (n=747) | 59.0% | 59.5% | 58.9% | +0.6 pp |
| BS human-holdout (n=97) | 54.6% | 60.0% | 52.8% | +7.2 pp |
| **BGA top-10 (n=705)** | **50.1%** | **60.2%** | **45.6%** | **+14.6 pp** |

At top-tier (BGA), opening F6 lifts white's win rate by 14.6 percentage
points — a sharp positive. At mixed-tier (Boardspace validated), F6 is
flat or slightly *negative*. This matches the friend's "gets the first
ring quick, hard to defend" framing exactly: F6 is a powerful move *if
you can capitalize on the tempo it gives you*, and that capitalization
is a skill the strong players have and the mid-tier doesn't.

The BS-validated 43% white WR is also a red flag: white normally has
first-mover advantage in YINSH, so under-50% white-WR means that corpus
includes a lot of weak-white (probably bot) play. Treat BS-validated as
"mixed quality" rather than "human baseline." The BS human-train and
BGA splits are both around-50%-white-WR and behave like real games.

### 1.1b Per-slot dynamics: where each tier diverges

| Slot | BGA top-3 | BS human-train top-3 |
|---|---|---|
| 1 (W1) | F6 31%, J11 6%, G2 5% | F6 22%, J10 3%, B2 3% |
| 2 (W2) | K10 6%, F2 5%, G7 5% | J6 3%, G5 3%, F6 3% |
| 3 (W3) | K7 5%, G11 5%, **A5 4%** | G5 3%, G7 3%, E5 3% |
| 4 (W4) | J5 4%, E1 4%, E5 3% | D4 3%, E7 3%, H7 3% |
| 5 (W5) | G2 4%, G11 3%, J6 3% | D5 3%, E4 2%, G2 2% |

The top-tier pattern is **center-anchor + perimeter-expansion**: F6
first, then walls and corners (K10, K7, G11, J5, J11, A5, E1).
Mid-tier instead **drifts further into the center** for placements 2-5
(G5, G7, E5, G6, D4, D5). This is precisely the kind of "in-the-zone
clustering" the friend described in the model's play — but it's the
*mid-tier* pattern, not the top-tier pattern. That's a meaningful
correction to H4 in §2: even if BGA tight-cluster rate is 12%, the
top-tier *modal* opening is diffuse (center-anchor + perimeter), so
the model's cluster is closer to mid-tier than top-tier.

**Aggregate across all 5 white placements** — note the per-position rates
are bounded by the fact that F6 can only be placed once per game, so the
aggregate is structurally capped near uniform (5/85 ≈ 5.9%):

| Position | Boardspace agg | BGA agg |
|---|---|---|
| F6 | 5.2% | 7.6% |
| Next 5 | F5/G6/E5/G5/H7 (center ring) | G11/E1/J11/G2/K10 (perimeter ring) |

### 1.2 How tightly do players cluster their 5 rings?

Spread metrics over the 5 white rings (axial-grid Euclidean — fine for
ranking):

| Metric (white-side, all 5 placements) | Boardspace median | BGA median |
|---|---|---|
| mean_pairwise distance | 4.18 | 4.52 |
| max_pairwise distance | 7.07 | 7.81 |
| max dist to centroid | 4.02 | 4.40 |
| **Tight cluster rate (max_dc ≤ 2.0)** | **5.8%** | **11.9%** |
| Mean centroid x,y | (5.08, 4.84) | (5.36, 5.17) |

Two updates from the BGA data vs my first cut of this doc:

1. **Top players cluster more, not less.** Tight-cluster (4-of-5 within
   max_dc ≤ 2) rate is ~12% at BGA top-10, ~6% at Boardspace mixed-tier.
   The friend's description of the model's opening lands well within the
   ~12% band of BGA top-10 play. That doesn't make the model's opening
   *good* — top players still play diffuse ~88% of the time — but it does
   mean the pattern isn't "unheard of," it's "minority strategy among
   strong players."

2. **The friend (who beats Jack but isn't a tournament-level YINSH player)
   is calibrated against a Boardspace-ish opening distribution.** That's
   probably why "unheard of strat" — in the mixed-tier corpus the friend
   has experienced, this opening shape really is ~6% rare. In the
   top-tier corpus it's ~12% — twice as common but still minority.

I have NOT directly measured the model's actual placement distribution yet
— `analysis_board/multiplayer/deploy/games/` is empty in the local
checkout, so the only model-opening evidence is the friend's single-game
anecdote. The measurement protocol in §4 fixes this; the diagnosis below
explains *why* placement is structurally under-trained regardless of
whether the friend saw the modal opening or a tail sample.

### 1.3 Structural observations about placement in the current training stack

Three things make the placement phase under-trained in a way that's
consistent with the friend's anecdote:

1. **The heuristic value gives no signal during placement.** All 7 learned
   features (`yinsh_ml/heuristics/features.py`) are differential and
   marker-dependent: completed_runs, potential_runs, connected_marker_chains,
   board_control return 0 with no markers. Only `ring_positioning` and
   `ring_spread` produce any nonzero value. In `evaluation_mode: hybrid`
   with `heuristic_weight_start: 0.5`, half of the MCTS leaf value during
   placement is essentially zero plus a touch of ring-geometry. The value
   anchor that stabilizes main-game play is absent for placement.

2. **No placement-specific exploration knob.** The Dirichlet-noise and
   temperature schedules are global, not phase-aware:
   - `dirichlet_alpha: 0.3`, `epsilon_mix_start: 0.25 → end 0.0` linearly
     over the first 20 plies. At the end of placement (ply 10), eps ≈ 0.14.
   - `initial_temp 1.0 → final_temp 0.1` with
     `annealing_steps * temp_clamp_fraction = 30 * 0.6 = 18`. So at the
     end of placement (ply 9, white's 5th ring), temperature ≈ 0.55 — still
     soft, but visit counts already dominate.
   Noise + temperature are reasonable for ply 1. By ply 9 they've decayed
   enough that the network's policy prior dominates.

3. **The placement phase is weighted 1.0 in training** (vs `MAIN_GAME: 2.0`)
   under `phase_weights`. Combined with the fact that placement is only
   ~15% of game positions, placement positions see ~7-8% of the effective
   training mass. The value head has even less signal to learn from than
   the position counts suggest, because placement value targets are the
   game's terminal outcome propagated back through ~60 plies of
   credit-assignment.

4. **Self-play is its own teacher during placement.** Because the value
   head can't directly distinguish placement positions, the placement
   policy is shaped almost entirely by terminal outcomes. If the network
   stumbles into a placement pattern that happens to win against itself,
   self-play reinforces it — there's no expert correction in the loop
   unless the supervised warm-start is doing it, and the warm-start data
   shape for placement specifically has not been audited.

I have NOT directly measured the model's actual placement distribution
yet — the `analysis_board/multiplayer/deploy/games/` directory is empty
in the local checkout, so the only model-opening evidence is the friend's
single-game anecdote. The diagnosis in 1.3 explains *why* a placement
anomaly is structurally plausible; section 4 proposes how to measure it.

---

## 2. Hypotheses, ranked

### H1 (highest confidence): The value head can't ground placement decisions, so the policy head drifts to a self-play attractor.

**Mechanism:** the value head gets no per-position discriminative signal
during placement — no markers means the heuristic value is ~0, and the NN
value head only learns from terminal-outcome credit assignment across ~60
plies. In hybrid mode this means *both* the heuristic anchor and the value
head are weak during placement, leaving the policy head almost entirely
self-supervised through self-play. Once a placement pattern gets played by
both sides, it never gets locally punished, so the policy head's prior on
that pattern strengthens iteration over iteration.

**Why I rate this highest:** it's the only hypothesis that explains *both*
the cluster-tightness anomaly *and* the structural property that placement
positions get fewer effective training gradient updates than their
position count suggests.

**Testability:** add an auxiliary loss term that distills strong-search
placement values back into the value head, and check whether the placement
policy distribution decorrelates from the previous iteration's. See E2.

### H2: The Dirichlet/temperature schedule under-explores placement.

**Mechanism:** by ply 9 the exploration knobs are mostly decayed
(`eps≈0.14`, `temp≈0.55`). The Dirichlet noise + softened temperature do
give the first 2-3 placements diverse coverage, but the last 2-3 are
already mostly greedy. If the network's prior is wrong for those late
placements, it stays wrong because there's not enough noise to discover
the alternative.

**Why I rate this medium:** this would explain the "4-of-5 cluster with
one outlier" pattern — the outlier is the early, well-explored
placement; the cluster is the late, near-greedy placements collapsing
onto whatever the policy prefers. But noise alone shouldn't drive a
*pattern* that wasn't already in the policy — it can only fail to break
one that's there.

**Testability:** boost noise specifically for plies 1-10. If the model's
placement distribution doesn't broaden, H2 wasn't dominant. See E1.

### H3: Self-play echo chamber.

**Mechanism:** because the model plays the same prior on both sides, a
suboptimal opening never gets punished — both sides play it, both sides
"win" half the time, gradient ≈ zero.

**Why I rate this lower:** it's a real mechanism but it's the *generic*
case for any self-play training. It only explains the placement anomaly
specifically if H1 is also true (no value-head signal to correct it).
So H3 is essentially "if H1, then this attractor is also self-reinforcing"
— hard to test in isolation.

**Testability:** mix expert placement positions into the training batch.
If model placements drift toward expert distribution, H3 was a meaningful
contributor on top of H1. See E3.

### H4 (moved up from "rejected"): The model has converged on a minority but real top-tier opening pattern.

**Mechanism:** ~12% of BGA top-10 games show the same tight-cluster
signature the friend described. If the model's training has pulled it
toward this pattern (via supervised data, self-play attractors, or both),
that's an *interesting* finding, not a failure mode. The friend's "unheard
of" reaction is calibrated against Boardspace-mixed-tier opening
distribution where the pattern is ~6%; against BGA top-10 it's twice as
common.

**Why I moved this up:** my first cut rejected this hypothesis based on
the Boardspace-only finding (bottom 6%). Including BGA shifted it to
bottom 12%, which is in "minority but legitimate" territory, not
"anomaly" territory.

**Why I still rate it below H1:** even if the model has converged on a
*real* opening pattern, H1 explains *why* the model would converge there
even if it weren't optimal. The two are not exclusive — H1 is the
mechanism, H4 is the possibility that the mechanism happened to find
something good. We can't distinguish them with one model and one anecdote.

**Testability:** the `opening_map/DESIGN.md` sweep (opening × model
strength × n games) is the rigorous answer. If the model's cluster
opening wins against weaker models *and* against stronger models, it's
legitimate; if it only wins against models in its own training lineage,
it's an echo-chamber attractor. Out of scope for this experiment, but
the sweep is queued.

---

## 3. Three experiments to run

Each is independent — they can be run as separate config diffs against
the post-pretrain warm-start anchor (currently
`models/yngine_volume_15ch_pretrain_b1b2b3_rerun2_iter1/iter1_ema.pt`).
Order in the table is by cost-to-information ratio.

### E1 — Phase-aware exploration boost during placement

**Hypothesis:** H2 (under-exploration during placement).

**Config diff (proposed knobs, not yet in schema):**

```yaml
self_play:
  # New: per-phase exploration knobs that override the global schedule
  # while phase == RING_PLACEMENT (plies 0-9).
  placement_dirichlet_alpha: 1.0        # was 0.3 — flatter Dirichlet,
                                        # more uniform exploration
  placement_epsilon_mix: 0.5            # was tapering 0.25 → 0.14 over
                                        # plies 0-9; lock at 0.5 instead
  placement_temperature: 1.0            # was tapering 1.0 → 0.55 over
                                        # plies 0-9; lock at 1.0 (sample
                                        # proportionally from visits)
```

**Wiring sketch:** in `yinsh_ml/training/self_play.py::MCTS`, gate
`_compute_epsilon_mix`, `get_temperature`, and the Dirichlet alpha used
in `_apply_root_dirichlet_noise` on `game_state.phase`. Falls through
to the existing global schedule when phase != RING_PLACEMENT.

**Rationale:** explicitly tests whether the placement cluster is a noise-
budget issue. Cheapest experiment because no training-loss-shape changes
— only the data-generation side moves.

**Metric to move:** model first-5-placement entropy (Shannon, computed
over the marginal position distribution) compared against the warm-start
anchor. Also: cluster-tightness metric from section 1.2 — expect
max_dist_to_centroid to rise from the friend-anecdote ~2.0 range to the
human median of ~4.0.

**Cost:** ~12-15h on the cloud box (same as B1B2B3).

### E2 — Auxiliary placement-value loss with search distillation

**Hypothesis:** H1 (value head can't ground placement).

**Config diff:**

```yaml
trainer:
  search_consistency:
    enabled: true                       # was false
    policy_weight: 0.0                  # disable policy distillation; we
                                        # want to test value, not policy
    value_weight: 1.0
    every_k_steps: 5                    # was 10; double the cadence
    long_sims: 128                      # was 64; deeper for placement
    warmup_iters: 0                     # was 3; start immediately
    # New knob — only distill placement positions, not all batches
    placement_only: true                # filter sampled positions to
                                        # phase == RING_PLACEMENT
```

**Wiring sketch:** the search-consistency probe already exists; add a
`placement_only` filter to the position sampler in the trainer's
consistency-batch builder. When true, sample only from positions where
`phase == RING_PLACEMENT`.

**Rationale:** if H1 is right, this should sharpen the value head's
placement predictions and let normal self-play gradients propagate a
meaningful signal back to the policy head.

**Metric to move:** placement-position value-loss should drop sharply
in the first 2-3 iterations (the head is filling in a previously-empty
slot). Headline metric is still placement entropy + cluster-tightness
from section 1.2.

**Cost:** ~+30-50% wall-clock per iter (per the existing
`search_consistency` comment), so ~18-22h for a full run.

### E3 — Expert-data mix-in for placement positions only

**Hypothesis:** H3 (echo chamber) — but only meaningfully effective if H1
is also true.

**Config diff:**

```yaml
trainer:
  # New: mix expert placement positions into each training batch.
  expert_mixin:
    enabled: true
    fraction: 0.10                      # 10% of each batch
    source: expert_games/validated/training_data.npz
    phase_filter: RING_PLACEMENT        # only placement positions —
                                        # don't disturb main-game training
    value_target: outcome               # use game outcome as value target
                                        # (cheapest; could also try MCTS
                                        # targets if we re-search expert
                                        # positions)
```

**Wiring sketch:** the boardspace pipeline already produces
`training_data.npz` (8.5 MB). Need a batch builder that mixes expert
samples into the existing self-play buffer at the configured fraction,
filtered to placement. Could be a 50-line addition to the trainer's
dataloader.

**Rationale:** the most direct fix if the cluster is a local minimum —
top-tier BGA play has F6 + perimeter (J11/E1/K10/G11/G2) as the
strongest-five aggregate while still placing diffusely 88% of the
time. Expert gradient pulls the policy away from the cluster attractor.

**Subquestion: which corpus to mix?** Boardspace (3,886 games, mixed
tier) is bigger; BGA (781 games, top-10 tier) is higher quality but the
sample is small. First attempt: BGA-only, accepting the smaller sample
in exchange for cleaner signal. If we want more data, the Boardspace
holdout split (`expert_games/boardspace_human_train/`) is already
filtered for human play.

**Metric to move:** model first-placement distribution should shift
toward the expert F6 mode (model presumably already has F6 mass; expect
the *tail* to broaden — top-1 agreement rate goes up *and* entropy goes
up). Cluster-tightness from section 1.2 should rise toward human median.

**Caveat:** if the warm-start already trained on this expert data, this
experiment is just "more of the same gradient." The 3-axis sweep in
`opening_map/DESIGN.md` is the better way to validate that mix-in is
producing genuine improvement and not just memorization.

**Cost:** ~12-15h plus the one-time engineering for the mix-in
dataloader (~2 hours).

### E4 — Tabula-rasa run: zero heuristic anywhere

**Hypothesis:** the heuristic is acting as a *human prior* injected at
three points (supervised warm-start, hybrid MCTS leaf eval, weighted
training samples). All seven heuristic features are pulled from human
games and all are attack-oriented. `ring_centrality` in particular is
literally a learned reward for placing rings at F6/E5/G6/D6/G5 — i.e.,
the same cluster the friend identified as the model's opening.
Self-play under this prior may be incapable of finding non-human
strategies even if they exist, because the gradient pressure to
override a three-place injection is large.

**Config diff (the maximalist version):**

```yaml
# NO supervised warm-start. Start training from random weights.
warm_start:
  enabled: false                        # was: load pretrain checkpoint

self_play:
  evaluation_mode: pure_neural          # was: hybrid
  heuristic_weight: 0.0                 # was: 0.5 → 0.0 anneal
  heuristic_weight_start: 0.0
  heuristic_weight_end: 0.0
  # Boost exploration to compensate for the missing anchor
  dirichlet_alpha: 0.5                  # was: 0.3
  epsilon_mix_start: 0.4                # was: 0.25
  epsilon_mix_taper_moves: 30           # was: 20

trainer:
  # No phase-weighted sampling either — the original AlphaZero recipe
  # weights all positions equally.
  phase_weights:
    RING_PLACEMENT: 1.0
    MAIN_GAME: 1.0
    RING_REMOVAL: 1.0
```

**Rationale:** if this run converges on a meaningfully different opening
distribution *and* is at competitive strength (≥ Wilson LB 0.50 vs
the warm-start anchor), the heuristic was hiding strategies. If it
converges on roughly the same opening — different in the tails but
similar at the mode — the model's opening prior was probably an
honest reflection of the game, not a teacher artifact.

**Caveat: even E4 isn't pure AlphaZero.** The MCTS itself, the
architecture, the encoding, and the training loop all came from the
existing repo, which was built with human-game intuition. Real
tabula-rasa would require starting from a freshly-initialized network
and probably 5-10x the iteration count. E4 is the closest practical
approximation.

**Metric to move:** opening *mode* should diverge from the anchor's
mode by a measurable amount (e.g., top-1 first-placement disagrees with
anchor in ≥ 30% of games). Strength must not regress (Wilson LB ≥ 0.50
vs anchor, 400 games SPRT).

**Cost:** ~50-100h on the cloud box for 30-50 iterations. The first 10
iterations will play very weakly (no warm-start) so most of the cost
is in the late-iteration convergence. Big swing. Worth doing once,
even if the answer is "the heuristic was right all along."

### E5 — Add a perimeter-occupation feature and let it compete

**Hypothesis:** the heuristic's center bias is concentrated in *one*
feature (`ring_centrality`, weight 0.211) which is mathematically
incapable of rewarding non-center play. Adding a competing feature
that rewards *perimeter* placement, with the weight learned from the
data instead of fixed, lets the model trade off center vs perimeter
based on what actually correlates with winning, not what the original
heuristic-design pass assumed.

**Config diff:**

```yaml
# In yinsh_ml/heuristics/features.py, add:
#   def perimeter_occupation(state, player) -> float:
#     return (#my_rings_in_edge_band) - (#opp_rings_in_edge_band)
#
# Then re-run the weight-learning analysis on the 100K-games corpus:
#   python run_complete_analysis.py
# to get the data-derived weight for this feature (and updated weights
# for the rest, since they shift slightly when a new feature is added).
```

**Rationale:** if `perimeter_occupation` gets a positive learned weight
from the corpus analysis, the original heuristic was missing it. If it
gets a near-zero weight, perimeter strategy genuinely isn't a
winning-correlated pattern in human play. Either result is informative.

**Note:** this experiment tests whether the *feature set* is biased,
not whether the *training pipeline* is biased. E4 tests the pipeline.
E5 is cheaper but less ambitious — it adds one feature and lets the
data speak. Should be run first; if it surfaces a real perimeter
signal, E4 becomes lower-priority.

**Metric to move:** learned weight for `perimeter_occupation` on the
100K-games dataset. > +0.1 = real signal worth retraining on. < +0.05
= heuristic feature set was probably complete enough on the placement
question.

**Cost:** ~2-3 hours engineering + ~30 min analysis re-run. The
training run after that is a normal-cost iteration (~12-15h).

---

## 4. Measurement plan

A successful experiment moves *both* of these from the warm-start
anchor baseline:

### Metric 1 — Top-1 placement agreement rate with expert distribution

For each white-side placement position 1-5 (i.e., plies 0, 2, 4, 6, 8):

```
agreement_rate(k) = P(model_top1_placement_k matches human_top1_placement_k_given_history)
```

Compute by playing the model against itself for N games (N = 100 is
enough for a coarse signal; N = 500 if we want CI's tight). At each white
placement, record the top-1 model choice. Aggregate vs the expert
marginal distribution.

**Pass:** at least +10 percentage points on aggregate top-1 agreement
across the 5 placement positions vs the anchor.

### Metric 2 — Cluster-tightness against both expert corpora

For each of N self-play games, compute `max_dist_to_centroid` and
`mean_pairwise` from section 1.2. Compare model median to *both*
Boardspace and BGA bands:

| Metric | Boardspace p25-p75 | BGA p25-p75 |
|---|---|---|
| mean_pairwise | ~3.3 – ~5.2 | ~3.5 – ~5.6 |
| max_dist_to_centroid | ~3.0 – ~5.0 | ~3.3 – ~5.4 |

**Pass condition is now conditional:** model median should land in
*either* corpus's band, AND model's tight-cluster rate (max_dc ≤ 2.0)
should drop from the friend-anecdote ~100% (n=1) toward at most ~15%
(matching BGA's tail). The model isn't asked to play like the median —
it's asked to stop concentrating mass in the bottom 12% of human play.

### Metric 3 — Head-to-head against the anchor (guardrail)

A new opening is only worth shipping if main-game strength doesn't
regress. Standard SPRT match vs the warm-start anchor, 400 games, Wilson
LB ≥ 0.50.

**Pass:** new model is at least 50% by Wilson lower bound. This is a
*non-regression* guardrail, not a strength claim — we're explicitly OK
with placement diversity that costs ~0% main-game ELO.

### Decision matrix

| M1 (top-1 agreement) | M2 (cluster tightness) | M3 (head-to-head) | Verdict |
|---|---|---|---|
| ≥ +10 pp | in human band | ≥ 50% Wilson LB | ✅ ship |
| ≥ +10 pp | in human band | < 50% Wilson LB | 🤔 opening moved but main-game regressed; revisit |
| < +10 pp | in human band | ≥ 50% Wilson LB | ⚠️ opening shape changed but didn't match expert pattern — investigate |
| < +10 pp | not in band | any | ❌ exploration knobs / aux loss didn't change anything; null result |

---

## 5. Data sources and measurement queue

| Source | n games | Status | What it tells us |
|---|---|---|---|
| Boardspace validated mixed | 3,886 | analyzed | First-cut human baseline, but 43% white-WR ⇒ contaminated by weak bots. Treat as ceiling estimate for noise, not as expert baseline. |
| Boardspace human-train | 761 | analyzed | Filtered to human-only play. 59% white-WR consistent with first-mover advantage. Reliable mid-tier baseline. |
| Boardspace human-holdout | 100 | analyzed | Same population as above, held out. Small sample but consistent with human-train. |
| Boardspace holdout 500 | 500 | excluded | Anomalous (F6-first WR 27%, slot-5 top is I8). Probably a low-rated or bot-heavy subset; skipped. |
| BGA top-10 leaderboard | 781 | analyzed | High-tier baseline. 50% white-WR, F6 lifts WR +15 pp ⇒ skill-dependent strong move. The reference standard. |
| CodinGame validated | 0 | empty | Scraper exists (`yinsh_ml/data/scrapers/codingame.py`) but no validated games. Not useful for opening analysis. |
| **Model self-play, deployed_sampled (iter1_ema)** | 200 | **in flight** | What the network believes given temp 0.5 sampling. PID 50065, ETA ~3.3h, output `analysis_board/multiplayer/model_openings_iter1_ema_deployed.json`. |
| **Heuristic-baseline (HA-vs-HA, depth 2-4)** | 200 | **in flight** | What the *pure heuristic teacher* plays as opening, no NN involved. Tests whether the model is just reproducing the heuristic's preferences. PID 53905, output `analysis_board/multiplayer/data_runs/heuristic_baseline/`. CPU-only. |
| Model greedy-mode (iter1_ema, temp 0.1) | 50 | queued | The *mode* of the network's belief, not the support. Tells us if the friend's experience is the modal opening or a tail sample. Queued for after the deployed_sampled run frees MPS. |
| Supervised baseline (supervised_2026-05-27, deployed_sampled) | 100 | queued | What the model played *before* any self-play could shift it. Diff vs iter1_ema isolates the self-play contribution to opening shape. Queued for after greedy-mode. |

### What each comparison answers

- **Model deployed_sampled vs BGA top-10:** does the model's *distribution* match top-tier human shape?
- **Model greedy vs Model deployed_sampled:** is the friend's experience the mode or a tail?
- **Model vs Heuristic-baseline:** is the model just reproducing what the heuristic teaches, or has it learned something the heuristic alone doesn't produce?
- **iter1_ema vs Supervised baseline:** how much did self-play shift the opening distribution from the warm-start?
- **Model F6-first WR vs BGA F6-first WR:** can the model *play* the F6 opening competitively? (Looks at main-game performance from F6 starts.)

If all four diffs come back small (model ≈ heuristic ≈ supervised ≈ BGA), the model is faithfully reproducing what was taught — no novel discovery, no anomaly to chase. If the model diverges from the heuristic baseline, it learned something. If it diverges from BGA, the something is non-standard. If main-game performance from F6 starts is worse than BGA's, the model has the opening but not the followup.

---

## What I'm explicitly not concluding

- **Whether the model's opening is bad.** I have one anecdote (friend's
  first-game observation) and structural arguments. The
  `opening_map/DESIGN.md` sweep is the rigorous answer to "is opening X
  actually worse than opening Y under good play."
- **Whether E1, E2, or E3 will work.** Each tests a hypothesis with
  different cost. E1 is the cheapest and most surgical — try it first.
- **Whether the friend is right that this is unusual.** Against
  Boardspace mixed-tier the cluster shape is in the bottom ~6%, but
  against BGA top-10 it's in the bottom ~12%. So it's "unusual in
  amateur play, minority strategy in expert play." The friend is right
  *and* the model isn't necessarily wrong.

---

## Open questions for Jack

1. Do you want to wire the `placement_dirichlet_alpha` /
   `placement_epsilon_mix` / `placement_temperature` knobs into the
   schema at the same time, or as separate work? E1 needs them.
2. The placement-only filter in `search_consistency` (E2) is a small
   change but it touches the trainer. Worth doing in the same PR as E1's
   schema work, or kept apart?
3. The expert-data mix-in (E3) is conceptually simple but introduces a
   training-loop dependency on the boardspace pipeline. Acceptable? Or
   should we defer E3 until E1/E2 land null?
4. Are you OK running E1 against `iter1_ema.pt` even though the post-
   B1B2B3 work landed NOT_STRONGER? It's still the strongest checkpoint
   we have, but you might want to start from a fresh-er base if A4+D1
   ships first.
