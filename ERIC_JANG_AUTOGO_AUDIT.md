# Eric Jang AutoGo audit — YinshML architectural findings

**Audit date:** May 2026
**Branch:** `claude/game-replay-viewer`
**Audited against:** Eric Jang's reproduction-of-AlphaGo project, his
discussion of it on the Dwarkesh Patel podcast, and KataGo's
modernization-era techniques referenced therein.

## Source material

The audit triangulates three sources:

1. **Dwarkesh Patel × Eric Jang interview (May 2026).** Eric walks
   through MCTS first principles, KataGo's compute-multiplier
   optimizations, and the empirical lessons he's drawn from rebuilding
   AlphaGo himself. The strongest practical claim — *"what took a team
   of research scientists at DeepMind and millions of dollars of
   research and compute can now be done for a few thousand dollars of
   rented compute"* — directly motivates whether YinshML's recipe has
   caught up to that envelope and what gaps remain.
2. **github.com/ericjang/AutoGo.** Eric's working reproduction. Three
   network variants (`GoTransformer`, `MuPGoResNet`,
   `SizeInvariantGoResNet`), distributed CPU/GPU training infra, and
   an explicit research-automation goal (the agent is co-equal to the
   game-mastery objective). Verified architecture details by reading
   `src/alpha_go/model.py` directly.
3. **github.com/temhelk/yngine.** C++ MCTS engine with 128-bit
   bitboards and parallel lock-free UCT, specifically for YINSH. Cited
   in the project as the inspiration for the bitboard port roadmap
   (`BITBOARD_PORT_PROMPT.md`, `BITBOARD_FOLLOWUP_PLAN.md`). Used in
   YINSIM (the static-HTML viewer that motivated the dashboard work in
   this branch).

The audit looked at YinshML's *as-implemented* state on `main`, not
its documented intent. The 7-step verification pattern was: read each
claim from Eric → search the YinshML codebase for the
corresponding behavior → smoke-test it → either tick it as
"implemented" or flag the gap with concrete file:line evidence.

## TL;DR

YinshML is much more mature than a first read of the project's older
planning docs (`ARCHITECTURAL_IMPROVEMENTS_PLAN.md`, Feb 2026) would
suggest — many of the techniques Eric covers are already implemented
thoughtfully. The actual remaining gaps, ordered by leverage:

1. **No global feature pooling in the network.** SE blocks in the
   trunk + GAP-then-MLP in the value head are the canonical fixes,
   confirmed in Eric's `MuPGoResNet`. YinshML's `SpatialAttention` is
   per-pixel gating — local features only. YINSH's row-capture
   mechanic (local move → global state change) makes this gap matter
   more than it would in Go.
2. **300-move cap value labels are partial-credit**
   (`yinsh_ml/training/self_play.py:1751-1754`):
   `value = clip(score_diff/3, -1, 1)`. Games that time out before a
   real winner contribute non-terminal partial-credit labels to the
   value head. Needs measurement before deciding on a fix.
3. **Supervised pretraining uses one-hot policy targets**
   (`yinsh_ml/data/converter.py`, `scripts/run_supervised_pretraining.py:162`).
   Eric's bits-per-sample observation applies: a distribution carries
   far more signal than the argmax. Mitigable if the heuristic's move
   scores can be surfaced as a soft distribution.
4. **Heuristic curriculum sometimes never fully anneals off** (e.g.
   `configs/cloud_run_v1.yaml`: 25 iters, anneal over 25, so
   `heuristic_weight` stays > 0 for the entire run). Bitter-Lesson
   check: scaffolding becomes ceiling, ceiling is harmful.

Each is well-scoped — none requires a from-scratch rewrite.
Actionable items mirrored into `TODO_baseline.md`. Durable insights
mirrored into `RESEARCH_LOG.md`.

A fifth, lower-priority gap is in the next-tier audit metrics for
defense quality (the basic 4-row threat detector landed but a richer
multi-dimensional analysis is queued; see "Open research questions"
below).

## Walkthrough of the source claims and how YinshML maps to them

### Initialization is everything

> *"In deep learning, initialization is everything. You always want to
> initialize your experiments to something as close to success as
> possible."* — Eric, on AlphaGo Lee's supervised init from human
> expert games before AlphaGo Zero went tabula rasa.

**YinshML status:** ✅ Implemented at a sophisticated level.
`scripts/run_supervised_pretraining.py` trains from a scraped corpus
(BGA + Boardspace + Codingame + Little Golem). `run_training.py
--init-checkpoint` warm-starts the self-play loop without inheriting
optimizer state. `configs/phase_d_warmstart.yaml` documents the
specific recipe: heuristic curriculum starts at 1.0 (full heuristic
anchoring) and anneals to 0.0 over 10 iters so the supervised value
head doesn't get washed out on iter 1.

The empirical result from `WARMSTART_PHASE_LOG.md`: the warm-start
`iter_0` checkpoint dominates every subsequent self-play iteration
40-0 in head-to-head matches. The supervised init is the strongest
model in the project's history; the open question is making the
self-play loop *productive* on top of it.

### Value-first training

> *"It doesn't really make a lot of sense to do search on garbage
> value predictions."*

**YinshML status:** ✅ The warm-start curriculum encodes this — value
head trains on real expert outcomes during supervised pretraining,
then on score-delta-from-final values during self-play. The 7-class
discretized outcome with Gaussian smoothing (`trainer.py:995-1008`)
is actually *richer* than the binary CE Eric's AutoGo uses.

The one wrinkle: the 300-move cap (see Gap 2 below) contaminates the
value training signal with partial-credit non-terminations.

### Soft MCTS targets — bits per sample

> *"In AlphaGo, you don't train the policy network to imitate the
> MCTS action. You train it to imitate the MCTS distribution. Both of
> these are valid, but the distribution has way more information in
> bits per sample."*

**YinshML status:** ✅ for self-play (`trainer.py:948` uses
`-(target_probs * log_probs).sum(dim=1).mean()` over the full visit
distribution). ❌ for supervised pretraining (see Gap 3) — one-hot at
the expert move.

### MCTS as a policy-improvement operator

> *"Importantly, what MCTS is doing is saying: for every action we
> took, we did a pretty exhaustive search to see if we could do
> better, and we're going to make every action that we took better by
> having the policy network predict that outcome instead."*

**YinshML status:** ✅ This is how `yinsh_ml/training/self_play.py`
self-play data flows into the trainer. Soft visit distributions →
trainer → policy head distills toward "MCTS at this state would have
done this."

The deeper insight from the transcript — *MCTS doesn't try to do
credit assignment; it relabels every action with a strictly better
one* — explains why AlphaZero RL has dramatically lower variance than
LLM-style REINFORCE despite being "RL." Worth keeping in mind when
comparing YinshML's training characteristics to what you'd expect
from policy gradient methods.

### Architecture: ResNets, attention, global pooling

> *"For small data regimes, my experience is that ResNets still
> outperform transformers and give you more bang for the buck at lower
> budgets. ... One interesting finding from the KataGo paper was that
> they found it quite useful to pool together and aggregate global
> features throughout the network."*

**YinshML status:** ✅ for ResNet choice. ❌ for global pooling. This
is the largest remaining architectural gap. See **Gap 1** below.

Eric's `MuPGoResNet` confirms the recipe in code:
- **Squeeze-and-Excitation blocks** in every residual block (per his
  README description: *"Squeeze-and-Excitation block for channel
  recalibration"*)
- **Global average pooling in the value head** (*"1×1 conv to value
  channels, global average pooling, then dense layers with MuReadout"*)

YinshML has spatial attention (per-pixel gating via 1x1 conv +
sigmoid) but no global aggregation. The value head specifically uses
`Conv → Flatten(64·11·11=7744) → Linear(7744, 512) → ...` — ~4M
params in a single linear layer that has no spatial-translation
equivariance, on a head that's asking a fundamentally global question
("who's winning overall").

### Initialization off small boards / cheap data

> *"What you can try to do is play random games on a small board.
> Just take a random agent. If you play 50,000 games, you'll actually
> learn a pretty good value function as well."*

**YinshML status:** N/A (YINSH doesn't scale board size cleanly the
way Go does — the column counts [4,7,8,9,10,9,10,9,8,7,4] don't
trivially shrink). The analogue is the heuristic-self-play data
generator we built this session
(`scripts/generate_heuristic_games.py`): cheap CPU-only games that
ground the value head before the expensive neural self-play kicks in.

This maps directly onto the audit's recommendation: pair a 1,000-1,500
high-quality expert game corpus (for policy head) with 10K-100K
heuristic self-play games (for value head pretraining).

### The 10%-don't-resign rule

> *"In practice, what you could also do is, for 10% of the games,
> prevent the bots from resigning and just say, 'Resolve it to the
> end.' That way you get some training data in your replay buffer to
> really resolve those late-stage playouts."*

**YinshML status:** N/A. No resignation logic anywhere in the
codebase. Games always play to terminal or the 300-move cap. The
*reason* the rule exists in Eric's discussion (avoid undertraining the
value head on endgame positions) maps onto YinshML's analogous concern
about the 300-move cap (Gap 2).

### Best-response training

> *"In a game like StarCraft where you don't have complete control...
> what they do instead is train what's known as a best-response policy.
> You fix your opponent... your goal is just to beat this guy."*

**YinshML status:** Not implemented as a distinct training mode. The
existing `hybrid` MCTS evaluation mode (heuristic blends into leaf
value via `w_h * value_heuristic + (1-w_h) * value_neural`) does
something *related* — uses the heuristic as a teacher — but is not
the same as best-response training. Best-response would generate
trajectories under "play against fixed HeuristicAgent" objective,
which produces complementary data: *how to beat the heuristic*, not
just *the heuristic's value estimate*.

Worth one ablation iteration as a supplementary data source. Tracked
in TODO_baseline.md.

### Off-policy MCTS relabeling

> *"As part of this project, I did try an experiment where I took a
> bunch of trajectories, and to try to saturate the GPU as much as
> possible, I took random states from the dataset and reran MCTS on
> just those states... In practice, this actually does work."*

**YinshML status:** Not implemented; flagged in the transcript as
moderately useful but with the off-policy caveat. Not high priority
given other gaps. The "saturate the GPU" motivation is real but
secondary to fixing the architectural gaps above.

### Soft action masking / encoding ergonomics

Already extensively covered in `ARCHITECTURAL_IMPROVEMENTS_PLAN.md`
(Phase 1, Phase 4). Not re-audited here.

### Game-end value signal

> *"It's quite easy to evaluate a late-stage Go game. When almost all
> the pieces are on the board, it's almost like a decidable problem."*

**YinshML status:** Different — YINSH games end when 3 rows are
captured, not when the board fills. The "almost decidable" property
doesn't apply. The closest YinshML analogue is the RING_REMOVAL phase
(after a row is captured, deterministic). This is captured correctly
by the existing game-state machinery; no gap.

## Maturity audit — checklist of what's already in place

Verified in code during this session:

- [x] **Warm-start pipeline.** `scripts/run_supervised_pretraining.py`
  + `scripts/run_training.py --init-checkpoint`. Phase D config
  documents the recipe.
- [x] **Soft MCTS policy targets in self-play.**
  `trainer.py:948`.
- [x] **Soft value targets** (Gaussian-smoothed CE on 7-class
  discretized outcome). `trainer.py:995-1008`.
- [x] **D2 symmetry augmentation.** Klein 4-group, correctly verified
  against the board's actual symmetries (D6 was wishful thinking, see
  `RESEARCH_LOG.md`).
- [x] **No resignation logic** — games always play to terminal or
  300-move cap.
- [x] **HeuristicAgent as evaluation anchor.** Every iteration runs
  candidate vs HeuristicAgent, both raw-policy and MCTS-mode.
- [x] **Differential heuristic features** — defense is mathematically
  captured at depth ≥3 without adding explicit defensive features.

## Gap detail

### Gap 1: Global feature pooling

**Locus:** `yinsh_ml/network/model.py`.

**Current state:** Network is a 12-block ResNet alternating `ResBlock`
and `AttentionBlock`. `SpatialAttention` (lines 28-46) is per-pixel
gating via 1x1 conv → sigmoid → multiply, not global aggregation.
Value head is `Conv(num_channels, 64) → BN → ReLU → Conv → BN → ReLU
→ Flatten(64·11·11=7744) → Linear(7744, 512) → ... → Linear(256,
num_value_classes)`.

**Eric's `MuPGoResNet` for comparison:**

| Choice | YinshML | `MuPGoResNet` |
|---|---|---|
| Trunk global aggregation | None | SE blocks per ResBlock |
| Value head input | Flatten(7744) | GAP(64) |
| Value head first linear | Linear(7744, 512) — ~4M params | Linear(64, ...) — ~32K params |
| Translation equivariance in value head | No | Yes |

**Three flavors of global pooling, by complexity:**

1. **Squeeze-and-Excitation blocks** in the trunk (Eric's choice).
   Add to each `ResBlock`:
   ```
   y = ResBlock(x)
   g = mean(y, dim=(H,W))                 # [B, C]
   s = sigmoid(FC2(ReLU(FC1(g))))         # [B, C], small MLP, r=8 ratio
   y = y * s.unsqueeze(-1).unsqueeze(-1)  # broadcast channel reweight
   ```
   Param cost: 2·C²/r per block. For C=128, r=8 → ~4K params per
   block, ~50K extra across 12 blocks. Cheap.

2. **Global average pooling in the value head** (Eric's choice).
   Replace `Flatten → Linear(7744, 512)` with `GAP → Linear(64, 512)`.
   Drops ~4M params; makes the value head translation-equivariant;
   value asks "who's winning overall" — a fundamentally global
   question.

3. **KataGo-style global pooling bias** (most involved, not in Eric's
   AutoGo repo as far as I can tell). Split channels into "regular"
   and "global-pool", aggregate global-pool with `concat(mean,
   mean·scale, max)`, project through Linear, broadcast-add as bias
   to every spatial position of the regular channels. Every position
   then has direct access to a learned board-wide summary.

**Why YINSH especially benefits:** the 3-row capture mechanic means
local move → global state change (RING_REMOVAL phase, score update).
A 4-marker row in the bottom-left corner is functionally equivalent to
the same row anywhere else, but the current network's spatial features
have to learn this equivalence from data. SE blocks would provide it
as an architectural inductive bias.

**Validation approach:** A/B against the current architecture in a
short cloud run (24h, ~$5). Metric: anchor win rate against
HeuristicAgent(depth=3) at deployment MCTS budget. Expected lift:
small at this scale (the network isn't currently bottlenecked on
expressiveness so much as on data); 5-10 ELO would be a real win.

### Gap 2: 300-move cap value-label partial-credit

**Locus:** `yinsh_ml/training/self_play.py:1674-1754`.

**Current state:**
```python
max_game_moves = 300
# ... game loop ...
score_diff = state.white_score - state.black_score
outcome_white = float(np.clip(score_diff / 3.0, -1.0, 1.0))
values = [outcome_white if p == Player.WHITE else -outcome_white
          for p in players]
```

A 1-0 game capped at move 300 contributes `value = +0.33` for every
WHITE turn and `-0.33` for every BLACK turn — *as if that were the
authoritative final outcome*. But the game isn't over; one or both
players might still be winning under deeper play.

**Risk:** the value distribution gets compressed toward zero (most
timeouts have small score deltas), and the value head learns
incorrect labels for positions that look winning but are actually
draws-pending.

**Action before any fix:** instrument the existing self-play corpus
to find what % of games hit `max_game_moves`. If <5%, this is
negligible. If ≥15%, address. Currently NO INSTRUMENTATION exists for
this — it's not logged anywhere accessible. The dashboard built in
this session can be pointed at any parquet self-play corpus and the
`replay_truncated_at` field on `GameReplay` flags artificially-capped
games.

**Concrete fix options if measurement says it matters:**

- Treat capped games as draws (`value = 0`) rather than partial
  credit. Loses some signal but stops mislabeling.
- Down-weight capped games in the training loss (`weight = 0.5` or
  similar).
- Raise the cap to 500 and accept the throughput hit.
- Eric's "10%-don't-resign" rule analogue: force a subset of games
  to play to natural completion regardless of length, ensuring the
  value head sees real terminations.

### Gap 3: One-hot supervised policy targets

**Locus:** `yinsh_ml/data/converter.py:26` (the schema definition)
and `scripts/run_supervised_pretraining.py:162` (the loss).

**Current state:**
```python
# converter.py:
#   policy: np.ndarray of shape (total_moves,) — one-hot at the
#   expert move
# run_supervised_pretraining.py:
log_probs = F.log_softmax(pred_logits, dim=1)
policy_loss = -(policies * log_probs).sum(dim=1).mean()
```

With one-hot `policies`, this collapses to CE on the argmax move.
Eric's bits-per-sample point applies: the *distribution* over moves
(e.g. expert was 70% confident in move A, 20% in B, 10% in C) carries
the expert's tactical uncertainty in a way the argmax cannot.

For supervised distillation from a *teacher network* this would be
the natural source of soft targets. For supervised pretraining from
*human expert games*, we don't have move-level uncertainty in the
data — humans played one move, not a distribution.

**Mitigation paths:**

1. **Heuristic-teacher supervised pretraining.** Run `HeuristicAgent`
   at every position in the expert corpus, extract its full move-score
   distribution (currently only the argmax is exposed via
   `select_move`; would need a small API addition), softmax with
   temperature, and use the resulting distribution as a *secondary*
   policy target alongside the expert's one-hot move. Loss becomes
   `α·CE(one_hot) + (1-α)·CE(heuristic_dist)`.

2. **Label smoothing** on the one-hot targets. Cheap; provides a
   small bits-per-sample lift; doesn't capture tactical uncertainty
   but does prevent the loss from collapsing to overconfidence.

3. **Off-policy MCTS relabeling** (Eric's experiment): replay the
   expert games, run MCTS at every position with the current network,
   and use the MCTS visit distribution as a policy target. Higher
   bits-per-sample than (1) or (2); higher compute cost. Probably the
   right thing to do *after* the basic architectural changes have
   landed.

### Gap 4: Heuristic curriculum sunset

**Locus:** `configs/*.yaml` schedules.

**Current state — two contrasting configs:**

```yaml
# configs/cloud_run_v1.yaml
num_iterations: 25
heuristic_weight_start: 0.5
heuristic_weight_end: 0.0
heuristic_weight_anneal_iterations: 25   # never reaches 0 during run
```

```yaml
# configs/phase_d_warmstart.yaml
num_iterations: 12
heuristic_weight_anneal_iterations: 10   # off by iter 11 of 12 — correct
```

The cloud_run_v1 config schedules `heuristic_weight` to anneal over
the *entire* run length. By the math: at iter 25 of 25, weight = 0.0,
but at iter 24 it's still ~0.02 — and iter 24 is the last one with
any training signal effect. So functionally, the heuristic is never
fully off.

**Bitter-Lesson concern:** the heuristic is scaffolding for a young
value head. Once the value head is good enough, the heuristic becomes
a *ceiling*: it constrains what the network can express beyond the
7 hand-engineered features. There's no instrumented signal in the
training loop for "the heuristic has become a ceiling."

**Action items:**

1. **Add a curriculum-sunset signal.** If anchor win rate stops
   improving for K consecutive iterations (K=3-5), force
   `heuristic_weight=0` from then on. Simple plumbing — anchor win
   rate is already tracked.
2. **Default to faster annealing.** Anneal over ~half the run length,
   not the full length. Phase D pattern (anneal over 10 of 12 iters)
   is closer to correct than cloud_run_v1's full-length anneal.
3. **Empirical study.** Run two identical configs differing only in
   `heuristic_weight_anneal_iterations` (10 vs 25 over a 25-iter run),
   compare anchor win rate trajectories. If the faster-anneal config
   beats the slower one in the back half, the bitter-lesson concern
   is confirmed for this scale.

## YINSH-specific findings (durable lessons)

Captured in `RESEARCH_LOG.md` under "Heuristic evaluation (7-feature
set)":

- **All 7 heuristic features are differential** (`my - opponent`).
  Defense is mathematically captured by the existing set as long as
  negamax search runs at depth ≥3. Adding explicit defensive features
  buys nothing for the static-eval ceiling.
- **4-row threats have two distinct mechanisms.** Case (a) gradual
  buildup: 4-row IS visible on the opponent's turn, IS defendable.
  Case (b) path-flip: single ring move flips N markers into a 5-row
  directly, no precursor 4-row state, no defender warning. Audit
  metric `count(length == 4)` catches case (a) only.
- **Action space is 7,433 slots** — bilocational moves are quadratic
  in board cells. Breakdown: 85 placement + 7140 ring-move (85·84) +
  85 ring-removal + 123 marker-removal. Comparable to chess (~4672)
  and shogi (~11K), not Go (362, unilocational).

## Operational outcome — the live game viewer

This audit was paired with a build. The viewer
(`yinsh_ml/viz/board_render.py`, `game_replay.py`, `annotators.py`;
`scripts/dashboard_games.py`, `generate_heuristic_games.py`,
`run_heuristic_audit.sh`) operationalizes several of the audit
findings:

- **Gap 2 instrumentation** — replay any self-play corpus, the
  `replay_truncated_at` field on `GameReplay` flags artificially-capped
  games. Per-game inspection lets you see whether the timeouts are in
  legitimately-stuck positions or whether the cap is firing too early.
- **Gap 4 measurement** — the audit harness generates corpora at
  configurable heuristic-curriculum settings; the trajectory plot
  visualizes whether the heuristic's effect is still useful or has
  flattened to noise.
- **YINSH-specific findings #2** — the 4-row detector + case
  (a)/case (b) classification fired as direct outputs of using the
  viewer during smoke testing. Wouldn't have surfaced without the
  build.

The viewer is documented in `yinsh_ml/viz/README.md`, with a
loader/annotator abstraction that supports future producers (neural
self-play, BGA expert-game review with network annotation, etc.) on
the same infrastructure.

## Open research questions

These came up during the audit and don't yet have an owner:

- **Heuristic-curriculum sunset trigger.** What's the right signal
  for "the heuristic has become a ceiling"? Anchor-win-rate plateau
  for K iters is the obvious candidate but unvalidated.
- **Best-response-to-HeuristicAgent as supplementary data.** Eric's
  AlphaStar framing. Generates trajectories under "play to beat HA"
  objective; cheaper than neural self-play, richer than HA self-play.
  Worth one ablation iteration.
- **Game-end-mode distribution.** Of the games in a self-play corpus,
  what fraction end by 3-row win vs. move-cap timeout vs. stalemate?
  Value-head training signal quality depends on this. Not instrumented.
- **MCTS visit annotator** for the viewer. Re-run MCTS at each
  replayed position; emit visit distribution + root value as a
  per-turn annotation. Plugs into the existing `annotate()` framework
  with no viewer redesign. Expensive but very informative for
  understanding why specific moves were chosen.
- **Soft-supervised vs heuristic-teacher policy targets.** Empirically
  test the bits-per-sample claim on YinshML: same network, same
  hyperparameters, train one on one-hot expert moves and one on
  heuristic-teacher distributions. Compare downstream self-play
  anchor win rates. Probably ~$5 of cloud compute.

## Methodology

What made this audit useful (worth replicating for future
external-lens audits):

1. **Read code, not docs.** The project has 27 root-level planning
   .md files. They describe intent and history; the *current* state
   of the codebase is the only ground truth. Every claim Eric made
   was verified against a specific file:line in YinshML, not against
   what the older docs said the project should be.
2. **Smoke-test claims.** "Soft MCTS targets are used" was verified
   by reading the trainer code AND running a test that confirmed
   non-argmax loss values. "Warm-start pipeline exists" was verified
   by tracing `--init-checkpoint` through `run_training.py` →
   `_load_init_checkpoint` → `NetworkWrapper.load_model`.
3. **Operationalize the audit.** Building the viewer during the audit
   exposed concrete data dynamics (the 4-row case (a)/(b) split,
   captures via path-flip in low-depth play) that would have been
   purely theoretical otherwise. Two of the durable lessons in
   `RESEARCH_LOG.md` came from this.
4. **Be willing to be wrong out loud.** The "markers go 4→5 in a
   single ring-move so threats are transient" claim turned out to be
   misleading on closer inspection. Owning that correction and fixing
   the affected comments / RESEARCH_LOG entries is part of the audit
   value — a stale incorrect note is worse than no note.

## Cross-references

**Actionable items in `TODO_baseline.md`:**

- Global pooling architectural change (Gap 1) — to be appended to
  TODO_baseline.md's Training/research or a new Architecture section
- 300-move cap value-grounding measurement task (Gap 2)
- Heuristic-teacher soft policy targets in supervised pretraining
  (Gap 3)
- Heuristic curriculum sunset trigger (Gap 4)
- Best-response training as supplementary data source

**Durable insights in `RESEARCH_LOG.md`:**

- 7 features are differential → defense at depth ≥3 (existing entry)
- 4-row case (a) vs case (b) capture mechanism (existing entry)

**Operationalized in code on this branch:**

- `yinsh_ml/viz/` (the viewer module)
- `scripts/dashboard_games.py` (live audit dashboard)
- `scripts/generate_heuristic_games.py` (bulk HA self-play harness)
- `scripts/run_heuristic_audit.sh` (canonical audit-corpus wrapper)
- `yinsh_ml/viz/README.md` (runbook + adapter/annotator API)

**External references:**

- Eric Jang AutoGo: <https://github.com/ericjang/AutoGo>
  (`src/alpha_go/model.py` has SE blocks + GAP in value head)
- temhelk/yngine: <https://github.com/temhelk/yngine> (C++ MCTS +
  bitboards for YINSH; potential third-opponent source for the audit
  harness; basis of YINSIM)
- Dwarkesh × Eric Jang interview, May 2026 (the transcript that
  triggered this audit)
- KataGo paper (David Wu, 2020) — the source of the 40× compute
  reduction claim and the global-pooling techniques
