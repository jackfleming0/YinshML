# Experiment Orchestration

A thin control plane on top of the existing `yinsh_ml.experiments` machinery. It
schedules experiments, runs each through a tiered evaluation funnel, and routes
results to a non-blocking **feed** (visibility) or a blocking **gate**
(ratification). It builds on what already exists rather than replacing it:

| Capability | Reused from |
|---|---|
| Experiment as a first-class object + SQLite registry | `yinsh_ml/experiments/experiment_db.py` |
| Per-experiment execution engine | `scripts/run_training.py` (the real training entrypoint) |
| Win-rate confidence intervals | `yinsh_ml/utils/stats.py::wilson_bounds` |
| Head-to-head match play | `yinsh_ml/utils/tournament.py::ModelTournament` |

The only genuinely new logic is the **SPRT** (`utils/stats.py::sprt_decision`, the
sequential sibling of Wilson) and the **offense-only-equilibrium detector**
(`failure_panel.py`, automating what `scripts/replay_heuristic_vs_heuristic.py`
does by eye). Everything else is wiring.

## Storage split

- **SQLite = tabular truth** (`experiments/experiments.db`, shared with
  `ExperimentDB`): experiment status, `eval_results`, `gate_queue`. Queryable —
  "every candidate that beat its anchor" is one join.
- **Markdown = the narrative** (`experiments/journal/<id>.md` + append-only
  `experiments/FEED.md`): the interpretation and reasons-to-doubt. Git-committable,
  diffable, and the artifact you read during a sniff test.

## Lifecycle

```
queued → (launch) → running → evaluated → route:
    clear loss   → rejected               (auto; cheap & reversible; feed only)
    clear win    → awaiting_ratification   (gate: promotion — irreversible)
    ambiguous    → awaiting_ratification   (gate: review)
ratify → promoted | rejected
```

**Everything** lands in the feed, including the clear-cut results that
auto-advance — auto-proceed is never silent. Only ambiguous results and
irreversible actions (promotion) block at the gate.

## CLI

```bash
yinsh-track schedule configs/smoke.yaml --baseline <exp-id>   # run + Tier-0 evaluate
yinsh-track gate                                              # what's blocked on you
yinsh-track ratify <exp-id>                                   # approve (promote)
yinsh-track ratify <exp-id> --reject                          # decline
```

## What this slice ships (and what it doesn't)

This is the **thin end-to-end slice**: every station is real but minimal, to prove
the seams. Deliberately deferred:

- **Concurrency** — `Scheduler` runs one spec at a time. `max_concurrent` is the
  seam; a worker pool is not yet wired.
- **Cloud-burst** — `CloudLauncher` is a stub. The design guarantee is that
  bursting changes only `ExperimentSpec.target`, never the spec itself.
- **Tiers 1–2** — only Tier 0 (failure panel + SPRT vs predecessor) runs.
  `EvaluationFunnel.run_tier1/run_tier2` are `NotImplementedError` seams for the
  anchor ladder, heuristic/GH-engine gauntlet, and human milestones.
- **PI reasoning** — now LLM-authored (see Rung 1 below); falls back to a template
  when no API key is set.
- **Run-diff trajectory extraction** — **wired** (`trajectories.py`): the
  offense-only check extracts per-game completed-runs-differential trajectories from
  replayable game parquet (`spec.games_dir` or `<save_dir>/parquet_data`). Since
  training self-play writes no replayable games, `--audit-games N` (`audit_games.py`)
  records N candidate MCTS self-play games post-training so the check fires on the
  candidate's own play; without it the check skips gracefully.
- **Baseline by checkpoint path** — `--baseline-checkpoint <path>` (or
  `spec.baseline_checkpoint`) evaluates a candidate against a real model file (e.g.
  your champion `best_model.pt`), not only against an orchestration experiment id.
- **Improvement loop (warm-start)** — `--init-checkpoint <path>` warm-starts the
  candidate from a checkpoint's weights (the `run_training.py --init-checkpoint`
  path: weights only, optimizer/iteration reset). `--champion <path>` is the
  shortcut for "iterate on my best": it warm-starts *from* and evaluates *against*
  the same model. The improvement-loop run is:
  ```bash
  yinsh-track schedule <config> --champion experiments/<run>/best_model.pt \
    --audit-games 10 --iterations <N>
  ```
- **Anchor-ladder update on promotion** — `ratify` marks `promoted`; making the
  promoted model the next baseline is a follow-up.

## Agentic layers (the ladder)

True agency is added in rungs, each reusing the last. The guiding rule throughout:
**the LLM advises, the rules gate** — a model call never decides promotion/rejection
(that stays deterministic in `pi.py`); the LLM enriches judgment and, higher up,
proposes work. All rungs run on Opus 4.8 and degrade gracefully without an API key.

| Rung | Type | Module | Status |
|---|---|---|---|
| **1** | Augmented step (single call) | `interpreter.py::PIInterpreter` | **built** |
| **2** | Workflow (code-controlled tool loop) | `triage.py::TriageWorkflow` | **built** |
| **3** | Agent (model-driven trajectory) | `proposer.py::ProposerAgent` | **built** |

**Rung 1 — PI interpreter.** Replaces the templated writeup with a Claude-authored
interpretation: a structured read (`messages.parse` → Pydantic), reasons-to-doubt,
and a suggested next step. It is *not* an agent — one call, no tools, no loop. The
routing decision is already made deterministically before it runs; the interpreter
only authors the narrative that lands in the journal and feed. If `ANTHROPIC_API_KEY`
is unset or the call fails, `interpret()` returns `None` and the journal falls back
to its template — the pipeline never breaks. Enable/disable per run with
`yinsh-track schedule ... --llm / --no-llm` (on by default). Teaches: structured
outputs, prompt caching (the frozen rubric rides the cache), adaptive thinking,
graceful degradation. Cost ≈ $0.05/run.

**Rung 2 — triage workflow.** The first real agentic loop, but **code holds the
reins**. When a result is *ambiguous* (inconclusive SPRT or a panel flag), Claude is
given bounded tools — `order_more_games`, `inspect_failure` — and decides what
evidence to gather before the human is involved; our code runs the manual loop,
executes the tools, and caps the budget (`max_iterations`, the game budget). The
safety property: the tools only *gather evidence*; the result is then **re-routed on
the deterministically-recomputed SPRT over the new games**, not on Claude's opinion —
so a candidate can self-resolve to a clear win (still gated for promotion) or a clear
loss (auto-rejected), draining the ratification bottleneck without ever letting the
model promote on its own. Wired in the scheduler on the review-gate path only;
shares the `--llm` flag. Teaches: tool-surface design, the manual agentic loop,
human-in-the-loop boundaries.

**Rung 3 — proposer agent.** The self-propagating piece: it reads what's been tried
(`list_experiments`, `get_experiment` — read-only) and *proposes the next experiment
to run*, closing the loop results → proposal → run → results. Unlike Rung 2's narrow
triage, the model drives its own trajectory (which results to inspect, when it has
enough); code only caps `max_iterations`. The guardrail, straight from the autonomy
policy: opening a *new direction* is the irreversible lever, so the tools are
read-only and the output is a **proposal artifact** (`experiments/proposals/<id>.md`)
for you to review and `schedule` — the agent never spends compute itself. Run it with
`yinsh-track propose`. Teaches: agent design, read-only tool surfaces, guardrails,
cost-of-error containment. Cost ≈ $0.11/proposal.

## Production shakedown

Before trusting the pipeline with real compute, run the shakedown — it exercises
the *real* path (train → checkpoint → eval) end-to-end on a small config:

```bash
python scripts/bootstrap_orchestration.py                 # baseline + self-vs-self integrity
python scripts/bootstrap_orchestration.py --full          # + a candidate-vs-baseline run
```

It trains a baseline through `run_training.py`, then plays that checkpoint against a
copy of itself through the funnel and asserts the eval does **not** rate a model
stronger than itself (SPRT must not be `accept_h1`). A model "beating" itself means
the eval is broken — catch it here, not after a week of runs.

### Panel calibration (don't trust guessed thresholds)

The failure-panel thresholds can't be guessed — they depend on the value-head mode
and this codebase's metric scales. Measured against real runs, `value_accuracy` sits
around **0.06–0.12** (it's a 7-class value head), so the original `0.40` default
would have flagged *every* real candidate. `scripts/calibrate_panel.py` derives
grounded thresholds from actual run metrics:

```bash
python scripts/calibrate_panel.py --runs experiments   # writes configs/panel_calibration.json
```

`FailurePanel.from_calibration(...)` loads them, and `yinsh-track schedule` /
the bootstrap auto-load `configs/panel_calibration.json` if present. A grounded
calibration (from the repo's existing runs) is committed; **re-run the tool on your
Mac with your real configs** — thresholds tuned to a CPU smoke aren't the right ones
for a Mac-trained model.

### Loss-divergence collapse check

`value_accuracy` is a *weak* health signal at this scale (it barely moves), so the
panel also runs **loss-divergence checks** on `value_loss` and `policy_loss` — the
signals that actually catch a broken run. They flag a non-finite (NaN/Inf) loss
unconditionally, and a finite loss above a calibrated ceiling (3× the worst observed
real-run loss, so only a genuine explosion trips it — e.g. `value_loss > 27.67`).
The NaN/Inf guard needs no calibration; the ceiling comes from `calibrate_panel.py`.

The **policy-entropy collapse check is now live**: the trainer logs the network's
predicted-policy entropy (`EpochMetrics.policy_entropy`, distinct from the MCTS
target entropy), the launcher threads it into the panel, and `policy_entropy` flags
when the policy collapses onto a narrow set of moves. Its floor is calibrated by
`calibrate_panel.py` once you have runs trained after this change (older runs lack
the field, so it skips on them). Recalibrate on the Mac after your first run to
ground `min_policy_entropy` to your model's real entropy scale.

### Why `LocalLauncher` drives `run_training.py`

The repo's configs are all in the campaign format
(`self_play:`/`trainer:`/`arena:`/`num_iterations:`) consumed by `run_training.py`.
An earlier version of the launcher went through `experiments.ExperimentRunner`,
which expects a *different* schema and whose loader **silently drops unrecognized
keys** — so pointing it at a real config quietly ran a default 10-iteration job
instead of the configured one. `LocalLauncher` now subprocesses the real entrypoint
(so orchestrated runs behave exactly like manual ones) and records the run in the
shared `experiments.db` itself. Checkpoints land under
`<output_dir>/<experiment_id>/<timestamp>/iteration_<N>/`; the locator globs
recursively, so the match runner and panel find them regardless of the timestamp dir.

> **Note:** `run_training.py` imports `coremltools` (macOS CoreML export). It's in
> `requirements.txt`; on a fresh non-Mac box you may need `pip install coremltools`
> for the import to resolve (the native proxy is unused off-Mac — the warnings are
> benign).

## Note on imports

Importing this package transitively pulls `torch` via `yinsh_ml.utils` (whose
`__init__` eagerly imports the viz stack). The orchestration modules themselves
defer the heavy tournament/model code via lazy imports, but the `utils` package
coupling means a fully torch-free import would require making `utils/__init__`
lazy — a separate, broader change.
