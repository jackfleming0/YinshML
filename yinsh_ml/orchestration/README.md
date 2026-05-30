# Experiment Orchestration

A thin control plane on top of the existing `yinsh_ml.experiments` machinery. It
schedules experiments, runs each through a tiered evaluation funnel, and routes
results to a non-blocking **feed** (visibility) or a blocking **gate**
(ratification). It builds on what already exists rather than replacing it:

| Capability | Reused from |
|---|---|
| Experiment as a first-class object + SQLite registry | `yinsh_ml/experiments/experiment_db.py` |
| Per-experiment execution engine | `yinsh_ml/training/supervisor.py` via `ExperimentRunner` |
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
- **PI reasoning** — the writeup is templated from the result row, not yet
  LLM-authored. The routing seam stays the same when that deepens.
- **Run-diff trajectory extraction** — the offense-only check runs on supplied
  trajectories; wiring it from parquet is the `_default_panel_input` gap (the
  check skips, visibly, until then).
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
| **3** | Agent (model-driven trajectory) | next-experiment proposer | planned |

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

## Note on imports

Importing this package transitively pulls `torch` via `yinsh_ml.utils` (whose
`__init__` eagerly imports the viz stack). The orchestration modules themselves
defer the heavy tournament/model code via lazy imports, but the `utils` package
coupling means a fully torch-free import would require making `utils/__init__`
lazy — a separate, broader change.
