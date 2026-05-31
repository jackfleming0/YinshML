"""Orchestration commands: schedule runs, inspect the gate, ratify decisions.

These drive the thin-slice pipeline in ``yinsh_ml.orchestration``:

    yinsh-track schedule configs/smoke.yaml --baseline a1b2c3d4
    yinsh-track gate
    yinsh-track ratify <experiment_id>          # approve (promote)
    yinsh-track ratify <experiment_id> --reject # decline

``schedule`` runs the experiment in-process (real training) and then Tier-0
evaluation; ``gate`` lists what's blocked on you; ``ratify`` resolves it.
"""

import click

DEFAULT_DB = "experiments/experiments.db"


@click.command()
@click.argument("config_path")
@click.option("--baseline", "baseline_id", default=None,
              help="Experiment id of the anchor/predecessor to judge against.")
@click.option("--baseline-checkpoint", default=None,
              help="Path to a baseline checkpoint (.pt or run dir) — e.g. your champion "
                   "best_model.pt. Takes precedence over --baseline.")
@click.option("--init-checkpoint", default=None,
              help="Warm-start the candidate from this checkpoint (weights only). The "
                   "'iterate on my champion' loop.")
@click.option("--champion", default=None,
              help="Shortcut for the improvement loop: warm-start FROM and evaluate "
                   "AGAINST this checkpoint (sets both --init-checkpoint and "
                   "--baseline-checkpoint).")
@click.option("--target", type=click.Choice(["local", "cloud"]), default="local",
              help="Where to run (cloud is a stubbed seam).")
@click.option("--name", default="", help="Human label for the run.")
@click.option("--iterations", type=int, default=None,
              help="Override the config's num_iterations (e.g. 1 for a quick run).")
@click.option("--games-dir", default=None,
              help="Replayable game parquet dir for the offense-only audit.")
@click.option("--audit-games", type=int, default=0,
              help="Record N candidate self-play games post-training so the offense-only "
                   "check runs (a few is enough; each is an MCTS game).")
@click.option("--llm/--no-llm", default=True,
              help="Use the Claude PI interpreter for the writeup (needs ANTHROPIC_API_KEY).")
@click.option("--calibration", default="configs/panel_calibration.json",
              help="Panel calibration file (auto-loaded if it exists).")
@click.option("--db", "db_path", default=DEFAULT_DB, help="Path to experiments.db")
@click.option("--output-dir", default="experiments", help="Experiment output dir.")
def schedule(config_path, baseline_id, baseline_checkpoint, init_checkpoint, champion,
             target, name, iterations, games_dir, audit_games, llm, calibration,
             db_path, output_dir):
    """Schedule + run an experiment from CONFIG_PATH, then Tier-0 evaluate it."""
    import os as _os

    # --champion is the improvement-loop shortcut: warm-start from AND judge against
    # the same model (explicit flags win if also given).
    if champion:
        init_checkpoint = init_checkpoint or champion
        baseline_checkpoint = baseline_checkpoint or champion

    from ...orchestration import (
        EvaluationFunnel, ExperimentSpec, FailurePanel, Journal, OrchestrationStore,
        PIInterpreter, Scheduler, TriageWorkflow,
    )

    store = OrchestrationStore(db_path)
    panel = FailurePanel.from_calibration(calibration) if _os.path.isfile(calibration) else None
    if panel is not None:
        click.echo(f"Loaded panel calibration from {calibration}")
    funnel = EvaluationFunnel(panel=panel)
    scheduler = Scheduler(
        store=store,
        journal=Journal(output_dir),
        funnel=funnel,
        interpreter=PIInterpreter() if llm else None,
        triage=TriageWorkflow(funnel) if llm else None,
        audit_games=audit_games,
        output_dir=output_dir,
    )
    spec = ExperimentSpec(
        config_path=config_path, name=name, baseline_id=baseline_id,
        baseline_checkpoint=baseline_checkpoint, init_checkpoint=init_checkpoint,
        target=target, iterations=iterations, games_dir=games_dir,
    )

    _baseline_desc = baseline_checkpoint or baseline_id or "none"
    click.echo(f"Scheduling {config_path} (baseline={_baseline_desc}, target={target})...")
    result = scheduler.process(spec)

    click.echo("")
    click.echo(f"Experiment:  {result.experiment_id}")
    click.echo(f"Launch:      {result.launch_status}")
    if result.decision is not None:
        d = result.decision
        color = {"feed": "green", "gate": "yellow"}.get(d.route, "white")
        click.echo(f"Routing:     " + click.style(d.route.upper(), fg=color))
        click.echo(f"Status:      {d.next_status}")
        click.echo(f"Reason:      {d.reason}")
        if result.report_path:
            click.echo(f"Report:      {result.report_path}")
        if d.gate_kind:
            click.echo(click.style(
                f"\n⚠️  Queued at the gate ({d.gate_kind}) — run `yinsh-track gate`.",
                fg="yellow",
            ))


@click.command()
@click.option("--goal", default="Propose the next experiment to push toward a stronger model.",
              help="The objective to hand the proposer agent.")
@click.option("--db", "db_path", default=DEFAULT_DB, help="Path to experiments.db")
@click.option("--output-dir", default="experiments", help="Experiment output dir.")
def propose(goal, db_path, output_dir):
    """Run the Rung-3 proposer agent: review results, propose the next experiment."""
    import time

    from ...orchestration import Journal, OrchestrationStore, ProposerAgent

    store = OrchestrationStore(db_path)
    agent = ProposerAgent(store)
    click.echo("Proposer agent reviewing the registry...")
    proposal = agent.propose(goal)

    if proposal is None:
        raise click.ClickException(
            "No proposal produced (no ANTHROPIC_API_KEY, agent error, or nothing to propose)."
        )

    proposal_id = f"proposal_{int(time.time())}"
    path = Journal(output_dir).write_proposal(proposal_id, proposal)
    click.echo("")
    click.echo(click.style(f"Proposal: {proposal.hypothesis}", fg="green"))
    click.echo(f"  Base config: {proposal.base_config}")
    click.echo(f"  Overrides:   {proposal.overrides}")
    click.echo(f"  Rationale:   {proposal.rationale}")
    click.echo(f"\nWritten to {path}")
    click.echo(click.style(
        "\nReview it, then run it with `yinsh-track schedule` — the agent does not "
        "spend compute on its own.", fg="yellow",
    ))


@click.command()
@click.option("--db", "db_path", default=DEFAULT_DB, help="Path to experiments.db")
def gate(db_path):
    """List decisions blocked on your sign-off (the ratification queue)."""
    from ...orchestration import OrchestrationStore

    store = OrchestrationStore(db_path)
    items = store.open_gate_items()
    if not items:
        click.echo("Gate is clear — nothing awaiting ratification.")
        return

    click.echo(f"{len(items)} item(s) awaiting ratification:\n")
    for item in items:
        kind_color = {"promotion": "green", "review": "yellow"}.get(item.kind, "white")
        click.echo(
            f"  • {click.style(item.kind.upper(), fg=kind_color)}  "
            f"{item.experiment_id}  —  {item.summary}"
        )
        if item.report_path:
            click.echo(f"      report: {item.report_path}")
    click.echo("\nRatify with: yinsh-track ratify <experiment_id> [--reject]")


@click.command()
@click.argument("experiment_id")
@click.option("--reject", is_flag=True, help="Decline instead of approving.")
@click.option("--db", "db_path", default=DEFAULT_DB, help="Path to experiments.db")
@click.option("--output-dir", default="experiments", help="Experiment output dir.")
def ratify(experiment_id, reject, db_path, output_dir):
    """Resolve the open gate item for EXPERIMENT_ID (approve by default)."""
    from ...orchestration import (
        EvaluationFunnel, Journal, OrchestrationStore, Scheduler,
    )

    store = OrchestrationStore(db_path)
    scheduler = Scheduler(
        store=store, journal=Journal(output_dir),
        funnel=EvaluationFunnel(), output_dir=output_dir,
    )
    try:
        new_status = scheduler.ratify(experiment_id, approve=not reject)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    verb = "Declined" if reject else "Ratified"
    color = "red" if reject else "green"
    click.echo(click.style(f"{verb} — {experiment_id} is now `{new_status}`.", fg=color))
