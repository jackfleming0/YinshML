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
@click.option("--target", type=click.Choice(["local", "cloud"]), default="local",
              help="Where to run (cloud is a stubbed seam).")
@click.option("--name", default="", help="Human label for the run.")
@click.option("--llm/--no-llm", default=True,
              help="Use the Claude PI interpreter for the writeup (needs ANTHROPIC_API_KEY).")
@click.option("--db", "db_path", default=DEFAULT_DB, help="Path to experiments.db")
@click.option("--output-dir", default="experiments", help="Experiment output dir.")
def schedule(config_path, baseline_id, target, name, llm, db_path, output_dir):
    """Schedule + run an experiment from CONFIG_PATH, then Tier-0 evaluate it."""
    from ...orchestration import (
        EvaluationFunnel, ExperimentSpec, Journal, OrchestrationStore,
        PIInterpreter, Scheduler,
    )

    store = OrchestrationStore(db_path)
    scheduler = Scheduler(
        store=store,
        journal=Journal(output_dir),
        funnel=EvaluationFunnel(),
        interpreter=PIInterpreter() if llm else None,
        output_dir=output_dir,
    )
    spec = ExperimentSpec(
        config_path=config_path, name=name, baseline_id=baseline_id, target=target,
    )

    click.echo(f"Scheduling {config_path} (baseline={baseline_id or 'none'}, target={target})...")
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
