"""Markdown narrative writer — the prose half of the storage split.

SQLite holds the queryable facts; this writes the human-readable story: one
``journal/<exp-id>.md`` report per candidate (config summary, what it tested,
the interpretation, and an explicit *reasons-to-doubt* section), plus an
append-only ``FEED.md`` digest where **everything** lands — including the
clear-cut results that auto-advance without your sign-off. The feed is the
"I still want to read the ones I'm not reviewing" channel; the gate (in SQLite)
is the blocking one.

These files are git-committable on purpose: they're the durable record that
survives the ephemeral container, and they're the artifact you and I read during
the sniff test.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .funnel import Tier0Result
    from .interpreter import Interpretation
    from .pi import RoutingDecision
    from .spec import ExperimentSpec


class Journal:
    def __init__(self, root: str = "experiments"):
        self.root = Path(root)
        self.journal_dir = self.root / "journal"
        self.feed_path = self.root / "FEED.md"

    def write_report(
        self,
        experiment_id: str,
        spec: "ExperimentSpec",
        tier0: "Tier0Result",
        decision: "RoutingDecision",
        interpretation: "Interpretation | None" = None,
    ) -> str:
        """Write the per-experiment report; return its path.

        When an ``interpretation`` is supplied (Rung 1 — the PI interpreter), its
        Claude-authored read and reasons-to-doubt are rendered; otherwise the
        report falls back to the deterministic template.
        """
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        path = self.journal_dir / f"{experiment_id}.md"
        path.write_text(
            self._render_report(experiment_id, spec, tier0, decision, interpretation)
        )
        return str(path)

    def append_feed(self, experiment_id: str, headline: str, detail: str) -> None:
        """Append a one-entry digest line for this result to FEED.md."""
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.feed_path.exists():
            self.feed_path.write_text("# Experiment Feed\n\n")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"- **{ts}** · `{experiment_id}` — {headline}\n  - {detail}\n"
        with self.feed_path.open("a") as f:
            f.write(entry)

    # --- rendering --------------------------------------------------------

    def _render_report(
        self,
        experiment_id: str,
        spec: "ExperimentSpec",
        tier0: "Tier0Result",
        decision: "RoutingDecision",
        interpretation: "Interpretation | None" = None,
    ) -> str:
        o = tier0.outcome
        lines = [
            f"# Experiment `{experiment_id}` — {spec.name or 'unnamed'}",
            "",
            f"*Generated {datetime.now().isoformat(timespec='seconds')}*",
            "",
            "## What this tested",
            "",
            f"- Config: `{spec.config_path}`",
            f"- Baseline: `{spec.baseline_id or '(none — first candidate)'}`",
            f"- Launch target: `{spec.target}`",
            "",
            "## Result",
            "",
            f"- **Routing:** {decision.route.upper()} — {decision.reason}",
            f"- **Status:** `{decision.next_status}`",
            "",
            "### Match (candidate's perspective)",
            "",
            f"- Record: **{o.wins}W / {o.draws}D / {o.losses}L** ({o.total} games)",
        ]
        if tier0.had_baseline:
            lines.append(
                f"- Win rate (decisive): "
                f"[{tier0.wilson_lower:.2f}, {tier0.wilson_upper:.2f}] Wilson 95%"
            )
        if tier0.sprt is not None:
            lines += [
                f"- SPRT: **{tier0.sprt.verdict}** "
                f"(LLR {tier0.sprt.llr:.2f}; bounds [{tier0.sprt.lower:.2f}, {tier0.sprt.upper:.2f}])",
            ]
        else:
            lines.append("- SPRT: not run (no baseline to play against)")

        lines += ["", "### Failure-mode panel", ""]
        for c in tier0.panel.checks:
            mark = "⚠️ SKIP" if c.skipped else ("✅" if c.passed else "❌")
            lines.append(f"- {mark} **{c.name}** — {c.detail}")

        # PI read (Rung 1 — Claude-authored). Present only when the interpreter ran.
        if interpretation is not None:
            lines += [
                "",
                f"## PI read ({self._model_label})",
                "",
                f"*Confidence: {interpretation.confidence}*",
                "",
                interpretation.assessment,
                "",
                f"**Suggested next step:** {interpretation.suggested_next_step}",
            ]

        # Reasons-to-doubt: the honesty channel. Prefer the PI's, fall back to template.
        doubts = (
            interpretation.reasons_to_doubt
            if interpretation is not None and interpretation.reasons_to_doubt
            else self._reasons_to_doubt(tier0)
        )
        lines += ["", "## Reasons to doubt this conclusion", ""]
        if doubts:
            lines += [f"- {d}" for d in doubts]
        else:
            lines.append("- None surfaced by Tier-0. (Tiers 1-2 not yet run.)")

        lines.append("")
        return "\n".join(lines)

    _model_label = "Claude Opus 4.8"

    @staticmethod
    def _reasons_to_doubt(tier0: "Tier0Result") -> list[str]:
        doubts: list[str] = []
        for c in tier0.panel.skips:
            doubts.append(f"`{c.name}` was skipped — {c.detail}")
        if tier0.had_baseline and tier0.sprt is not None and not tier0.sprt.decisive:
            doubts.append(
                "SPRT did not reach a boundary — strength difference is within "
                "the indifference region (could be noise either way)."
            )
        if tier0.had_baseline and tier0.outcome.decisive < 10:
            doubts.append(
                f"Only {tier0.outcome.decisive} decisive games — small sample."
            )
        doubts.append("Only Tier-0 ran; no anchor ladder, GH engines, or human review yet.")
        return doubts
