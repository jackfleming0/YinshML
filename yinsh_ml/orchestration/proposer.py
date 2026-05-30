"""Rung 3 of the agentic ladder: the next-experiment proposer (a true *agent*).

This is the genuinely self-propagating piece: it reads what's been tried and its
outcomes, then proposes what to run next — closing the research loop
(results → proposal → run → results). Unlike Rung 2's narrow triage, the agent
drives its own trajectory: it decides which experiments to look at, how deep to
drill, and when it has enough to propose. Code still caps the loop (max_iterations).

Guardrails, straight from the agreed autonomy policy — opening a *new research
direction* is the irreversible lever, so:
- The agent's exploration tools are **read-only** (`list_experiments`,
  `get_experiment`). It cannot mutate state or spend compute.
- Its output is a **Proposal artifact** for the human to review and schedule. It
  never enqueues or runs anything itself. The human choosing to `schedule` the
  proposal *is* the gate on new directions.

This satisfies the "should I build an agent?" test: the task is open-ended
(synthesize many results into a next step), high-value, viable, and the cost of a
bad proposal is fully contained (a human reads it before any compute is spent).
Degrades to ``None`` on any failure. Model: Opus 4.8, adaptive thinking.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .registry import OrchestrationStore

logger = logging.getLogger(__name__)

_SYSTEM_RUBRIC = """\
You are the principal investigator (PI) for a YINSH AlphaZero-style training \
program, deciding the single next experiment to run to push toward a stronger \
model. Use the read-only tools to review what has been tried and how it turned \
out, then call propose_experiment exactly once with a concrete, testable plan.

Principles:
- Propose ONE experiment that isolates ONE lever (e.g. MCTS simulations, c_puct, \
heuristic_weight, value_lr_factor, temperature schedule, encoding type). A clean \
single-variable change is worth more than a vague direction.
- Prefer cheap, decisive experiments: ones whose SPRT vs the current baseline will \
resolve quickly and that exercise a real hypothesis.
- Use the evidence. If a prior result flagged offense_only_equilibrium, a search-\
depth / simulations experiment is well-motivated. If policy entropy collapsed, look \
at temperature or learning rate. Don't repeat an experiment that already settled.
- Give the base config to start from, the concrete overrides (dotted keys), a \
one-line testable hypothesis, and the rationale tying it to the evidence.
You do not run anything — a human reviews and schedules your proposal. Reason only \
from tool results; never invent results that weren't returned."""


@dataclass
class Proposal:
    base_config: str
    overrides: Dict[str, Any]
    hypothesis: str
    rationale: str

    def to_markdown(self, proposal_id: str) -> str:
        lines = [
            f"# Experiment proposal `{proposal_id}`",
            "",
            f"**Hypothesis:** {self.hypothesis}",
            "",
            f"**Base config:** `{self.base_config}`",
            "",
            "**Overrides:**",
            "",
            "```json",
            json.dumps(self.overrides, indent=2),
            "```",
            "",
            "**Rationale:**",
            "",
            self.rationale,
            "",
            "---",
            "*To run this proposal, review it and schedule it:*",
            "",
            "```bash",
            f"yinsh-track schedule {self.base_config} --baseline <current-anchor>",
            "```",
            "",
        ]
        return "\n".join(lines)


_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "list_experiments",
        "description": "List recent experiments with status and their latest eval "
        "(record, SPRT verdict, panel health). The overview to start from.",
        "input_schema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "description": "How many to list."}},
        },
    },
    {
        "name": "get_experiment",
        "description": "Full config and eval history for one experiment id (drill-down).",
        "input_schema": {
            "type": "object",
            "properties": {"experiment_id": {"type": "string"}},
            "required": ["experiment_id"],
        },
    },
    {
        "name": "propose_experiment",
        "description": "Submit the single next experiment to run. Terminal — call once.",
        "input_schema": {
            "type": "object",
            "properties": {
                "base_config": {"type": "string", "description": "Config file to start from."},
                "overrides": {
                    "type": "object",
                    "description": "Dotted-key → value overrides for the single lever.",
                },
                "hypothesis": {"type": "string", "description": "One-line testable hypothesis."},
                "rationale": {"type": "string", "description": "Why, tied to the evidence."},
            },
            "required": ["base_config", "overrides", "hypothesis", "rationale"],
        },
    },
]


class ProposerAgent:
    """Read-only registry exploration → one experiment proposal."""

    def __init__(
        self,
        store: "OrchestrationStore",
        model: str = "claude-opus-4-8",
        max_tokens: int = 8000,
        max_iterations: int = 6,
        client=None,
    ):
        self.store = store
        self.model = model
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self._client = client

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def propose(
        self, goal: str = "Propose the next experiment to push toward a stronger model."
    ) -> Optional[Proposal]:
        """Run the agent loop; return its Proposal, or None on failure/no proposal."""
        try:
            client = self._get_client()
            messages: List[Dict[str, Any]] = [{"role": "user", "content": goal}]

            for _ in range(self.max_iterations):
                response = client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    thinking={"type": "adaptive"},
                    system=[
                        {"type": "text", "text": _SYSTEM_RUBRIC, "cache_control": {"type": "ephemeral"}}
                    ],
                    tools=_TOOL_SCHEMAS,
                    messages=messages,
                )
                messages.append({"role": "assistant", "content": response.content})
                tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
                if not tool_uses:
                    break

                proposal, results = self._run_tools(tool_uses)
                messages.append({"role": "user", "content": results})
                if proposal is not None:
                    return proposal

            return None  # explored but never proposed
        except Exception as exc:  # noqa: BLE001 - advisory layer must never break
            logger.warning("Proposer agent unavailable (%s).", exc)
            return None

    def _run_tools(self, tool_uses):
        proposal: Optional[Proposal] = None
        results: List[Dict[str, Any]] = []
        for b in tool_uses:
            if b.name == "propose_experiment":
                proposal = self._parse_proposal(b.input)
                content = "Proposal recorded."
            elif b.name == "list_experiments":
                content = self._fmt_digest(b.input.get("limit", 20))
            elif b.name == "get_experiment":
                content = self._fmt_detail(b.input.get("experiment_id", ""))
            else:
                content = f"Unknown tool: {b.name}"
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": content})
        return proposal, results

    def _fmt_digest(self, limit: int) -> str:
        digest = self.store.experiment_digest(int(limit))
        if not digest:
            return "No experiments have been run yet."
        lines = []
        for d in digest:
            ev = d["last_eval"]
            tail = (
                f" | last: {ev['record']} vs {ev['baseline']}, SPRT {ev['sprt']}, "
                f"panel {'green' if ev['panel_green'] else 'FLAGGED'}"
                if ev else " | no eval"
            )
            lines.append(f"- {d['experiment_id']} ({d['name']}) [{d['status']}]{tail}")
        return "\n".join(lines)

    def _fmt_detail(self, experiment_id: str) -> str:
        detail = self.store.experiment_detail(experiment_id)
        if detail is None:
            return f"No experiment with id '{experiment_id}'."
        return json.dumps(detail, indent=2)

    @staticmethod
    def _parse_proposal(data: Dict[str, Any]) -> Proposal:
        return Proposal(
            base_config=str(data.get("base_config", "")),
            overrides=dict(data.get("overrides", {}) or {}),
            hypothesis=str(data.get("hypothesis", "")),
            rationale=str(data.get("rationale", "")),
        )
