"""Experiment orchestration layer.

A thin control plane on top of the existing ``yinsh_ml.experiments`` machinery:
schedules experiments, runs them through a tiered evaluation funnel, and routes
results to either a non-blocking *feed* (visibility) or a blocking *gate*
(ratification). Tabular truth lives in the existing ``experiments.db`` SQLite;
the human-readable narrative is written as Markdown by ``journal``.

This is the "thin end-to-end slice": every station is real but minimal —
single-run scheduling (concurrency is a seam, not yet wired), local launch only
(cloud is a stubbed target), Tier-0 evaluation only (Tiers 1-2 stubbed), and a
templated PI writeup (LLM reasoning deepens later). See the module docstrings for
what each station defers.
"""

from .spec import ExperimentSpec
from .registry import OrchestrationStore, EvalResult, GateItem, Status
from .launcher import Launcher, LocalLauncher, CloudLauncher, LaunchResult
from .match_runner import MatchRunner, MatchOutcome
from .failure_panel import FailurePanel, PanelResult, CheckResult
from .funnel import EvaluationFunnel, Tier0Result
from .interpreter import Interpretation, PIInterpreter
from .journal import Journal
from .pi import PIRouter, RoutingDecision
from .scheduler import Scheduler

__all__ = [
    "ExperimentSpec",
    "OrchestrationStore", "EvalResult", "GateItem", "Status",
    "Launcher", "LocalLauncher", "CloudLauncher", "LaunchResult",
    "MatchRunner", "MatchOutcome",
    "FailurePanel", "PanelResult", "CheckResult",
    "EvaluationFunnel", "Tier0Result",
    "Journal",
    "PIRouter", "RoutingDecision",
    "Interpretation", "PIInterpreter",
    "Scheduler",
]
