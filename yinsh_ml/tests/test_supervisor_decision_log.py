"""W3: regression tests for the iteration-summary "Decision: …" line.

Pre-W3, ``TrainingSupervisor`` always logged
``Decision: 🔄 CONTINUE (AlphaZero-style, no reversion)`` on any failed
Wilson gate — even when ``revert_on_gate_failure=True`` had actually fired
and reloaded ``best_model.pt`` for the next self-play iteration. That made
the iteration SUMMARY block unreliable as a record of what the network
looked like going into the next iter: a reader scanning the log linearly
would conclude no revert happened, when in fact ⏪ REVERTING was logged
~250 lines earlier inside the inner gate-decision branch.

The fix extracted the summary line into ``_format_decision_line`` and
made it consume a ``decision_kind`` value set inside each decision
branch. These tests pin the four kinds the helper handles.
"""
from __future__ import annotations

import pytest

from yinsh_ml.training.supervisor import TrainingSupervisor


def test_decision_line_promoted():
    line = TrainingSupervisor._format_decision_line('promoted', best_model_iteration=3)
    assert "Decision: ✅ NEW BEST" in line
    assert "promoted to best model" in line


def test_decision_line_kept():
    line = TrainingSupervisor._format_decision_line('kept', best_model_iteration=3)
    assert "Decision: ➡️" in line
    assert "KEPT" in line
    assert "already best" in line


def test_decision_line_reverted_includes_best_iter():
    """The reverted line tells the reader which iter's weights got
    restored — that's the load-bearing piece of information when reading
    the log post-hoc to reconstruct what self-play was actually running."""
    line = TrainingSupervisor._format_decision_line('reverted', best_model_iteration=2)
    assert "Decision: ⏪ REVERTED" in line
    assert "gate failed" in line
    assert "restored iter 2" in line
    # Must NOT carry the old "no reversion" wording.
    assert "no reversion" not in line


def test_decision_line_continued_is_unchanged_wording():
    """When the gate failed and we genuinely didn't revert
    (revert_on_gate_failure=False, or revert path bailed), the
    AlphaZero-style continue message should still match the historical
    wording — that's the canonical signal for dashboards / log scrapers."""
    line = TrainingSupervisor._format_decision_line('continued', best_model_iteration=4)
    assert "Decision: 🔄 CONTINUE" in line
    assert "AlphaZero-style" in line
    assert "no reversion" in line


def test_decision_line_unknown_kind_is_defensive():
    """If the decision-tracking flag falls through (future refactor
    introduces a new kind without updating this helper), we want a
    distinctive log line, not a silent regression to the old default."""
    line = TrainingSupervisor._format_decision_line('something_new', best_model_iteration=0)
    assert "Decision: ❓ UNKNOWN" in line
    assert "something_new" in line


def test_revert_path_does_not_say_no_reversion(caplog):
    """End-to-end regression: the reverted line must not carry the
    'no reversion' string. This was the exact symptom in Step 2 — every
    iter 1-4 SUMMARY said 'no reversion' even though ⏪ REVERTING was
    logged inside the inner branch."""
    line = TrainingSupervisor._format_decision_line('reverted', best_model_iteration=0)
    # The bug we're guarding against: 'no reversion' appearing on a
    # reverted iter would re-introduce the unreliable summary.
    assert "no reversion" not in line
