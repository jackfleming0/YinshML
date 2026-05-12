"""Tests for the supervisor ⇄ MetricsLogger wiring (#12 + #13).

Wave-2 smoke surfaced two related bugs:
  * #12 — ``run_anchor_eval`` calls ``metrics_logger.log_eval_value_pair`` /
    ``compute_and_log_value_outcome_correlation`` on the supervisor's proxy,
    but the proxy only exposed ``log_event`` / ``log_scalar`` (W1e). The
    missing methods raised ``AttributeError`` mid-eval.
  * #13 — Supervisor never instantiated a real ``MetricsLogger``, so all
    B1 (effective_child_visits), B2 (value_outcome_correlation), and B3
    (policy_target_entropy) telemetry silently no-op'd.

These tests pin the fix: the supervisor owns a real ``MetricsLogger``,
the proxy delegates to it for the B2 API + iteration boundaries, and an
iteration's worth of B1 / B2 / B3 calls land in
``save_dir/metrics/iteration_<N>.json`` after ``save_iteration()``.
"""

import json
import types
from pathlib import Path

import pytest

from yinsh_ml.training.supervisor import (
    TrainingSupervisor,
    _SupervisorMetricsProxy,
)
from yinsh_ml.utils.metrics_logger import MetricsLogger


# ---------------------------------------------------------------------------
# Helpers — bypass the heavy TrainingSupervisor constructor; we only need
# the metrics-wiring slice.
# ---------------------------------------------------------------------------


def _make_supervisor_with_metrics(save_dir: Path) -> TrainingSupervisor:
    """Build a TrainingSupervisor with just enough state to exercise the
    metrics-wiring slice. The full ctor instantiates SelfPlay, YinshTrainer,
    a NetworkWrapper, etc. — none of that is needed here."""
    sup = TrainingSupervisor.__new__(TrainingSupervisor)
    sup.save_dir = Path(save_dir)
    sup.save_dir.mkdir(parents=True, exist_ok=True)
    import logging
    sup.logger = logging.getLogger('test_supervisor_metrics')
    sup.experiment_tracker = None
    sup.experiment_id = None
    # Replicate the lines from TrainingSupervisor.__init__ we care about.
    sup.metrics_logger = MetricsLogger(save_dir=sup.save_dir, debug=False)
    sup._tournament_metrics_proxy = _SupervisorMetricsProxy(
        sup, metrics_logger=sup.metrics_logger,
    )
    return sup


# ---------------------------------------------------------------------------
# 1. metrics_logger is instantiated and is a MetricsLogger
# ---------------------------------------------------------------------------


def test_supervisor_owns_real_metrics_logger(tmp_path: Path):
    """#13: after construction the supervisor must hold a real
    MetricsLogger, not None or the proxy."""
    sup = _make_supervisor_with_metrics(tmp_path)
    assert sup.metrics_logger is not None
    assert isinstance(sup.metrics_logger, MetricsLogger)
    # And the on-disk dir was created.
    assert (sup.save_dir / "metrics").is_dir()


def test_tournament_proxy_holds_metrics_logger_reference(tmp_path: Path):
    """The proxy must hold the same MetricsLogger instance the supervisor
    owns, so delegation lands in one place."""
    sup = _make_supervisor_with_metrics(tmp_path)
    assert sup._tournament_metrics_proxy._metrics_logger is sup.metrics_logger


# ---------------------------------------------------------------------------
# 2. Proxy exposes the methods that run_anchor_eval calls (#12 fix)
# ---------------------------------------------------------------------------


def test_proxy_exposes_b2_methods(tmp_path: Path):
    """Pre-fix the proxy didn't have these — ``run_anchor_eval`` crashed
    with AttributeError mid-eval (Wave-2 smoke)."""
    sup = _make_supervisor_with_metrics(tmp_path)
    proxy = sup._tournament_metrics_proxy
    # Bound, callable methods on the proxy.
    assert callable(getattr(proxy, 'log_eval_value_pair', None))
    assert callable(getattr(proxy, 'compute_and_log_value_outcome_correlation', None))
    assert callable(getattr(proxy, 'start_iteration', None))
    assert callable(getattr(proxy, 'save_iteration', None))


def test_proxy_delegates_value_pair_to_real_logger(tmp_path: Path):
    """A pair pushed through the proxy must end up in the real logger's
    buffer (verified indirectly via correlation aggregation)."""
    sup = _make_supervisor_with_metrics(tmp_path)
    proxy = sup._tournament_metrics_proxy
    sup.metrics_logger.start_iteration(iteration=0)

    # Inject a perfectly correlated set of pairs via the proxy.
    proxy.log_eval_value_pair(1.0, 1.0)
    proxy.log_eval_value_pair(-1.0, -1.0)
    proxy.log_eval_value_pair(0.5, 0.5)
    proxy.log_eval_value_pair(-0.5, -0.5)

    r = proxy.compute_and_log_value_outcome_correlation(step=0)
    assert r is not None
    assert r == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. Old proxy behavior still works without a backing logger (back-compat)
# ---------------------------------------------------------------------------


def test_proxy_without_backing_logger_is_safe():
    """Older unit tests construct the proxy without a MetricsLogger. The
    delegating methods must no-op rather than crash, and log_event /
    log_scalar must still forward to the supervisor's experiment tracker."""
    calls = []

    class _FakeSupervisor:
        def __init__(self):
            import logging
            self.logger = logging.getLogger('fake_super')

        def _log_metric_safe(self, name, value, iteration=None):
            calls.append((name, float(value), iteration))

    sup = _FakeSupervisor()
    proxy = _SupervisorMetricsProxy(sup)  # no metrics_logger
    # These must not raise.
    proxy.log_eval_value_pair(1.0, 1.0)
    assert proxy.compute_and_log_value_outcome_correlation(step=0) is None
    proxy.start_iteration(0)
    proxy.save_iteration()
    # And event forwarding still works.
    proxy.log_event('foo', severity='info', iteration=2, details={})
    proxy.log_scalar('bar', 1.5, iteration=2)
    assert ('event/foo', 1.0, 2) in calls
    assert ('bar', 1.5, 2) in calls


# ---------------------------------------------------------------------------
# 4. End-to-end: B1 / B2 / B3 calls land in iteration_<N>.json
# ---------------------------------------------------------------------------


def test_iteration_json_contains_all_safeguards(tmp_path: Path):
    """Mirror what train_iteration does: start_iteration, emit a B1
    scalar (effective_child_visits), a B3 scalar (policy_target_entropy),
    a B2 correlation (via the proxy), then save_iteration. The resulting
    JSON must contain all three scalar streams."""
    sup = _make_supervisor_with_metrics(tmp_path)
    ml = sup.metrics_logger
    proxy = sup._tournament_metrics_proxy

    iteration = 7
    ml.start_iteration(iteration)

    # B1: MCTS effective child visits (one observation, but log_scalar
    # supports multiple).
    ml.log_scalar('mcts/effective_child_visits', 3.2, step=iteration)

    # B3: policy-target entropy from the trainer.
    ml.log_scalar('train/policy_target_entropy_mean', 0.55, step=iteration)

    # B2: via the proxy — exercises the #12 wiring.
    proxy.log_eval_value_pair(1.0, 1.0)
    proxy.log_eval_value_pair(-1.0, -1.0)
    proxy.log_eval_value_pair(0.0, 0.0)
    proxy.log_eval_value_pair(0.5, 0.5)
    proxy.compute_and_log_value_outcome_correlation(step=iteration)

    ml.save_iteration()

    out = sup.save_dir / "metrics" / f"iteration_{iteration}.json"
    assert out.exists(), f"expected {out} after save_iteration()"
    payload = json.loads(out.read_text())
    scalars = payload['metrics']['scalars']

    assert 'mcts/effective_child_visits' in scalars
    assert 'train/policy_target_entropy_mean' in scalars
    assert 'eval/value_outcome_correlation' in scalars

    # Sanity on the values themselves.
    b1 = scalars['mcts/effective_child_visits'][0]
    assert b1['value'] == pytest.approx(3.2)
    assert b1['step'] == iteration

    b3 = scalars['train/policy_target_entropy_mean'][0]
    assert b3['value'] == pytest.approx(0.55)

    b2 = scalars['eval/value_outcome_correlation'][0]
    # Pairs were perfectly correlated → r ≈ 1.0.
    assert b2['value'] == pytest.approx(1.0, abs=1e-6)
    assert b2['step'] == iteration


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
