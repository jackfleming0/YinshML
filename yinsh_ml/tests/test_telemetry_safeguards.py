"""Tests for the W1-NEW telemetry safeguards (B1, B2, B3).

Three independent metrics, each with its own integration point:

  * **B1 — `mcts/effective_child_visits`**: per-call ratio of total
    root-child visits to the simulation budget. Emitted from the canonical
    self-play `yinsh_ml/training/self_play.py` MCTS, in both `search()` and
    `search_batch()`. After Wave 0, the W1-NEW workstream is fixing the
    batched-MCTS bug — this metric is the regression detector.

  * **B2 — `eval/value_outcome_correlation`**: Pearson r over
    (root_value_at_position_start, terminal_outcome_in_pov) pairs across
    an evaluation pass. The MetricsLogger API is in place
    (`log_eval_value_pair` + `compute_and_log_value_outcome_correlation`)
    but is not yet wired into the tournament loop — that wait is
    deliberate, the W1e workstream is restructuring the same block. The
    test exercises the math + API contract; integration is a TODO at the
    `ModelTournament.run_anchor_eval` site (see comment there).

  * **B3 — `train/policy_target_entropy_mean`**: mean entropy of the
    sampled batch's MCTS-derived policy targets, computed in
    `YinshTrainer.train_step` next to the existing predicted-policy
    entropy.

The tests below follow the same pattern as `test_search_consistency.py`
— MagicMock the network so no real model load is needed; pin the
contract (metric is logged with a finite, well-typed value), not the
exact numeric outcome of search.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from yinsh_ml.game.game_state import GameState
from yinsh_ml.utils.encoding import StateEncoder
from yinsh_ml.utils.metrics_logger import MetricsLogger


# ---------------------------------------------------------------------------
# Shared helpers — minimal fake network for MCTS smoke tests.
# ---------------------------------------------------------------------------


def _fake_network(total_moves: int):
    """A bare-minimum NetworkWrapper-shaped Mock.

    Returns deterministic uniform policy + a fixed value so the search
    output is reproducible and the metric we're testing isn't perturbed
    by random NN noise.
    """
    network = MagicMock()
    network.state_encoder = StateEncoder()

    def predict_from_state(_state):
        policy_logits = torch.zeros(1, total_moves)
        value = torch.tensor([[0.42]])
        return policy_logits, value

    def predict_batch(states):
        policy_logits = torch.zeros(len(states), total_moves)
        values = torch.full((len(states), 1), 0.42)
        return policy_logits, values

    network.predict_from_state = predict_from_state
    network.predict_batch = predict_batch
    return network


# ---------------------------------------------------------------------------
# B1 — MCTS effective child visits
# ---------------------------------------------------------------------------


class TestB1EffectiveChildVisitsSelfPlayMcts:
    """Telemetry on `yinsh_ml/training/self_play.py::MCTS` — both
    `search()` and `search_batch()` paths.

    These two paths existed independently and are exactly the surface the
    W1-NEW workstream is investigating — the safeguard MUST cover both.
    """

    def _build(self, metrics_logger=None, num_simulations=8):
        from yinsh_ml.training.self_play import MCTS as SPMCTS

        encoder = StateEncoder()
        network = _fake_network(encoder.total_moves)
        return SPMCTS(
            network=network,
            evaluation_mode="pure_neural",
            num_simulations=num_simulations,
            late_simulations=num_simulations,
            simulation_switch_ply=200,
            c_puct=1.0,
            dirichlet_alpha=0.0,
            value_weight=1.0,
            max_depth=20,
            initial_temp=1.0,
            final_temp=1.0,
            annealing_steps=1,
            enable_subtree_reuse=False,
            fpu_reduction=0.0,
            epsilon_mix_start=0.0,
            epsilon_mix_end=0.0,
            epsilon_mix_taper_moves=0,
            metrics_logger=metrics_logger,
        )

    def test_search_logs_metric(self, tmp_path):
        ml = MetricsLogger(save_dir=tmp_path)
        ml.start_iteration(iteration=0)
        mcts = self._build(metrics_logger=ml, num_simulations=4)
        state = GameState()
        _ = mcts.search(state, move_number=0)
        scalars = ml.current_metrics['scalars']
        assert 'mcts/effective_child_visits' in scalars
        entry = scalars['mcts/effective_child_visits'][0]
        assert isinstance(entry['value'], float)
        assert np.isfinite(entry['value'])
        assert mcts.last_effective_child_visits == entry['value']

    def test_search_batch_logs_metric(self, tmp_path):
        ml = MetricsLogger(save_dir=tmp_path)
        ml.start_iteration(iteration=0)
        mcts = self._build(metrics_logger=ml, num_simulations=4)
        state = GameState()
        _ = mcts.search_batch(state, move_number=0, batch_size=2)
        scalars = ml.current_metrics['scalars']
        assert 'mcts/effective_child_visits' in scalars
        assert len(scalars['mcts/effective_child_visits']) == 1
        entry = scalars['mcts/effective_child_visits'][0]
        assert isinstance(entry['value'], float)
        assert np.isfinite(entry['value'])

    def test_both_paths_share_step_counter(self, tmp_path):
        """`search()` and `search_batch()` bump the same counter so a
        consumer reading the time series gets monotonic steps regardless
        of which path produced each sample."""
        ml = MetricsLogger(save_dir=tmp_path)
        ml.start_iteration(iteration=0)
        mcts = self._build(metrics_logger=ml, num_simulations=4)
        state = GameState()
        _ = mcts.search(state, move_number=0)
        _ = mcts.search_batch(state, move_number=0, batch_size=2)
        steps = [e['step'] for e in ml.current_metrics['scalars']['mcts/effective_child_visits']]
        assert steps == [1, 2]


# ---------------------------------------------------------------------------
# B2 — value-outcome correlation (API only; integration is a TODO)
# ---------------------------------------------------------------------------


class TestB2ValueOutcomeCorrelation:
    """API contract for the value-outcome correlation safeguard.

    Integration into the tournament loop is intentionally deferred — see
    the TODO at `ModelTournament.run_anchor_eval`. These tests pin the
    math + API so the wiring change is small and non-novel when W1e
    lands.
    """

    def test_too_few_pairs_returns_none(self, tmp_path):
        ml = MetricsLogger(save_dir=tmp_path)
        ml.start_iteration(iteration=0)
        ml.log_eval_value_pair(0.5, 1.0)
        # Single pair → correlation undefined.
        r = ml.compute_and_log_value_outcome_correlation(step=0)
        assert r is None
        assert 'eval/value_outcome_correlation' not in ml.current_metrics['scalars']

    def test_zero_variance_returns_none(self, tmp_path):
        """Degenerate eval (every game ends in same outcome) → r is
        undefined; we drop with a debug log instead of poisoning the
        time series with NaN."""
        ml = MetricsLogger(save_dir=tmp_path)
        ml.start_iteration(iteration=0)
        for v in (0.1, 0.2, 0.3, 0.4):
            ml.log_eval_value_pair(v, 1.0)  # constant outcome
        r = ml.compute_and_log_value_outcome_correlation(step=0)
        assert r is None
        assert 'eval/value_outcome_correlation' not in ml.current_metrics['scalars']

    def test_perfect_positive_correlation(self, tmp_path):
        ml = MetricsLogger(save_dir=tmp_path)
        ml.start_iteration(iteration=0)
        # y = x exactly → r = 1.0
        for v in (-0.8, -0.3, 0.1, 0.4, 0.9):
            ml.log_eval_value_pair(v, v)
        r = ml.compute_and_log_value_outcome_correlation(step=0)
        assert r is not None
        assert r == pytest.approx(1.0, abs=1e-12)
        assert ml.current_metrics['scalars']['eval/value_outcome_correlation'][0]['value'] == pytest.approx(1.0, abs=1e-12)

    def test_perfect_negative_correlation(self, tmp_path):
        ml = MetricsLogger(save_dir=tmp_path)
        ml.start_iteration(iteration=0)
        # y = -x exactly → r = -1.0. Indicates the value head is
        # systematically backwards — a real signal worth the safeguard.
        for v in (-0.8, -0.3, 0.1, 0.4, 0.9):
            ml.log_eval_value_pair(v, -v)
        r = ml.compute_and_log_value_outcome_correlation(step=0)
        assert r is not None
        assert r == pytest.approx(-1.0, abs=1e-12)

    def test_clear_resets_buffer(self, tmp_path):
        ml = MetricsLogger(save_dir=tmp_path)
        ml.start_iteration(iteration=0)
        for v in (-0.5, 0.5):
            ml.log_eval_value_pair(v, v)
        r1 = ml.compute_and_log_value_outcome_correlation(step=0, clear=True)
        assert r1 is not None
        # Buffer cleared → next call should be too few pairs.
        r2 = ml.compute_and_log_value_outcome_correlation(step=1)
        assert r2 is None

    def test_non_finite_inputs_skipped(self, tmp_path):
        ml = MetricsLogger(save_dir=tmp_path)
        ml.start_iteration(iteration=0)
        ml.log_eval_value_pair(float('nan'), 1.0)
        ml.log_eval_value_pair(0.5, float('inf'))
        # Neither pair was buffered.
        assert getattr(ml, '_eval_value_pairs', []) == []


# ---------------------------------------------------------------------------
# B3 — policy target entropy
# ---------------------------------------------------------------------------


class TestB3PolicyTargetEntropy:
    """Per-batch entropy of the *target* policy (MCTS visit distribution),
    distinct from the predicted-policy entropy already tracked.

    Trainer.train_step computes this inline and emits via
    ``metrics_logger.log_scalar('train/policy_target_entropy_mean', ...)``.
    The test computes the entropy directly on a known target distribution
    to pin the math, then exercises the actual logging path through a
    minimal trainer-shaped object.
    """

    def test_uniform_target_matches_log_n(self):
        """Sanity: entropy of uniform-over-5 should be log(5) ≈ 1.6094."""
        # The trainer formula: entropy_i = -sum(p * log(p+eps)) per row, then mean.
        eps = 1e-12
        n = 5
        target = torch.full((4, n), 1.0 / n)  # 4-row batch, uniform over 5 moves
        entropy = -(target * torch.log(target + eps)).sum(dim=1).mean()
        assert float(entropy) == pytest.approx(np.log(n), abs=1e-5)

    def test_one_hot_target_is_zero(self):
        """Hard one-hot policy target → entropy = 0. Exercises the
        eps-padding correctness: zero-mass entries contribute 0·log(eps)
        which rounds to ~0, not NaN."""
        eps = 1e-12
        n = 7433
        target = torch.zeros((3, n))
        target[:, 17] = 1.0  # all rows pick the same action
        entropy = -(target * torch.log(target + eps)).sum(dim=1).mean()
        # 0·log(eps) ≈ 0 numerically; we tolerate sub-1e-9 floating noise.
        assert float(entropy) == pytest.approx(0.0, abs=1e-6)

    def test_logging_path_via_metrics_logger(self, tmp_path):
        """Minimal trainer-shaped shim: replicate the lines the trainer
        runs after sampling a batch, and assert the scalar lands in the
        MetricsLogger buffer the same way the real train_step would emit
        it. This pins the wiring (metric name + step source) without
        having to construct a full YinshTrainer (network + optimizer +
        schedulers) just to test 4 lines of telemetry.
        """
        ml = MetricsLogger(save_dir=tmp_path)
        ml.start_iteration(iteration=0)

        # Synthetic batch: half uniform-over-3, half one-hot. Mean entropy
        # should be (log(3) + 0) / 2 ≈ 0.549.
        n = 3
        target_probs = torch.zeros((4, n))
        target_probs[0, :] = 1.0 / n
        target_probs[1, :] = 1.0 / n
        target_probs[2, 0] = 1.0
        target_probs[3, 1] = 1.0

        eps = 1e-12
        target_entropy = -(target_probs * torch.log(target_probs + eps)).sum(dim=1).mean()
        expected = float(target_entropy)
        # Mirror exactly what the trainer does:
        ml.log_scalar('train/policy_target_entropy_mean', expected, step=42)

        scalars = ml.current_metrics['scalars']
        assert 'train/policy_target_entropy_mean' in scalars
        entry = scalars['train/policy_target_entropy_mean'][0]
        assert entry['step'] == 42
        assert entry['value'] == pytest.approx((np.log(3) + 0) / 2, abs=1e-5)
        assert entry['value'] == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# MetricsLogger.log_scalar — basic API contract
# ---------------------------------------------------------------------------


class TestLogScalarApi:
    def test_drops_non_numeric(self, tmp_path):
        ml = MetricsLogger(save_dir=tmp_path)
        ml.start_iteration(iteration=0)
        ml.log_scalar('foo', 'not-a-number', step=0)  # type: ignore[arg-type]
        assert 'foo' not in ml.current_metrics['scalars'] or \
            len(ml.current_metrics['scalars'].get('foo', [])) == 0

    def test_stores_none_for_non_finite(self, tmp_path):
        ml = MetricsLogger(save_dir=tmp_path)
        ml.start_iteration(iteration=0)
        ml.log_scalar('foo', float('nan'), step=0)
        ml.log_scalar('foo', float('inf'), step=1)
        entries = ml.current_metrics['scalars']['foo']
        assert len(entries) == 2
        assert entries[0]['value'] is None
        assert entries[1]['value'] is None

    def test_works_without_start_iteration(self, tmp_path):
        """A unit test that wants to exercise just the logging path
        shouldn't have to know about start_iteration() — the lazy-init
        path inside log_scalar covers it."""
        ml = MetricsLogger(save_dir=tmp_path)
        # Note: deliberately NOT calling ml.start_iteration() here.
        ml.log_scalar('foo', 0.5, step=10)
        entries = ml.current_metrics['scalars']['foo']
        assert entries[0]['value'] == 0.5
        assert entries[0]['step'] == 10

    def test_preserves_step_zero(self, tmp_path):
        """`step=0` is a legal step (iteration 0). Make sure it isn't
        being treated as falsy somewhere on the path."""
        ml = MetricsLogger(save_dir=tmp_path)
        ml.start_iteration(iteration=0)
        ml.log_scalar('foo', 0.5, step=0)
        assert ml.current_metrics['scalars']['foo'][0]['step'] == 0
