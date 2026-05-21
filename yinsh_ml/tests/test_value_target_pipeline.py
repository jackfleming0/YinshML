"""Regression tests for the value-target bootstrap bug (T1.2).

Pre-fix bug
-----------
``YinshTrainer._search_consistency_step`` was building its value-head
distillation target from ``mcts.last_root_value`` — i.e., the *network's own*
prediction at the search root. That made the value head distill from its own
bootstrapped predictions, with no external grounding signal. In a long
training run the value head trends toward whatever value the freshly-trained
network is currently outputting at the root, regardless of the actual game
outcome — a pure self-bootstrap loop.

Post-fix
--------
The replay buffer already stores per-position terminal outcomes
(``self.experience.values[idx]``, written by ``_run_game_loop`` at
``self_play.py:1751-1754``: ``[outcome_white if p == WHITE else
-outcome_white for p in players]``). That is the right target — the actual
game outcome from the leaf player's POV. The fix replaces ``v_long`` with
``float(self.experience.values[idx])``.

These tests pin:

  1. The materialized ``v_target`` tensor inside ``_search_consistency_step``
     equals the per-position terminal outcomes from ``self.experience.values``
     — and *not* the MCTS root value.
  2. The bug's signature: if MCTS root value differs sharply from the
     stored terminal outcome, the trained target tracks the outcome.
"""

import contextlib
from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from yinsh_ml.training.trainer import GameExperience, YinshTrainer
from yinsh_ml.utils.encoding import StateEncoder


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


def _make_experience_with_known_outcomes(values, encoder=None):
    """Build a real ``GameExperience`` populated with ``len(values)`` positions
    whose terminal outcomes are exactly ``values``.

    State tensors are zero blobs — the SC step decodes them via the encoder,
    which we patch in the test to return a non-terminal ``MagicMock`` so the
    state never short-circuits the search loop.
    """
    if encoder is None:
        encoder = StateEncoder()
    n = len(values)
    states = [np.zeros((6, 11, 11), dtype=np.float32) for _ in range(n)]
    policies = [
        np.full(encoder.total_moves, 1.0 / encoder.total_moves, dtype=np.float32)
        for _ in range(n)
    ]
    exp = GameExperience(max_size=max(n * 2, 64))
    # `add_game_experience` stores values verbatim into self.values, so by
    # passing terminal outcomes here we get exactly the per-position targets
    # the SC step should read.
    exp.add_game_experience(states, policies, list(values))
    return exp


def _stub_trainer_for_sc_step(experience, mcts_root_value, encoder=None):
    """Build the *minimum* trainer-shaped object that ``_search_consistency_step``
    can run end-to-end without instantiating a real network or optimizers.

    The strategy:
      * Use a real ``GameExperience`` (so ``self.experience.values[idx]``
        actually returns the buffered terminal outcome).
      * Stub ``self.network.network`` as a tiny ``torch.nn.Module`` whose
        forward returns shape-correct logits/values so backward succeeds.
      * Stub ``state_encoder.decode_state`` to return a non-terminal
        ``MagicMock`` (skipping the terminal-skip branch and the real game
        engine).
      * Stub the MCTS via ``_build_consistency_mcts`` so ``search_batch``
        returns a uniform pi and sets ``last_root_value = mcts_root_value``.
        If the bug regressed, the SC step would feed ``mcts_root_value``
        into the value head — making it trivially distinguishable from the
        per-position terminal outcomes in ``experience.values``.
    """
    if encoder is None:
        encoder = StateEncoder()

    # Tiny network: ignores the input shape, just returns shape-correct
    # logits and value heads. Must be a real Module (not MagicMock) so
    # autograd/backward work.
    class _TinyNet(torch.nn.Module):
        def __init__(self, total_moves):
            super().__init__()
            self.total_moves = total_moves
            # One trainable parameter so backward/step are well-defined.
            self.bias = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            batch = x.shape[0]
            logits = self.bias.expand(batch, self.total_moves)
            values = self.bias.expand(batch, 1)
            return logits, values

    tiny = _TinyNet(encoder.total_moves)

    network_wrapper = MagicMock()
    network_wrapper.network = tiny
    network_wrapper.state_encoder = encoder

    # Mock MCTS: search_batch returns valid uniform pi, sets last_root_value.
    mcts_stub = MagicMock()
    mcts_stub.network = network_wrapper
    mcts_stub.state_encoder = encoder
    mcts_stub.last_root_value = float(mcts_root_value)

    def _search_batch(_state, move_number=0, batch_size=32):
        # Uniform — passes the `pi.sum() > 0` finite/non-degenerate guard.
        return np.full(encoder.total_moves, 1.0 / encoder.total_moves,
                       dtype=np.float32)

    mcts_stub.search_batch = MagicMock(side_effect=_search_batch)
    mcts_stub.reset_tree = MagicMock()

    trainer = MagicMock()
    trainer.enable_search_consistency = True
    trainer.experience = experience
    trainer.current_iteration = 100  # well past warmup
    trainer.search_consistency_warmup_iters = 3
    trainer.search_consistency_batch_size = experience.size()  # use all positions
    trainer.search_consistency_long_sims = 4
    trainer.search_consistency_weight = 1.0
    trainer.search_consistency_value_weight = 1.0
    trainer.network = network_wrapper
    trainer.state_encoder = encoder
    trainer.device = torch.device('cpu')
    trainer._sc_mcts = mcts_stub
    trainer._build_consistency_mcts = MagicMock(return_value=mcts_stub)

    # Force decode_state to return a non-terminal stand-in.
    non_terminal_state = MagicMock()
    non_terminal_state.is_terminal = MagicMock(return_value=False)
    encoder_proxy = MagicMock()
    encoder_proxy.decode_state = MagicMock(return_value=non_terminal_state)
    trainer.state_encoder = encoder_proxy
    network_wrapper.state_encoder = encoder_proxy

    # Disable autocast (CPU stub doesn't need it) and provide null context.
    trainer._autocast = MagicMock(return_value=contextlib.nullcontext())

    # Real torch optimizers over the tiny network's single param.
    trainer.policy_optimizer = torch.optim.SGD(tiny.parameters(), lr=1e-3)
    trainer.value_optimizer = torch.optim.SGD(tiny.parameters(), lr=1e-3)

    trainer.ema = None
    trainer._sc_loss_history = []
    trainer.logger = MagicMock()

    return trainer


# --------------------------------------------------------------------------- #
# Core regression: v_target equals buffered terminal outcomes                  #
# --------------------------------------------------------------------------- #


def test_value_target_equals_buffered_terminal_outcome_not_root_value():
    """The smoking-gun test for T1.2.

    Build 5 positions with known per-position terminal outcomes
    [+1, -1, +1, -1, +1]. Stub MCTS so its ``last_root_value`` is 0.0 for
    every position — sharply different from any of the buffered outcomes.

    Pre-fix: every entry in ``v_target`` would be 0.0 (the root value).
    Post-fix: ``v_target`` matches the buffered outcomes verbatim.
    """
    encoder = StateEncoder()
    expected_outcomes = [1.0, -1.0, 1.0, -1.0, 1.0]
    exp = _make_experience_with_known_outcomes(expected_outcomes, encoder)
    trainer = _stub_trainer_for_sc_step(exp, mcts_root_value=0.0, encoder=encoder)

    # Force a deterministic sample order so we can compare element-wise.
    # `np.random.choice(n, size=n, replace=False)` is a permutation; pin it.
    captured = {}

    real_mse = F.mse_loss

    def _capture_mse(pred, target, *args, **kwargs):
        # The SC step computes value_consistency_loss = F.mse_loss(pred, v_target).
        # `pred` is the network's value head output (shape [N]); `target` is the
        # tensor we care about. There's also a policy CE step in the same method
        # that uses log_softmax — that path doesn't call mse_loss, so this
        # capture is unambiguous for the value target.
        captured['v_target'] = target.detach().cpu().numpy().copy()
        return real_mse(pred, target, *args, **kwargs)

    method = YinshTrainer._search_consistency_step.__get__(trainer)
    with patch('yinsh_ml.training.trainer.F.mse_loss', side_effect=_capture_mse), \
         patch('numpy.random.choice', return_value=np.arange(len(expected_outcomes))):
        stats = method()

    assert stats is not None and stats.get('samples', 0) == len(expected_outcomes), (
        f"SC step skipped unexpectedly: stats={stats}"
    )
    assert 'v_target' in captured, "Value-loss path was never reached."

    np.testing.assert_allclose(
        captured['v_target'],
        np.asarray(expected_outcomes, dtype=np.float32),
        rtol=0, atol=1e-6,
        err_msg=(
            "v_target must equal the per-position terminal outcomes from "
            "self.experience.values — NOT the MCTS root value (which was "
            "stubbed to 0.0). If this assertion fires with all-zeros, the "
            "T1.2 self-bootstrap bug has regressed."
        ),
    )


def test_value_target_does_not_track_root_value_when_outcome_differs():
    """Belt-and-suspenders form of the smoking-gun test.

    Same shape, but with a non-zero ``last_root_value`` (0.7) that differs
    from every buffered outcome. The value target must equal the outcomes,
    never the root value.
    """
    encoder = StateEncoder()
    expected_outcomes = [-1.0, 0.5, -0.5, 1.0]
    exp = _make_experience_with_known_outcomes(expected_outcomes, encoder)
    trainer = _stub_trainer_for_sc_step(exp, mcts_root_value=0.7, encoder=encoder)

    captured = {}
    real_mse = F.mse_loss

    def _capture_mse(pred, target, *args, **kwargs):
        captured['v_target'] = target.detach().cpu().numpy().copy()
        return real_mse(pred, target, *args, **kwargs)

    method = YinshTrainer._search_consistency_step.__get__(trainer)
    with patch('yinsh_ml.training.trainer.F.mse_loss', side_effect=_capture_mse), \
         patch('numpy.random.choice', return_value=np.arange(len(expected_outcomes))):
        method()

    np.testing.assert_allclose(
        captured['v_target'],
        np.asarray(expected_outcomes, dtype=np.float32),
        rtol=0, atol=1e-6,
    )
    # And explicitly confirm it isn't the root value (would imply regression).
    assert not np.allclose(captured['v_target'], 0.7), (
        "v_target tracks last_root_value=0.7 — T1.2 self-bootstrap bug "
        "has regressed."
    )


def test_value_target_indexing_aligned_with_sampled_indices():
    """The value target's i-th entry must come from
    ``self.experience.values[idx_i]``, where ``idx_i`` is the i-th sampled
    index — not from positional iteration order. This pins the indexing fix
    so a future refactor that switches to ``enumerate`` over the buffer
    would fail loudly.
    """
    encoder = StateEncoder()
    # Distinct outcomes per position so any index permutation is detectable.
    expected_outcomes = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]
    exp = _make_experience_with_known_outcomes(expected_outcomes, encoder)
    trainer = _stub_trainer_for_sc_step(exp, mcts_root_value=0.0, encoder=encoder)

    # Force a non-identity permutation — reverse order.
    permuted = np.asarray(list(reversed(range(len(expected_outcomes)))), dtype=np.int64)

    captured = {}
    real_mse = F.mse_loss

    def _capture_mse(pred, target, *args, **kwargs):
        captured['v_target'] = target.detach().cpu().numpy().copy()
        return real_mse(pred, target, *args, **kwargs)

    method = YinshTrainer._search_consistency_step.__get__(trainer)
    with patch('yinsh_ml.training.trainer.F.mse_loss', side_effect=_capture_mse), \
         patch('numpy.random.choice', return_value=permuted):
        method()

    expected_permuted = np.asarray(expected_outcomes, dtype=np.float32)[permuted]
    np.testing.assert_allclose(
        captured['v_target'],
        expected_permuted,
        rtol=0, atol=1e-6,
        err_msg=(
            "v_target rows must align with the *sampled* indices, not the "
            "buffer's positional order."
        ),
    )
