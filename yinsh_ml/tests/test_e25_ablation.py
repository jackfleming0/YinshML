"""E25 binding-constraint ablation flags — guard against silent regression.

The ablation flags on MCTS (ablate_policy / ablate_value) drive
scripts/e25_ablation_h2h.py. They must (a) be no-ops when off and (b) do exactly
what they claim when on: uniform prior over valid moves / constant-0 leaf value.
"""
import types

import numpy as np
import pytest

from yinsh_ml.training.self_play import MCTS
from yinsh_ml.game.game_state import GameState
from yinsh_ml.utils.encoding import StateEncoder


def _mcts(ablate_policy=False, ablate_value=False):
    # MCTS.__init__ only touches network.state_encoder; a stub avoids loading a net.
    stub_net = types.SimpleNamespace(state_encoder=StateEncoder())
    return MCTS(
        network=stub_net, evaluation_mode="pure_neural",
        ablate_policy=ablate_policy, ablate_value=ablate_value,
    )


def test_ablation_off_is_identity():
    mcts = _mcts()
    state = GameState()  # RING_PLACEMENT start has valid moves
    policy = np.random.default_rng(0).random(mcts.state_encoder.total_moves).astype(np.float32)
    out_policy, out_value = mcts._apply_head_ablation(policy.copy(), 0.42, state)
    assert out_value == 0.42
    np.testing.assert_array_equal(out_policy, policy)


def test_ablate_policy_is_uniform_over_valid_moves():
    mcts = _mcts(ablate_policy=True)
    state = GameState()
    valid = state.get_valid_moves()
    assert valid, "expected valid moves at game start"
    junk = np.random.default_rng(1).random(mcts.state_encoder.total_moves).astype(np.float32)
    out_policy, out_value = mcts._apply_head_ablation(junk, 0.9, state)

    # Value is untouched by policy ablation.
    assert out_value == 0.9

    valid_idx = sorted({mcts.state_encoder.move_to_index(m) for m in valid})
    expected = 1.0 / len(valid)
    # Every valid move carries equal prior...
    for idx in valid_idx:
        assert out_policy[idx] == pytest.approx(expected)
    # ...and the rest are zero.
    mask = np.ones(len(out_policy), dtype=bool)
    mask[valid_idx] = False
    assert np.all(out_policy[mask] == 0.0)


def test_ablate_value_is_constant_zero():
    mcts = _mcts(ablate_value=True)
    state = GameState()
    policy = np.random.default_rng(2).random(mcts.state_encoder.total_moves).astype(np.float32)
    out_policy, out_value = mcts._apply_head_ablation(policy.copy(), 0.77, state)

    # Value zeroed, prior untouched.
    assert out_value == 0.0
    np.testing.assert_array_equal(out_policy, policy)
