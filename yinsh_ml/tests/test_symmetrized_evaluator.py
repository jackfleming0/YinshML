"""Correctness tests for SymmetrizingEvaluator (test-time D2 averaging).

The headline test is **equivariance**: because the D2 group is closed, the set
of four per-transform network evaluations for a position ``s`` is identical to
the set for ``rot(s)`` (rotating the rotated board by the group reproduces the
same four boards). Therefore the *averaged* distribution must satisfy
``Sym(rot(s)) = rot(Sym(s))`` exactly — independent of whether the network
itself has learned any symmetry. A random-initialized network is a perfectly
valid (and stringent) test subject: any index-direction bug in the policy
remapping breaks equivariance immediately.
"""

import numpy as np
import pytest
import torch

from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.types import GamePhase
from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.network.symmetrized_evaluator import SymmetrizingEvaluator
from yinsh_ml.training.augmentation import YinshSymmetryAugmenter


def _main_game_state(seed: int = 0) -> GameState:
    """Play deterministic pseudo-random legal moves until MAIN_GAME."""
    rng = np.random.RandomState(seed)
    gs = GameState()
    for _ in range(400):
        if gs.phase == GamePhase.MAIN_GAME:
            # Advance a couple more plies so there are markers on the board.
            extra = 3
            while extra > 0 and not gs.is_terminal():
                moves = gs.get_valid_moves()
                if not moves:
                    break
                gs.make_move(moves[rng.randint(len(moves))])
                extra -= 1
            return gs
        moves = gs.get_valid_moves()
        if not moves or gs.is_terminal():
            break
        gs.make_move(moves[rng.randint(len(moves))])
    raise RuntimeError("could not reach MAIN_GAME")


@pytest.fixture(scope="module")
def net():
    # Random-init, basic 6-ch, CPU. Equivariance holds for any fixed network.
    torch.manual_seed(1234)
    return NetworkWrapper(device="cpu")


@pytest.fixture(scope="module")
def evaluator(net):
    return SymmetrizingEvaluator(net, num_transforms=4)


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def test_output_is_a_valid_distribution(evaluator):
    gs = _main_game_state(seed=1)
    logits, values = evaluator.evaluate_batch([gs])
    p = _softmax(logits[0].numpy())
    assert p.shape[0] == evaluator.encoder.total_moves
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(), 1.0, atol=1e-5)
    # Essentially all mass sits on legal-move indices.
    valid_idx = [evaluator.encoder.move_to_index(m) for m in gs.get_valid_moves()]
    assert p[valid_idx].sum() > 0.999
    assert torch.isfinite(values).all()
    assert -1.0 <= float(values[0]) <= 1.0


def test_value_invariant_under_rotation(evaluator):
    """The averaged value of a position and its 180° rotation must match: both
    average the value over the same four boards."""
    enc = evaluator.encoder
    gs = _main_game_state(seed=2)
    gs_rot = enc.decode_state(evaluator.aug._transform_state(enc.encode_state(gs), 1))
    _, v = evaluator.evaluate_batch([gs])
    _, v_rot = evaluator.evaluate_batch([gs_rot])
    np.testing.assert_allclose(float(v[0]), float(v_rot[0]), atol=1e-5)


def test_policy_equivariant_under_rotation(evaluator):
    """Sym(rot(s)) == rot(Sym(s)): the symmetrized policy of the rotated
    position equals the 180°-image of the symmetrized policy of the original.

    Checked move-by-move on the original position's legal moves, mapping each
    through the 180° transform to its index in the rotated frame.
    """
    enc = evaluator.encoder
    aug = YinshSymmetryAugmenter(include_reflections=True, state_encoder=enc)
    cmap180 = aug._coord_maps[1]

    gs = _main_game_state(seed=3)
    gs_rot = enc.decode_state(aug._transform_state(enc.encode_state(gs), 1))

    p = _softmax(evaluator.evaluate_batch([gs])[0][0].numpy())
    p_rot = _softmax(evaluator.evaluate_batch([gs_rot])[0][0].numpy())

    checked = 0
    for move in gs.get_valid_moves():
        old_idx = enc.move_to_index(move)
        rimg = aug._transform_move(move, cmap180)
        if rimg is None:
            continue
        new_idx = enc.move_to_index(rimg)
        # p assigns mass to `move`; the rotated position should assign the same
        # mass to the rotated move.
        np.testing.assert_allclose(p[old_idx], p_rot[new_idx], atol=1e-5)
        checked += 1
    assert checked >= 3, "too few legal moves mapped to make the test meaningful"


def test_batch_matches_singletons(evaluator):
    """Evaluating a batch must give the same per-state result as evaluating
    each state alone (no cross-talk in the i*T+t packing)."""
    states = [_main_game_state(seed=s) for s in (4, 5, 6)]
    batch_logits, batch_values = evaluator.evaluate_batch(states)
    for i, gs in enumerate(states):
        sl, sv = evaluator.evaluate_batch([gs])
        np.testing.assert_allclose(batch_logits[i].numpy(), sl[0].numpy(), atol=1e-5)
        np.testing.assert_allclose(float(batch_values[i]), float(sv[0]), atol=1e-5)
