"""Correctness tests for the E16 symmetric-weight regularizer (trainer.py).

The risky part is the precomputed full-move-space (7433-slot) policy
permutation and the spatial cell-gather: both must agree with the *validated*
per-state augmenter code (`_build_index_permutation` / `_transform_state`,
covered by the augmentation tests). These tests pin that agreement plus the
end-to-end regularizer numerics, using a fresh (untrained) network so they're
cheap and checkpoint-free.
"""

import numpy as np
import pytest
import torch

from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.training.trainer import YinshTrainer
from yinsh_ml.training.augmentation import YinshSymmetryAugmenter
from yinsh_ml.utils.encoding import StateEncoder
from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.types import Move, MoveType
from yinsh_ml.game.constants import Player, Position


def _parse(p):
    return Position(p[0].upper(), int(p[1:]))


def _mid_game_state(n_rings):
    """A RING_PLACEMENT/MAIN_GAME state with `n_rings` placed (mixed colours)."""
    seq = [('white', 'F6'), ('black', 'E5'), ('white', 'G6'), ('black', 'D4'),
           ('white', 'H7'), ('black', 'C3'), ('white', 'I5'), ('black', 'B6')]
    s = GameState()
    for pl, pos in seq[:n_rings]:
        s.make_move(Move(
            type=MoveType.PLACE_RING,
            player=Player.WHITE if pl == 'white' else Player.BLACK,
            source=_parse(pos),
        ))
    return s


@pytest.fixture(scope="module")
def trainer():
    # Fresh enhanced-encoding network — no checkpoint needed for permutation
    # geometry (it's encoder-defined, weight-independent).
    nw = NetworkWrapper(model_path=None, device='cpu', use_enhanced_encoding=True)
    return YinshTrainer(network=nw, device='cpu', batch_size=4,
                        enable_symmetric_reg=True, symmetric_reg_every_k_steps=1)


def test_policy_permutations_are_bijections(trainer):
    tensors = trainer._build_symmetric_reg_tensors()
    n = trainer.state_encoder.total_moves
    assert len(tensors) == 3
    for _cell_src, perm, inv_perm in tensors:
        assert sorted(perm.tolist()) == list(range(n))
        # inv_perm must be the gather-form inverse: inv_perm[perm[k]] == k
        assert perm[inv_perm].tolist() == list(range(n))


def test_global_perm_matches_validated_per_state_perm(trainer):
    """The full-space permutation, restricted to a state's legal moves, must
    equal the augmenter's per-state permutation (the validated reference)."""
    tensors = trainer._build_symmetric_reg_tensors()
    basic = StateEncoder()
    aug = YinshSymmetryAugmenter(include_reflections=True, state_encoder=basic)
    checked = 0
    for st in (GameState(), _mid_game_state(2), _mid_game_state(6)):
        st_basic = basic.encode_state(st)
        for tid in (1, 2, 3):
            per_state = aug._build_index_permutation(st_basic, tid)
            perm_global = tensors[tid - 1][1]
            for old, new in per_state.items():
                assert int(perm_global[old].item()) == new
                checked += 1
    assert checked > 0


def test_cell_gather_matches_transform_state(trainer):
    """The spatial input gather must reproduce augmenter._transform_state."""
    tensors = trainer._build_symmetric_reg_tensors()
    enc = trainer.network.state_encoder
    aug = YinshSymmetryAugmenter(include_reflections=True, state_encoder=StateEncoder())
    se = enc.encode_state(_mid_game_state(6)).astype(np.float32)
    t = torch.from_numpy(se).unsqueeze(0)
    flat = t.reshape(1, t.shape[1], -1)
    for tid in (1, 2, 3):
        cell_src = tensors[tid - 1][0]
        mine = flat[:, :, cell_src].reshape(t.shape).numpy()[0]
        assert np.array_equal(mine, aug._transform_state(se, tid))


def test_regularizer_runs_and_backprops(trainer):
    """End-to-end: finite, non-negative loss; gradients reach the weights."""
    enc = trainer.network.state_encoder
    sts = [GameState(), _mid_game_state(4)]

    def policy_for(st):
        t = np.zeros(enc.total_moves, dtype=np.float32)
        for m in st.get_valid_moves():
            i = enc.move_to_index(m)
            if 0 <= i < len(t):
                t[i] = 1.0
        return t / t.sum()

    batch = torch.from_numpy(np.stack([enc.encode_state(s).astype(np.float32) for s in sts]))
    targets = torch.from_numpy(np.stack([policy_for(s) for s in sts]))
    trainer.network.network.train()
    logits, values = trainer.network.network(batch)
    loss, diag = trainer._symmetric_reg_term(batch, logits, values, targets)

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    assert diag["sym_value_asym"] >= 0.0
    loss.backward()
    total_grad = sum(p.grad.abs().sum().item()
                     for p in trainer.network.network.parameters() if p.grad is not None)
    assert total_grad > 0.0
