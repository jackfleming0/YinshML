"""Shared D2 symmetric-weight regularizer (E16).

The YINSH board has D2 (Klein 4-group) symmetry, so the network's policy and
value outputs *should* be invariant under the 4 board symmetries. They aren't —
E11 measured the value head varying 2.8x across orientations by move 8 — because
D2 *data* augmentation symmetrizes the data but does not constrain the *weights*
to the symmetric subspace. This regularizer adds that constraint: every so often,
forward the net on all 4 D2 transforms and penalize divergence from the
symmetric mean (masked policy-KL + value-asymmetry MSE).

This module is the single source of truth, imported by BOTH training paths:
  - self-play:   yinsh_ml/training/trainer.py        (soft MCTS targets)
  - supervised:  scripts/run_supervised_pretraining.py (hard/soft expert targets)

so the whole pipeline enforces weight symmetry, not just the self-play half — the
supervised pretrain is where most of the representation (and thus most of the
asymmetry) is learned.
"""

from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F


def build_full_policy_permutation(augmenter, encoder, transform_id):
    """Build the full ``[total_moves]`` policy-index permutation for one D2
    transform: ``perm[idx_of(M)] = idx_of(T(M))`` for every move M.

    State-independent — ``move_to_index`` is a pure function of the move, so the
    permutation of the whole 7433-slot move space is fixed given the board
    geometry. (The augmenter's per-state ``_build_index_permutation`` is the same
    idea restricted to a position's legal moves, and is the validated reference
    this is checked against in the tests.) Enumerates every move from the board
    positions + the REMOVE_MARKERS line table; ``idx_to_move_map`` is legacy/
    unused so we don't rely on it. Asserts a full bijection.
    """
    from ..game.types import Move, MoveType
    from ..game.constants import Position
    from ..utils.encoding import _REMOVE_MARKERS_LINES

    coord_map = augmenter._coord_maps[transform_id]  # noqa: SLF001
    n = encoder.total_moves
    perm = np.full(n, -1, dtype=np.int64)
    positions = [Position.from_string(p) for p in encoder.position_to_index]

    def _set(move):
        t_move = augmenter._transform_move(move, coord_map)  # noqa: SLF001
        if t_move is None:
            raise ValueError(
                f"D2 transform {transform_id} sent {move} off-board — the valid "
                "set is supposed to be closed under D2"
            )
        perm[encoder.move_to_index(move)] = encoder.move_to_index(t_move)

    for p in positions:
        _set(Move(type=MoveType.PLACE_RING, player=None, source=p))
        _set(Move(type=MoveType.REMOVE_RING, player=None, source=p))
    for src in positions:
        for dst in positions:
            if src != dst:
                _set(Move(type=MoveType.MOVE_RING, player=None, source=src, destination=dst))
    for line in _REMOVE_MARKERS_LINES:
        _set(Move(type=MoveType.REMOVE_MARKERS, player=None, markers=tuple(line)))

    if (perm < 0).any() or len(set(perm.tolist())) != n:
        raise ValueError(
            f"symmetric policy permutation for transform {transform_id} is not a "
            f"bijection ({len(set(perm.tolist()))} distinct, "
            f"{int((perm < 0).sum())} unfilled, of {n})"
        )
    return perm


def build_reg_tensors(encoder, device):
    """Precompute, once, the per-transform tensors the regularizer needs.

    For each non-identity D2 transform returns ``(cell_src, perm, inv_perm)``:
      - ``cell_src``: ``[121]`` gather index that geometrically transforms an
        input tensor (``transformed[:, :, f] = state[:, :, cell_src[f]]`` — all
        channels; spatially-uniform channels are unchanged). Equivalent to the
        augmenter's ``_transform_state``, vectorized + differentiable.
      - ``perm`` / ``inv_perm``: forward and inverse full-move-space permutations.
        ``inv_perm`` is the gather form used to map a policy on the transformed
        action space back to the original: ``out[:, j] = src[:, inv_perm[j]]``
        (gather rather than ``index_copy_``, which is unimplemented on MPS).
    """
    from .augmentation import YinshSymmetryAugmenter
    from ..utils.encoding import StateEncoder

    augmenter = YinshSymmetryAugmenter(include_reflections=True, state_encoder=StateEncoder())
    out = []
    for tid in (1, 2, 3):
        coord_map = augmenter._coord_maps[tid]  # noqa: SLF001
        cell_src = list(range(121))
        for (orow, ocol), (nrow, ncol) in coord_map.items():
            cell_src[nrow * 11 + ncol] = orow * 11 + ocol
        cell_src_t = torch.tensor(cell_src, dtype=torch.long, device=device)
        perm = build_full_policy_permutation(augmenter, encoder, tid)
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(len(perm))
        perm_t = torch.tensor(perm, dtype=torch.long, device=device)
        inv_perm_t = torch.tensor(inv_perm, dtype=torch.long, device=device)
        out.append((cell_src_t, perm_t, inv_perm_t))
    return out


def valid_move_mask(encoder, states):
    """Decode a batch of encoded states into a ``[B, total_moves]`` float mask of
    legal moves, on the same device as ``states``.

    For training paths whose targets are hard move indices (supervised expert
    data), where there's no MCTS visit distribution whose ``> 0`` support could
    serve as the valid-move mask. Decoding is CPU-bound; only call it on the
    regularized batches (every K steps), not every step.
    """
    arr = states.detach().cpu().numpy().astype(np.float32)
    n = encoder.total_moves
    out = np.zeros((arr.shape[0], n), dtype=np.float32)
    for i in range(arr.shape[0]):
        gs = encoder.decode_state(arr[i])
        for mv in gs.get_valid_moves():
            j = encoder.move_to_index(mv)
            if 0 <= j < n:
                out[i, j] = 1.0
    return torch.from_numpy(out).to(states.device)


def symmetric_reg_term(network, states, pred_logits, pred_values, valid_mask,
                       reg_tensors, *, value_weight, weight, autocast=None):
    """Core E16 term for one batch. The identity forward is reused from the
    caller's main pass (``pred_logits`` / ``pred_values``); the 3 non-identity
    transforms are forwarded here.

    Policy KL is restricted to ``valid_mask`` (the moves with training signal) —
    the full 7433-slot softmax is ~99% never-supervised invalid-move logits whose
    asymmetry is meaningless and would swamp the term (~100x). Each policy is
    masked + renormalized so all 4 are distributions over the same valid set.

    ``autocast`` is an optional zero-arg callable returning a context manager
    (e.g. the trainer's ``self._autocast``); the KL/value math always runs in
    fp32 regardless. Returns ``(loss, diagnostics)``; ``loss`` already carries
    the outer ``weight``.
    """
    eps = 1e-9
    b = states.shape[0]
    ac = autocast if autocast is not None else nullcontext

    def _masked_dist(logits):
        p = F.softmax(logits.float(), dim=1) * valid_mask
        return p / (p.sum(dim=1, keepdim=True) + eps)

    flat = states.reshape(b, states.shape[1], -1)
    policies = [_masked_dist(pred_logits)]
    values = [pred_values.float().reshape(b, -1)]

    for cell_src, _perm, inv_perm in reg_tensors:
        state_t = flat[:, :, cell_src].reshape(states.shape)
        with ac():
            logits_t, value_t = network(state_t)
        logits_orig = logits_t.index_select(1, inv_perm)
        policies.append(_masked_dist(logits_orig))
        values.append(value_t.float().reshape(b, -1))

    policy_sym = torch.stack(policies, dim=0).mean(dim=0)
    kl = (policy_sym * (torch.log(policy_sym + eps) - torch.log(policies[0] + eps))).sum(dim=1).mean()
    vstack = torch.stack(values, dim=0)
    value_asym = ((vstack - vstack.mean(dim=0, keepdim=True)) ** 2).mean()

    loss = weight * (kl + value_weight * value_asym)
    return loss, {"sym_kl": float(kl.item()), "sym_value_asym": float(value_asym.item())}
