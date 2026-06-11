"""Test-time D2 symmetry averaging at the MCTS ``evaluator=`` seam.

Background
----------
The E26 symmetry *training* run (`project_symmetry_run_outcome`) lost ~20-25%
H2H to the champion, and the post-mortem conclusion was: "symmetry is an
inference-time problem (test-time D2 averaging), not a training one." That
test was never actually run. This is it.

What it does
------------
For each MCTS leaf state, evaluate the network on all ``num_transforms`` D2
images of the board (identity, 180° rotation, and — at 4 transforms — the two
diagonal reflections), map each output policy back into the *original* board
frame, and average in **probability** space. The averaged distribution is
returned as ``log(avg_prob)`` in the "logits" slot, so the ``torch.softmax``
that MCTS applies downstream (`self_play.py` leaf-eval) recovers the averaged
distribution exactly (softmax(log p) = p for normalized p). Values are averaged
directly — the four transforms are pure geometric board symmetries that never
swap the side to move, so the value target is invariant and averaging only
denoises it.

This plugs into the exact same ``evaluator=`` seam the process-based inference
server uses, so MCTS itself needs zero changes: pass
``MCTS(network=net, evaluator=SymmetrizingEvaluator(net), ...)``.

Cost: ``num_transforms``× the GPU forward per leaf (one batched call of
``N * T`` encoded states), plus one CPU ``decode_state`` per leaf to build the
move-index permutations. MCTS is GPU-forward bound, so the dominant added cost
is the T× forward. Permutation building is not cached across calls (a leaf is
rarely re-evaluated within one search); a Zobrist-keyed cache is the obvious
optimization if this ever ships into self-play rather than eval.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from yinsh_ml.training.augmentation import YinshSymmetryAugmenter


def _softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max()
    e = np.exp(logits - m)
    return e / e.sum()


class SymmetrizingEvaluator:
    """Drop-in ``evaluator`` that D2-symmetrizes the network's leaf evaluations.

    Args:
        network: a ``NetworkWrapper`` exposing ``predict_batch_encoded`` and
            ``state_encoder``.
        num_transforms: 4 (full D2: identity + 180° + 2 reflections) or 2
            (C2: identity + 180° only).
        autocast_dtype: optional ``torch.bfloat16`` / ``torch.float16`` passed
            through to ``predict_batch_encoded``.
    """

    def __init__(self, network, num_transforms: int = 4, autocast_dtype=None):
        if num_transforms not in (2, 4):
            raise ValueError("num_transforms must be 2 (C2) or 4 (D2)")
        self.network = network
        self.encoder = network.state_encoder
        self.num_transforms = num_transforms
        self.autocast_dtype = autocast_dtype
        # The augmenter owns the precomputed coord maps and the move-transform
        # logic; we reuse its internals rather than reimplementing the geometry.
        self.aug = YinshSymmetryAugmenter(
            include_reflections=(num_transforms == 4),
            state_encoder=self.encoder,
        )

    # ------------------------------------------------------------------ #
    def _perms_for_state(self, enc0: np.ndarray) -> List[Dict[int, int]]:
        """Decode the state once; return one ``{old_idx -> new_idx}`` map per
        transform.

        ``old_idx`` is a valid move's policy index in the *original* frame;
        ``new_idx`` is the index of that move's geometric image in the
        *t-transformed* frame. To gather a t-frame policy back into the
        original frame we do ``avg[old_idx] += probs_t[new_idx]`` — i.e. read
        this map backwards relative to how augmentation applies it. (All four
        D2 transforms are involutions, but we never rely on that here: the map
        is built by forward-encoding each move and its image, exactly like
        ``YinshSymmetryAugmenter._build_index_permutation``, just decoding once
        for all transforms.)
        """
        gs = self.encoder.decode_state(enc0)
        base: List[Tuple[object, int]] = []
        for move in gs.get_valid_moves():
            try:
                base.append((move, self.encoder.move_to_index(move)))
            except Exception:
                continue

        perms: List[Dict[int, int]] = []
        for t in range(self.num_transforms):
            cmap = self.aug._coord_maps[t]
            perm: Dict[int, int] = {}
            for move, old_idx in base:
                timg = self.aug._transform_move(move, cmap)
                if timg is None:
                    continue
                try:
                    new_idx = self.encoder.move_to_index(timg)
                except Exception:
                    continue
                perm[old_idx] = new_idx
            perms.append(perm)
        return perms

    # ------------------------------------------------------------------ #
    def evaluate_batch(self, states) -> Tuple[torch.Tensor, torch.Tensor]:
        """Symmetrized batched leaf evaluation.

        Returns ``(policy_logits, values)`` matching ``predict_batch``'s
        contract: logits ``(N, total_moves)`` (here = ``log`` of the averaged
        probability distribution) and values ``(N,)``.
        """
        N = len(states)
        T = self.num_transforms

        encs0 = [self.encoder.encode_state(s) for s in states]
        perms = [self._perms_for_state(e) for e in encs0]

        # (N*T, C, 11, 11), ordering index = i*T + t.
        big = np.empty((N * T,) + encs0[0].shape, dtype=np.float32)
        for i, e0 in enumerate(encs0):
            for t in range(T):
                big[i * T + t] = self.aug._transform_state(e0, t)

        logits_t, values_t = self.network.predict_batch_encoded(
            big, autocast_dtype=self.autocast_dtype
        )
        logits = logits_t.cpu().numpy()              # (N*T, total)
        values = values_t.cpu().numpy().reshape(-1)  # (N*T,)
        total = logits.shape[1]

        out_logits = np.full((N, total), np.log(1e-12), dtype=np.float32)
        out_values = np.empty(N, dtype=np.float32)

        for i in range(N):
            avg = np.zeros(total, dtype=np.float64)
            vsum = 0.0
            contributed = 0
            for t in range(T):
                perm = perms[i][t]
                if not perm:
                    # Transform unusable for this state (e.g. a ring-placement
                    # position whose moves don't all map). Skip it rather than
                    # mixing a wrong-frame distribution in.
                    continue
                pt = _softmax(logits[i * T + t])
                for old_idx, new_idx in perm.items():
                    avg[old_idx] += pt[new_idx]
                vsum += float(values[i * T + t])
                contributed += 1

            if contributed == 0:
                # Degenerate (no usable transform — shouldn't happen for a real
                # leaf with valid moves). Fall back to the plain identity output.
                out_logits[i] = logits[i * T]
                out_values[i] = values[i * T]
                continue

            s = avg.sum()
            if s > 1e-12:
                avg /= s
            out_logits[i] = np.log(np.maximum(avg, 1e-12)).astype(np.float32)
            out_values[i] = vsum / contributed

        return torch.from_numpy(out_logits), torch.from_numpy(out_values)
