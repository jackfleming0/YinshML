"""E16 prototype — symmetric weight regularizer for the training loss.

NOT WIRED INTO TRAINER.PY. This is a standalone proof-of-concept that
demonstrates the loss math + measures the asymmetry penalty on iter1_ema
to verify the term is meaningful before integrating into the training loop.

Loss formulation:
    For each state in batch:
      1. Forward on state s → policy_0, value_0
      2. For each tid in [1, 2, 3]:
           Transform state: s_tid = T_tid(s)
           Forward: policy_tid_raw, value_tid_raw
           Inverse-transform policy_tid_raw back to original action space:
             policy_tid = T_tid(policy_tid_raw)   (D2 transforms are involutions)
      3. Compute the symmetrized policy as the mean over the 4 versions:
           policy_sym = mean(policy_0, policy_1, policy_2, policy_3)
      4. Loss term: KL(policy_sym || policy_0)
           + value regularizer: ((value_0 - mean_value)^2)

The regularizer pushes the network toward producing the SAME prediction
for D2-equivalent inputs, putting continuous pressure on weights to
respect the game's symmetry.

Integration into trainer.py would be ~30 LOC:
- During training, every K batches (say K=10 for compute efficiency),
  compute the symmetric loss term in addition to policy + value losses
- Add it to the total loss with a small weight (e.g., 0.1) so it acts
  as a regularizer, not a primary objective
- 4× forward overhead per regularized batch is the only cost

Run this script to verify the math is right and the asymmetry penalty
is nonzero on iter1_ema (expected: significant value-head asymmetry
penalty on non-trivial positions per the E11 measurements).
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F

from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.types import Move, MoveType
from yinsh_ml.game.constants import Player, Position
from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.utils.encoding import StateEncoder
from yinsh_ml.utils.enhanced_encoding import EnhancedStateEncoder
from yinsh_ml.training.augmentation import YinshSymmetryAugmenter


def symmetric_loss_term(net, state, augmenter, basic_encoder, enhanced_encoder,
                        kl_weight=1.0, value_weight=0.5):
    """Compute the symmetric weight regularizer for a single state.

    Returns:
        total_loss: scalar Tensor (the regularizer)
        diagnostics: dict with per-component values for logging

    To use as a training regularizer:
        loss = policy_loss + value_loss + alpha * symmetric_loss_term(...)
        loss.backward()
    """
    basic = basic_encoder.encode_state(state).astype(np.float32)
    enhanced = enhanced_encoder.encode_state(state).astype(np.float32)

    # Forward pass on identity + all 3 transforms
    policies_orig_space = []
    values = []
    for tid in range(4):
        if tid == 0:
            t_enh = enhanced
            t_basic = basic
        else:
            t_enh = augmenter._transform_state(enhanced, tid)
            t_basic = augmenter._transform_state(basic, tid)

        t_in = torch.from_numpy(np.ascontiguousarray(t_enh)).unsqueeze(0).float()
        # IMPORTANT: do not torch.no_grad() — we need gradients for the loss
        logits_tid, value_tid = net(t_in)
        policy_tid_raw = F.softmax(logits_tid[0], dim=-1)

        if tid == 0:
            policy_orig = policy_tid_raw
        else:
            # Inverse-transform policy via the augmenter's permutation logic.
            # Note: augmenter._transform_policy operates on numpy arrays and
            # is non-differentiable. For a real training integration, we'd
            # build a torch-native version of the permutation (the permutation
            # itself is constant given state geometry).
            #
            # For the prototype, we apply the same permutation as a Tensor
            # index permutation, which IS differentiable.
            policy_orig = _torch_transform_policy(
                policy_tid_raw, t_basic, augmenter, basic_encoder, tid
            )
        policies_orig_space.append(policy_orig)
        values.append(value_tid.squeeze())

    # Symmetrized targets
    policy_sym = torch.stack(policies_orig_space, dim=0).mean(dim=0)
    value_sym = torch.stack(values, dim=0).mean(dim=0)

    # Loss terms
    # KL(policy_orig || policy_sym) — penalize deviation from the symmetric mean
    eps = 1e-9
    kl_loss = -(policy_sym * torch.log((policies_orig_space[0] + eps) / (policy_sym + eps))).sum()

    # Value asymmetry — squared deviation from symmetric mean
    value_asym = sum((v - value_sym) ** 2 for v in values) / 4.0

    total = kl_weight * kl_loss + value_weight * value_asym

    return total, {
        'kl_policy_asymmetry': float(kl_loss.item()),
        'value_asymmetry_mse': float(value_asym.item()),
        'value_range': (float(min(v.item() for v in values)),
                        float(max(v.item() for v in values))),
        'total_loss': float(total.item()),
    }


def _torch_transform_policy(policy_tensor, basic_state, augmenter,
                            basic_encoder, transform_id):
    """Apply the augmenter's policy index permutation to a torch tensor.

    Differentiable version of augmenter._transform_policy. Builds the
    permutation using the augmenter's logic, then applies via index_put_.
    """
    permutation = augmenter._build_index_permutation(basic_state, transform_id)
    out = torch.zeros_like(policy_tensor)
    if not permutation:
        return out
    # Vectorize the permutation
    old_idxs = torch.tensor(list(permutation.keys()), dtype=torch.long)
    new_idxs = torch.tensor(list(permutation.values()), dtype=torch.long)
    out.scatter_add_(0, new_idxs, policy_tensor.index_select(0, old_idxs))
    total = out.sum()
    if total > 1e-9:
        out = out / total
    return out


def main():
    print('Loading iter1_ema...')
    nw = NetworkWrapper(model_path='models/iter1_ema_2026-05-27/iter1_ema.pt',
                        device='cpu', use_enhanced_encoding=True)
    basic_encoder = StateEncoder()
    enhanced_encoder = EnhancedStateEncoder()
    augmenter = YinshSymmetryAugmenter(include_reflections=True, state_encoder=basic_encoder)

    # Build 6 test states matching the E11 set
    def parse_pos(p): return Position(p[0].upper(), int(p[1:]))
    test_states = [GameState()]
    s = GameState()
    moves = [
        ('white', 'F6'), ('black', 'E5'), ('white', 'G6'), ('black', 'D4'),
        ('white', 'H7'), ('black', 'C3'), ('white', 'I5'), ('black', 'B6'),
        ('white', 'J4'), ('black', 'A3'),
    ]
    captured_after = [2, 4, 6, 8, 10]
    for i, (pl, pos) in enumerate(moves):
        s.make_move(Move(type=MoveType.PLACE_RING,
                          player=Player.WHITE if pl == 'white' else Player.BLACK,
                          source=parse_pos(pos)))
        if i + 1 in captured_after:
            test_states.append(copy.deepcopy(s))

    print(f'\nComputing symmetric weight regularizer on {len(test_states)} states:\n')
    print(f'{"State":>18s} | {"kl_policy":>12s} | {"value_asym":>12s} | {"value range":>22s} | {"loss":>10s}')
    print('-' * 90)

    total_loss = 0.0
    for i, st in enumerate(test_states):
        label = 'empty' if i == 0 else f'after move {captured_after[i-1]}'
        loss, diag = symmetric_loss_term(nw.network, st, augmenter,
                                          basic_encoder, enhanced_encoder)
        total_loss += loss.item()
        vmin, vmax = diag['value_range']
        print(f'{label:>18s} | {diag["kl_policy_asymmetry"]:>12.5f} | '
              f'{diag["value_asymmetry_mse"]:>12.5f} | '
              f'[{vmin:>+8.4f}, {vmax:>+8.4f}] | '
              f'{diag["total_loss"]:>10.5f}')

    print('-' * 90)
    print(f'{"BATCH TOTAL":>18s} | {total_loss/len(test_states):>53.5f} (mean per state)')
    print()
    print('Interpretation:')
    print('  - If value_asym ~0 across all states: weights are value-symmetric (good)')
    print('  - If value_asym > 0.001: real asymmetry, regularizer will pull weights toward symmetric subspace')
    print('  - The kl_policy term should be small but consistently nonzero (matches E11 result)')
    print()
    print('Integration sketch for trainer.py (NOT YET IMPLEMENTED):')
    print('  Every K=10 batches, for each sample in batch:')
    print('    sym_loss = symmetric_loss_term(...)')
    print('  total_loss = policy_loss + value_loss + 0.1 * sym_loss.mean()')
    print('  total_loss.backward()')


if __name__ == '__main__':
    main()
