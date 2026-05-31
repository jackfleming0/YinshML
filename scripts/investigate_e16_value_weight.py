"""Investigate the right symmetric_reg_value_weight for the E16 regularizer.

E16's loss is  alpha * (kl + w * value_asym).  kl lives on a 7433-way policy
simplex (masked to valid moves); value_asym is the variance of a SCALAR value
in ~[-1,1] across the 4 D2 transforms. Different units => a fixed w doesn't
imply comparable training pressure. This script measures, on real positions:

  1. Raw magnitudes of kl and value_asym, by game phase / move number.
  2. The actual GRADIENT pressure each term exerts on the weights
     (||d kl/d theta|| vs ||d value_asym/d theta||) — the principled basis for w:
     w_equal = ||g_kl|| / ||g_val|| makes the two terms push equally hard.
     Reported for the whole net and trunk-only (the shared body, where the
     symmetry constraint actually has to land).

Run against iter1_ema (classification head, scalar value out, enhanced 15ch) —
a trained-but-asymmetric net, the regime the regularizer operates in.
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F

from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.training.trainer import YinshTrainer
from yinsh_ml.utils.encoding import decode_phase_from_state


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='models/iter1_ema_2026-05-27/iter1_ema.pt')
    p.add_argument('--data', default='expert_games/hvh_full_game_15ch.npz')
    p.add_argument('--n', type=int, default=256, help='positions to sample')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def valid_move_mask(encoder, state_tensor_np):
    """Boolean [total_moves] mask of legal moves for a decoded state. The honest
    symmetry domain for the policy (broader than MCTS visit support)."""
    gs = encoder.decode_state(state_tensor_np)
    m = np.zeros(encoder.total_moves, dtype=np.float32)
    for mv in gs.get_valid_moves():
        i = encoder.move_to_index(mv)
        if 0 <= i < len(m):
            m[i] = 1.0
    return m


def kl_and_value_asym(net, states, masks, reg_tensors, encoder):
    """Return (kl_tensor, value_asym_tensor), each a scalar with grad, computed
    the same way trainer._symmetric_reg_term does but kept separate so we can
    backprop each independently to read its gradient norm."""
    eps = 1e-9
    b = states.shape[0]
    flat = states.reshape(b, states.shape[1], -1)

    def masked_dist(logits):
        p = F.softmax(logits.float(), dim=1) * masks
        return p / (p.sum(dim=1, keepdim=True) + eps)

    logits0, value0 = net(states)
    policies = [masked_dist(logits0)]
    values = [value0.float().reshape(b, -1)]
    for cell_src, perm in reg_tensors:
        st = flat[:, :, cell_src].reshape(states.shape)
        lt, vt = net(st)
        lo = torch.empty_like(lt)
        lo.index_copy_(1, perm, lt)
        policies.append(masked_dist(lo))
        values.append(vt.float().reshape(b, -1))

    policy_sym = torch.stack(policies, 0).mean(0)
    kl = (policy_sym * (torch.log(policy_sym + eps) - torch.log(policies[0] + eps))).sum(1).mean()
    vstack = torch.stack(values, 0)
    value_asym = ((vstack - vstack.mean(0, keepdim=True)) ** 2).mean()
    return kl, value_asym


def grad_norm(net, loss, trunk_only=False):
    net.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    total = 0.0
    for name, prm in net.named_parameters():
        if prm.grad is None:
            continue
        if trunk_only and ('policy_head' in name or 'value_head' in name):
            continue
        total += float(prm.grad.detach().pow(2).sum().item())
    return total ** 0.5


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    print(f"Loading {args.checkpoint}")
    nw = NetworkWrapper(model_path=args.checkpoint, device='cpu', use_enhanced_encoding=True)
    net = nw.network
    enc = nw.state_encoder
    tr = YinshTrainer(network=nw, device='cpu', enable_symmetric_reg=True)
    reg_tensors = tr._build_symmetric_reg_tensors()

    d = np.load(args.data)
    all_states = d['states']
    n_total = all_states.shape[0]
    idx = rng.choice(n_total, size=min(args.n, n_total), replace=False)
    states_np = all_states[idx].astype(np.float32)
    print(f"Sampled {len(idx)} of {n_total} positions from {args.data}\n")

    # Phase + move-number proxy (ring/marker mass) per position.
    phases = [decode_phase_from_state(s) for s in states_np]
    piece_mass = (np.abs(states_np[:, :4, :, :]) > 0).sum(axis=(1, 2, 3))

    masks_np = np.stack([valid_move_mask(enc, s) for s in states_np])
    states = torch.from_numpy(states_np)
    masks = torch.from_numpy(masks_np)

    net.train()

    # ---- Per-position raw magnitudes (batched by phase bucket) ----
    print("=" * 74)
    print("RAW MAGNITUDES by game phase (mean over positions in bucket)")
    print("=" * 74)
    print(f"{'phase':>16s} | {'n':>4s} | {'kl (policy)':>12s} | {'value_asym':>12s} | {'ratio kl/va':>11s}")
    print("-" * 74)
    order = ['RING_PLACEMENT', 'MAIN_GAME', 'ROW_COMPLETION', 'RING_REMOVAL', 'GAME_OVER']
    seen = set(phases)
    for ph in order + sorted(seen - set(order)):
        sel = [i for i, p in enumerate(phases) if p == ph]
        if not sel:
            continue
        with torch.no_grad():
            sl = torch.tensor(sel)
            kl, va = kl_and_value_asym(net, states[sl], masks[sl], reg_tensors, enc)
        r = kl.item() / va.item() if va.item() > 0 else float('inf')
        print(f"{ph:>16s} | {len(sel):>4d} | {kl.item():>12.5f} | {va.item():>12.6f} | {r:>11.1f}")

    # ---- Whole-batch magnitudes + gradient pressure ----
    print("\n" + "=" * 74)
    print("GRADIENT PRESSURE on the weights (whole batch)")
    print("=" * 74)
    kl, va = kl_and_value_asym(net, states, masks, reg_tensors, enc)
    print(f"raw:  kl={kl.item():.5f}   value_asym={va.item():.6f}   "
          f"(raw ratio kl/va = {kl.item()/va.item():.1f})")

    for scope, trunk in (("whole net", False), ("trunk only", True)):
        g_kl = grad_norm(net, kl, trunk_only=trunk)
        g_va = grad_norm(net, va, trunk_only=trunk)
        w_equal = g_kl / g_va if g_va > 0 else float('inf')
        print(f"\n[{scope}]")
        print(f"   ||grad kl||        = {g_kl:.5f}")
        print(f"   ||grad value_asym|| = {g_va:.5f}")
        print(f"   w for EQUAL gradient pressure  (||g_kl||/||g_va||) = {w_equal:.1f}")
        for label, factor in (("value 2x policy (E11: value is the bigger problem)", 2.0),
                              ("value 5x policy", 5.0)):
            print(f"   w for {label} = {w_equal*factor:.1f}")
    net.zero_grad(set_to_none=True)


if __name__ == '__main__':
    main()
