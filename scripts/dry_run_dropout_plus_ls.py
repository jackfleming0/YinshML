"""Dropout-patched + label smoothing variant.

Same as dry_run_dropout_patch.py but adds label_smoothing=0.1 to the
cross-entropy loss. Tests the hypothesis that the over-concentration in
the pure dropout-patched run (F6 100% modal, white WR 6%) was caused by
unsmoothed CE on one-hot targets — and that label smoothing recovers
both correct sharpness AND playable variety.

Standard AlphaZero-style fix: train on soft targets to keep entropy in
the policy distribution. Label smoothing is the cheap approximation of
that without requiring MCTS visit distributions in the training data.
"""

import argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.game.game_state import GameState
from yinsh_ml.utils.enhanced_encoding import EnhancedStateEncoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='models/supervised_2026-05-27/best_supervised.pt')
    p.add_argument('--data', default='expert_games/hvh_full_game_15ch.npz')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--weight-decay', type=float, default=1e-3)
    p.add_argument('--placement-weight', type=float, default=4.0)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--output', default='models/supervised_2026-05-27/dropout_plus_ls01.pt')
    return p.parse_args()


def main():
    args = parse_args()
    print(f'Loading checkpoint: {args.checkpoint}')
    nw = NetworkWrapper(model_path=args.checkpoint, device='cpu', use_enhanced_encoding=True)
    net = nw.network

    print('Policy head architecture:')
    for i, mod in enumerate(net.policy_head):
        print(f'  [{i}] {mod.__class__.__name__}', end='')
        if isinstance(mod, torch.nn.Dropout):
            print(f'(p={mod.p})')
        else:
            print()
    print(f'Label smoothing: ε = {args.label_smoothing}')

    print(f'\nLoading dataset: {args.data}')
    d = np.load(args.data)
    states = torch.from_numpy(d['states']).float()
    policy_idx = torch.from_numpy(d['policy_indices']).long()
    values = torch.from_numpy(d['values']).float()
    n = states.shape[0]
    print(f'  {n} positions, states={states.shape}')

    ring_marker_mass = (states[:, :4, :, :] != 0).sum(dim=(1,2,3))
    placement_mask = ring_marker_mass < 10
    n_placement = placement_mask.sum().item()
    print(f'  Approximate placement positions: {n_placement} ({n_placement/n*100:.1f}%)')

    weights = torch.ones(n)
    weights[placement_mask] = args.placement_weight

    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=g)
    n_train = int(n * 0.9)
    tr_idx = perm[:n_train]
    va_idx = perm[n_train:]

    tr_weights = weights.clone()
    mask = torch.zeros(n, dtype=torch.bool)
    mask[tr_idx] = True
    tr_weights[~mask] = 0.0
    tr_sampler = WeightedRandomSampler(tr_weights, num_samples=n_train, replacement=True)

    train_loader = DataLoader(
        TensorDataset(states, policy_idx, values),
        batch_size=args.batch_size, sampler=tr_sampler, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(states[va_idx], policy_idx[va_idx], values[va_idx]),
        batch_size=args.batch_size, shuffle=False,
    )

    encoder = EnhancedStateEncoder()
    empty = GameState()
    empty_tensor = torch.from_numpy(encoder.encode_state(empty).astype(np.float32)).unsqueeze(0)

    def empty_top():
        net.eval()
        with torch.no_grad():
            logits, _ = net(empty_tensor)
            p = F.softmax(logits[0], dim=-1).numpy()
        pairs = []
        for m in empty.get_valid_moves():
            i = encoder.move_to_index(m)
            if 0 <= i < len(p): pairs.append((p[i], str(m.source)))
        pairs.sort(reverse=True)
        return pairs[:8], float(p.max()), float(-(p[p>1e-12] * np.log(p[p>1e-12])).sum())

    pairs0, peak0, ent0 = empty_top()
    print(f'\n=== Empty-board policy BEFORE training ===')
    for prob, pos in pairs0:
        print(f'  {pos}: {prob*100:.3f}%')
    print(f'  peak={peak0:.4f}, entropy={ent0:.3f}')

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for ep in range(args.epochs):
        t0 = time.time()
        net.train()
        tr_p = tr_v = tr_acc = tr_n = 0
        for sb, pb, vb in train_loader:
            opt.zero_grad()
            logits, value = net(sb)
            loss_p = F.cross_entropy(logits, pb, label_smoothing=args.label_smoothing)
            loss_v = F.mse_loss(value.squeeze(-1), vb) if value.dim() > 1 else F.mse_loss(value, vb)
            loss = loss_p + 0.5 * loss_v
            loss.backward()
            opt.step()
            tr_p += loss_p.item() * sb.size(0)
            tr_v += loss_v.item() * sb.size(0)
            tr_acc += (logits.argmax(-1) == pb).float().sum().item()
            tr_n += sb.size(0)
        net.eval()
        va_p = va_acc = va_n = 0
        with torch.no_grad():
            for sb, pb, vb in val_loader:
                logits, _ = net(sb)
                va_p += F.cross_entropy(logits, pb, reduction='sum', label_smoothing=args.label_smoothing).item()
                va_acc += (logits.argmax(-1) == pb).float().sum().item()
                va_n += sb.size(0)
        dt = time.time() - t0
        pairs_ep, peak_ep, ent_ep = empty_top()
        print(f'Epoch {ep+1:2d}/{args.epochs}: train_p={tr_p/tr_n:.3f} acc={tr_acc/tr_n:.3f} v={tr_v/tr_n:.3f} | '
              f'val_p={va_p/va_n:.3f} acc={va_acc/va_n:.3f} | '
              f'empty: peak={peak_ep:.4f} entropy={ent_ep:.3f} | {dt:.0f}s')
        print(f'  top-3 empty-board: {", ".join(f"{p}={pr*100:.1f}%" for pr,p in pairs_ep[:3])}')

    print('\n=== Empty-board policy AFTER training ===')
    pairs_f, peak_f, ent_f = empty_top()
    for prob, pos in pairs_f:
        print(f'  {pos}: {prob*100:.3f}%')
    print(f'  peak={peak_f:.4f}, entropy={ent_f:.3f}')
    print(f'\n  Δ from before: peak {peak0:.4f} → {peak_f:.4f} ({peak_f/peak0:.1f}× change), '
          f'entropy {ent0:.3f} → {ent_f:.3f}')

    torch.save(net.state_dict(), args.output)
    print(f'\nSaved: {args.output}')


if __name__ == '__main__':
    main()
