"""Dynamic confirmation of the E16 value_weight choice.

The static gradient analysis (investigate_e16_value_weight.py) says
value_weight ~10 equalizes the two terms' gradient pressure and 0.5 starves
the value term. This probe trains the net for a few hundred steps under a
realistic task loss + the E16 regularizer at several value_weights and tracks
whether value_asym actually gets driven down — and whether a high weight costs
policy accuracy. Same model/data as the static script.

Main loss   = CE(policy, label_smoothing=0.1) + 0.5 * MSE(value, target)
Reg (alpha) = 0.1 * (kl + w * value_asym)     applied every step
"""

import argparse, copy
import numpy as np
import torch
import torch.nn.functional as F

from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.training.trainer import YinshTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='models/iter1_ema_2026-05-27/iter1_ema.pt')
    p.add_argument('--data', default='expert_games/hvh_full_game_15ch.npz')
    p.add_argument('--pool', type=int, default=512, help='positions held fixed across steps')
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--steps', type=int, default=150)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--weights', default='0.5,10,20')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def valid_mask(enc, s):
    gs = enc.decode_state(s)
    m = np.zeros(enc.total_moves, dtype=np.float32)
    for mv in gs.get_valid_moves():
        i = enc.move_to_index(mv)
        if 0 <= i < len(m):
            m[i] = 1.0
    return m


def reg_terms(net, states, masks, reg_tensors):
    eps = 1e-9
    b = states.shape[0]
    flat = states.reshape(b, states.shape[1], -1)

    def md(logits):
        p = F.softmax(logits.float(), 1) * masks
        return p / (p.sum(1, keepdim=True) + eps)

    l0, v0 = net(states)
    pols = [md(l0)]
    vals = [v0.float().reshape(b, -1)]
    for cell_src, perm in reg_tensors:
        lt, vt = net(flat[:, :, cell_src].reshape(states.shape))
        lo = torch.empty_like(lt)
        lo.index_copy_(1, perm, lt)
        pols.append(md(lo))
        vals.append(vt.float().reshape(b, -1))
    psym = torch.stack(pols, 0).mean(0)
    kl = (psym * (torch.log(psym + eps) - torch.log(pols[0] + eps))).sum(1).mean()
    vs = torch.stack(vals, 0)
    va = ((vs - vs.mean(0, keepdim=True)) ** 2).mean()
    return kl, va, l0, v0


def main():
    args = parse_args()
    weights = [float(x) for x in args.weights.split(',')]
    rng = np.random.default_rng(args.seed)

    nw = NetworkWrapper(model_path=args.checkpoint, device='cpu', use_enhanced_encoding=True)
    enc = nw.state_encoder
    tr = YinshTrainer(network=nw, device='cpu', enable_symmetric_reg=True)
    reg_tensors = tr._build_symmetric_reg_tensors()
    base_state = copy.deepcopy(nw.network.state_dict())

    d = np.load(args.data)
    n_total = d['states'].shape[0]
    idx = rng.choice(n_total, size=min(args.pool, n_total), replace=False)
    states = torch.from_numpy(d['states'][idx].astype(np.float32))
    # hard policy targets
    pol = d['policy_indices'][idx] if 'policy_indices' in d.files else d['policies'][idx].argmax(1)
    targets = torch.from_numpy(pol.astype(np.int64))
    tvals = torch.from_numpy(d['values'][idx].astype(np.float32)).reshape(-1)
    masks = torch.from_numpy(np.stack([valid_mask(enc, s) for s in d['states'][idx].astype(np.float32)]))
    print(f"pool={len(idx)} positions, batch={args.batch}, steps={args.steps}, "
          f"lr={args.lr}, alpha={args.alpha}\n")

    def run(w):
        net = nw.network
        net.load_state_dict(base_state)
        net.train()
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-3)
        order = rng.permutation(len(idx))
        ptr = 0
        hist = []
        for step in range(args.steps):
            if ptr + args.batch > len(order):
                order = rng.permutation(len(idx)); ptr = 0
            bi = torch.from_numpy(order[ptr:ptr + args.batch]); ptr += args.batch
            sb, tb, vb, mb = states[bi], targets[bi], tvals[bi], masks[bi]
            opt.zero_grad()
            kl, va, logits, value = reg_terms(net, sb, mb, reg_tensors)
            ce = F.cross_entropy(logits, tb, label_smoothing=0.1)
            vmse = F.mse_loss(value.float().reshape(-1), vb)
            loss = ce + 0.5 * vmse + args.alpha * (kl + w * va)
            loss.backward(); opt.step()
            if step % 15 == 0 or step == args.steps - 1:
                acc = (logits.argmax(1) == tb).float().mean().item()
                hist.append((step, kl.item(), va.item(), acc, vmse.item()))
        return hist

    print("=" * 86)
    print(f"{'w':>5s} | {'step':>4s} | {'kl':>9s} | {'value_asym':>11s} | "
          f"{'va vs start':>11s} | {'pol_acc':>7s} | {'val_mse':>8s}")
    print("-" * 86)
    summary = {}
    for w in weights:
        h = run(w)
        va0 = h[0][2]
        for (st, kl, va, acc, vmse) in h:
            if st in (0, h[len(h) // 2][0], h[-1][0]):
                print(f"{w:>5.1f} | {st:>4d} | {kl:>9.5f} | {va:>11.6f} | "
                      f"{va/va0*100:>10.1f}% | {acc:>7.3f} | {vmse:>8.4f}")
        summary[w] = (va0, h[-1][2], h[-1][3])
        print("-" * 86)

    print("\nSUMMARY — value_asym reduction & final policy acc")
    print(f"{'w':>5s} | {'va start':>9s} | {'va final':>9s} | {'reduction':>9s} | {'final pol_acc':>13s}")
    for w in weights:
        va0, vaf, acc = summary[w]
        print(f"{w:>5.1f} | {va0:>9.6f} | {vaf:>9.6f} | {(1-vaf/va0)*100:>8.1f}% | {acc:>13.3f}")


if __name__ == '__main__':
    main()
