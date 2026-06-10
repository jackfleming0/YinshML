#!/usr/bin/env python
"""E26 distillation — bank the search-improved targets into the net.

Continues training from iter1 (the student) on the high-sim teacher corpus from
`gen_distill_corpus.py` (states, top-K search policy, search value). The POLICY is
the lever (E25): the MCTS visit distribution exceeds the raw prior, so distilling it
sharpens the part MCTS actually exploits. Value is distilled too (small weight) to
keep the value head anchored while the shared trunk shifts — it's at an intrinsic
ceiling, so it's secondary, not the target.

Loss mirrors `trainer.py`:
  policy: soft cross-entropy  -(target_probs * log_softmax(policy_logits)).sum
  value : cross-entropy to the nearest of 7 outcome bins (linspace(-1,1,7))

Output is a raw state_dict, loadable by NetworkWrapper for the H2H gate
(`measure_h2h.py` distilled vs frozen iter1 — the only verdict, R1).

Usage (box):
  python scripts/e26_distill.py --init models/iter1_ema_2026-05-27/iter1_ema.pt \
      --data expert_games/e26_teacher_800sim.npz --out models/e26_distilled.pt \
      --epochs 6 --lr 2e-4 --batch 1024 --value-weight 0.5 --device cuda

Local smoke:
  python scripts/e26_distill.py --init models/iter1_ema_2026-05-27/iter1_ema.pt \
      --data /tmp/e26_smoke.npz --out /tmp/e26_distilled_smoke.pt \
      --epochs 2 --batch 32 --device cpu
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from yinsh_ml.network.wrapper import NetworkWrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="checkpoint to continue from (iter1)")
    ap.add_argument("--data", required=True, help="teacher corpus from gen_distill_corpus.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--value-weight", type=float, default=0.5)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-last", action="store_true",
                    help="save the last epoch instead of the best-test-polCE epoch "
                         "(default = save best; distillation overfits, so last is usually worse)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else
                             ("mps" if torch.backends.mps.is_available() else "cpu"))

    d = np.load(args.data)
    S, PIDX, PPROB, V = d["states"], d["policy_idx"], d["policy_prob"], d["values"].astype(np.float32)
    n = len(S); n_te = int(n * args.test_frac)
    rng = np.random.default_rng(args.seed); perm = rng.permutation(n)
    te, tr = perm[:n_te], perm[n_te:]
    print(f"corpus={n}  train={len(tr)} test={len(te)}  states{S.shape} | device={device}")

    nw = NetworkWrapper(model_path=args.init, device=device)  # student = iter1
    net = nw.network
    total_moves = nw.state_encoder.total_moves
    # infer value bins from a forward
    with torch.no_grad():
        _, vl0 = net(torch.from_numpy(S[:2]).float().to(device))
    nbins = vl0.reshape(2, -1).shape[1]
    scalar_value = (nbins == 1)  # iter1 uses a regression (scalar) value head
    centers = None if scalar_value else torch.linspace(-1, 1, nbins, device=device)
    vmetric = "te_valMSE" if scalar_value else "te_valAcc"
    print(f"policy_size={total_moves}  value head={'scalar (MSE)' if scalar_value else f'{nbins}-class (CE)'}")

    def scatter_policy(idx, prob):
        """[B,K] sparse top-K -> dense [B,total_moves] soft target."""
        b = idx.shape[0]
        t = torch.zeros(b, total_moves, device=device)
        mask = (idx >= 0).float()
        t.scatter_add_(1, idx.clamp(min=0).long(), prob * mask)
        return t

    def value_loss_fn(vl, v):
        if scalar_value:
            return F.mse_loss(vl.view(-1), v)
        tgt = (v.view(-1, 1) - centers.view(1, -1)).abs().argmin(1)
        return F.cross_entropy(vl, tgt)

    @torch.no_grad()
    def evaluate(split):
        net.eval()
        pol_ce, vm_sum, seen = 0.0, 0.0, 0
        for i in range(0, len(split), args.batch):
            bi = split[i:i + args.batch]
            x = torch.from_numpy(S[bi]).float().to(device)
            tp = scatter_policy(torch.from_numpy(PIDX[bi]).to(device),
                                torch.from_numpy(PPROB[bi]).float().to(device))
            v = torch.from_numpy(V[bi]).float().to(device)
            pl, vl = net(x); vl = vl.reshape(len(bi), -1)
            pol_ce += float(-(tp * F.log_softmax(pl, dim=1)).sum(1).sum())
            if scalar_value:
                vm_sum += float(((vl.view(-1) - v) ** 2).sum())          # MSE (lower better)
            else:
                tgt = (v.view(-1, 1) - centers.view(1, -1)).abs().argmin(1)
                vm_sum += int((vl.argmax(1) == tgt).sum())                # acc (higher better)
            seen += len(bi)
        return pol_ce / seen, vm_sum / seen

    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    p0, v0 = evaluate(te)
    print(f"\nzero-shot (iter1):  test policy-CE={p0:.4f}  {vmetric}={v0:.4f}\n")
    print(f"{'epoch':>5}{'train_loss':>12}{'pol_loss':>10}{'val_loss':>10}{'te_polCE':>10}{vmetric:>11}")
    print("-" * 58)
    idx = np.array(tr)
    best_te = float('inf'); best_state = None; best_ep = 0
    for ep in range(1, args.epochs + 1):
        net.train(); np.random.shuffle(idx)
        tot = pl_s = vl_s = nb = 0.0
        for i in range(0, len(idx), args.batch):
            bi = idx[i:i + args.batch]
            x = torch.from_numpy(S[bi]).float().to(device)
            tp = scatter_policy(torch.from_numpy(PIDX[bi]).to(device),
                                torch.from_numpy(PPROB[bi]).float().to(device))
            v = torch.from_numpy(V[bi]).float().to(device)
            pl, vl = net(x); vl = vl.reshape(len(bi), -1)
            policy_loss = -(tp * F.log_softmax(pl, dim=1)).sum(1).mean()
            value_loss = value_loss_fn(vl, v)
            loss = policy_loss + args.value_weight * value_loss
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); pl_s += policy_loss.item(); vl_s += value_loss.item(); nb += 1
        te_p, te_v = evaluate(te)
        is_best = te_p < best_te
        if is_best:
            best_te, best_ep = te_p, ep
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        print(f"{ep:>5}{tot/nb:>12.4f}{pl_s/nb:>10.4f}{vl_s/nb:>10.4f}{te_p:>10.4f}{te_v:>11.3f}"
              f"{'  *best' if is_best else ''}")

    # Save the BEST checkpoint (lowest test policy-CE), not the last: distillation
    # overfits (test polCE bottoms then rises while train loss keeps dropping), so
    # the last epoch is usually worse. --save-last restores the old behavior.
    if args.save_last or best_state is None:
        torch.save(net.state_dict(), args.out)
        print(f"\nsaved distilled student (LAST epoch) -> {args.out}")
    else:
        torch.save(best_state, args.out)
        print(f"\nsaved distilled student (BEST: epoch {best_ep}/{args.epochs}, "
              f"te_polCE={best_te:.4f}) -> {args.out}")
    print("Next: H2H vs FROZEN iter1 (the only verdict):")
    print(f"  python scripts/measure_h2h.py --white {args.out} --black {args.init} "
          f"--white-label distilled --black-label iter1 --games 60 --output logs/e26_h2h.json")


if __name__ == "__main__":
    main()
