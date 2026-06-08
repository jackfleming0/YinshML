#!/usr/bin/env python
"""Value-head ceiling probe — can iter1_ema's value head beat its frozen 0.737
discrimination AUC if it's actually *trained* on decisive, value-spread data?

Context: across the self-play lineage the value head is stuck at AUC 0.737 on
the human (decisive) corpus, identical to the supervised base — self-play never
moved it (`value_head_calibration.py`). The e24 diagnosis: mirror self-play only
visits ~balanced positions, so the value targets have no spread to learn from.
This probe tests whether that's the binding constraint or an architectural wall:

  train ONLY on a human train split (decisive outcomes), eval AUC on a held-out
  human test split. The corpus is game-ordered, so we split contiguously with a
  gap to avoid same-game leakage between train and test.

  * frozen-backbone (default): retrain just the value head on iter1's existing
    features. AUC ≫ 0.737 ⇒ the representation already separates win/loss; the
    head just never saw spread data ⇒ cheap fix (retrain head on spread).
  * --full-net: also fine-tune the trunk. If even this can't beat 0.737, the
    ceiling is representational/architectural and value discrimination is not
    the lever.

Either outcome localizes the e25 binding constraint. No self-play, Mac-runnable.

Usage:
  python scripts/value_ceiling_probe.py \
    --ckpt models/iter1_ema_2026-05-27/iter1_ema.pt \
    --epochs 8 --lr 1e-3            # frozen head
  python scripts/value_ceiling_probe.py --full-net --lr 1e-4 --epochs 6
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from yinsh_ml.network.wrapper import NetworkWrapper


def auc_win_vs_loss(v, z):
    win, loss = v[z > 0], v[z < 0]
    if len(win) == 0 or len(loss) == 0:
        return float("nan")
    allv = np.concatenate([win, loss])
    order = allv.argsort()
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    r_win = ranks[: len(win)].sum()
    return (r_win - len(win) * (len(win) + 1) / 2) / (len(win) * len(loss))


def value_scalar(value_logits: torch.Tensor) -> np.ndarray:
    """Map the value head output to a scalar in [-1, 1] (matches calibration)."""
    v = value_logits.detach().cpu().float()
    v = v.reshape(v.shape[0], -1)
    if v.shape[1] == 1:
        return v[:, 0].numpy()
    p = torch.softmax(v, dim=1).numpy()
    centers = np.linspace(-1, 1, v.shape[1])
    return p @ centers


@torch.no_grad()
def eval_auc(net, S, Z, device, bs=1024):
    net.eval()
    vs = []
    for i in range(0, len(S), bs):
        b = torch.from_numpy(S[i:i + bs]).float().to(device)
        _, vlog = net(b)
        vs.append(value_scalar(vlog))
    v = np.concatenate(vs)
    return auc_win_vs_loss(v, Z), float(v.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/iter1_ema_2026-05-27/iter1_ema.pt")
    ap.add_argument("--data", default="expert_games/hvh_full_game_15ch.npz")
    ap.add_argument("--full-net", action="store_true",
                    help="fine-tune trunk too (default: value head only).")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0, help="weight decay (regularize overfit).")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--train-cap", type=int, default=0,
                    help="randomly subsample the train split to this many "
                         "positions (0 = use all). test split is untouched.")
    ap.add_argument("--eval-cap", type=int, default=0,
                    help="subsample the test set used for AUC eval (0 = all). "
                         "Keeps per-epoch eval fast on CPU.")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--gap", type=int, default=1000,
                    help="positions dropped at the train/test boundary to kill "
                         "same-game leakage.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="force cpu|mps|cuda.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"))

    d = np.load(args.data)
    S, Z = d["states"], d["values"].astype(np.float32)
    n = len(S)
    n_test = int(n * args.test_frac)
    cut = n - n_test
    Str, Ztr = S[: cut - args.gap], Z[: cut - args.gap]
    Ste, Zte = S[cut:], Z[cut:]
    if args.eval_cap and args.eval_cap < len(Ste):
        e = np.random.default_rng(args.seed + 1).choice(
            len(Ste), size=args.eval_cap, replace=False)
        Ste, Zte = Ste[e], Zte[e]
    if args.train_cap and args.train_cap < len(Str):
        sub = np.random.default_rng(args.seed).choice(
            len(Str), size=args.train_cap, replace=False)
        Str, Ztr = Str[sub], Ztr[sub]
    print(f"contiguous game-ordered split: train={len(Str)} test={len(Ste)} "
          f"(gap={args.gap}) | device={device}")
    print(f"train decisive={(Ztr != 0).sum()} draws={(Ztr == 0).sum()} | "
          f"test decisive={(Zte != 0).sum()} draws={(Zte == 0).sum()}")

    nw = NetworkWrapper(model_path=args.ckpt, device=device,
                        use_enhanced_encoding=True)
    net = nw.network
    # infer value-bin count from a dummy forward
    with torch.no_grad():
        _, vlog0 = net(torch.from_numpy(S[:2]).float().to(device))
    vlog0 = vlog0.reshape(vlog0.shape[0], -1)
    nbins = vlog0.shape[1]
    centers = torch.linspace(-1, 1, nbins, device=device) if nbins > 1 else None
    print(f"value head: {'scalar' if nbins == 1 else f'{nbins}-bin classification'}")

    # baseline (zero-shot) test AUC
    auc0, std0 = eval_auc(net, Ste, Zte, device)
    print(f"\nzero-shot (before any training):  test AUC={auc0:.3f}  v_std={std0:.3f}\n")

    # choose trainable params
    if args.full_net:
        params = [p for p in net.parameters()]
        mode = "FULL-NET"
    else:
        params = [p for n_, p in net.named_parameters() if "value" in n_.lower()]
        for n_, p in net.named_parameters():
            p.requires_grad = "value" in n_.lower()
        mode = "FROZEN-BACKBONE (value head only)"
    n_train_params = sum(p.numel() for p in params if p.requires_grad)
    print(f"mode: {mode}  ({n_train_params:,} trainable params)\n")
    opt = torch.optim.Adam([p for p in params if p.requires_grad],
                           lr=args.lr, weight_decay=args.wd)

    def target_bins(zb):
        if nbins == 1:
            return zb.view(-1, 1)
        # nearest bin index for z in {-1,0,1}
        idx = (torch.from_numpy(np.abs(np.subtract.outer(
            zb.cpu().numpy(), np.linspace(-1, 1, nbins))).argmin(1))).long()
        return idx.to(zb.device)

    idx_all = np.arange(len(Str))
    print(f"{'epoch':>5}{'train_loss':>12}{'train_auc':>11}{'test_auc':>10}{'test_std':>10}")
    print("-" * 48)
    for ep in range(1, args.epochs + 1):
        net.train()
        np.random.shuffle(idx_all)
        losses = []
        for i in range(0, len(idx_all), args.batch):
            bi = idx_all[i:i + args.batch]
            xb = torch.from_numpy(Str[bi]).float().to(device)
            zb = torch.from_numpy(Ztr[bi]).float().to(device)
            _, vlog = net(xb)
            vlog = vlog.reshape(vlog.shape[0], -1)
            if nbins == 1:
                loss = F.mse_loss(vlog.view(-1), zb)
            else:
                loss = F.cross_entropy(vlog, target_bins(zb))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        tr_auc, _ = eval_auc(net, Str[:8000], Ztr[:8000], device)
        te_auc, te_std = eval_auc(net, Ste, Zte, device)
        print(f"{ep:>5}{np.mean(losses):>12.4f}{tr_auc:>11.3f}{te_auc:>10.3f}{te_std:>10.3f}")

    print(f"\nVERDICT: zero-shot test AUC {auc0:.3f} -> best after training above.")
    print("  ≫0.737 -> value head CAN discriminate given spread data "
          "(binding constraint = self-play distribution).")
    print("  ≈0.737 -> representational ceiling (value discrimination not the lever).")


if __name__ == "__main__":
    main()
