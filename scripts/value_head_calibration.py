#!/usr/bin/env python
"""Value-head calibration / discrimination on a held-out, known-outcome corpus.

Measures how well each checkpoint's value head predicts the actual game outcome
(z in {-1,0,+1}, side-to-move POV) on the H-vs-H corpus. Reports calibration
(Brier) AND discrimination (sign-acc on decisive positions, correlation, and a
win/loss ranking AUC) — discrimination is what actually matters for guiding MCTS.

Convention validated against the 2026-05-29 P2 diagnostic: iter1_ema -> Brier
~0.67, baseline (predict 0) ~0.78.

Usage:
  python scripts/value_head_calibration.py \
    --data expert_games/hvh_full_game_15ch.npz --n 8000 --seed 0 \
    iter1_ema:models/iter1_ema_2026-05-27/iter1_ema.pt \
    armA_mirror:models/e22_checkpoints/armA_iter4_ema.pt \
    armB_cross:models/e22_checkpoints/armB_iter4_ema.pt
"""
import argparse, sys
import numpy as np
import torch
from yinsh_ml.network.wrapper import NetworkWrapper


def auc_win_vs_loss(v, z):
    """P(v_pred(win) > v_pred(loss)) over decisive positions — Mann-Whitney AUC.
    0.5 = no discrimination, 1.0 = perfect."""
    win, loss = v[z > 0], v[z < 0]
    if len(win) == 0 or len(loss) == 0:
        return float('nan')
    # rank-based: average over all win/loss pairs (vectorized via ranks)
    allv = np.concatenate([win, loss])
    order = allv.argsort()
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    r_win = ranks[:len(win)].sum()
    return (r_win - len(win) * (len(win) + 1) / 2) / (len(win) * len(loss))


def eval_checkpoint(label, path, S, Z, device):
    nw = NetworkWrapper(model_path=path, device=device, use_enhanced_encoding=True)
    nw.network.eval()
    vals = []
    with torch.no_grad():
        for i in range(0, len(S), 512):
            b = torch.from_numpy(S[i:i+512]).float().to(device)
            _, v = nw.network(b)
            vals.append(v.detach().cpu().float().numpy().reshape(len(b), -1))
    v = np.concatenate(vals)
    if v.shape[1] != 1:
        # classification head: map logits -> expected value over evenly-spaced bins in [-1,1]
        p = torch.softmax(torch.from_numpy(v), dim=1).numpy()
        centers = np.linspace(-1, 1, v.shape[1])
        vp = p @ centers
        note = f"(classification head, {v.shape[1]} bins -> expected value)"
    else:
        vp = v[:, 0]
        note = ""
    z = Z.astype(np.float32)
    dec = z != 0
    return {
        "label": label,
        "brier": float(np.mean((vp - z) ** 2)),
        "sign_acc_dec": float(np.mean(np.sign(vp[dec]) == np.sign(z[dec]))),
        "corr": float(np.corrcoef(vp, z)[0, 1]),
        "auc": float(auc_win_vs_loss(vp, z)),
        "v_std": float(vp.std()),
        "n_dec": int(dec.sum()),
        "note": note,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="expert_games/hvh_full_game_15ch.npz")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("checkpoints", nargs="+", help="label:path pairs")
    args = ap.parse_args()

    d = np.load(args.data)
    S, Z = d["states"], d["values"]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(S), size=min(args.n, len(S)), replace=False)
    S, Z = S[idx], Z[idx]
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Held-out: {len(S)} positions ({(Z != 0).sum()} decisive, {(Z == 0).sum()} draws) | device={device}")
    print(f"Baseline Brier (predict 0): {np.mean((0 - Z.astype(np.float32)) ** 2):.4f}\n")

    rows = []
    for spec in args.checkpoints:
        label, path = spec.split(":", 1)
        rows.append(eval_checkpoint(label, path, S, Z, device))

    print(f"{'checkpoint':<16}{'Brier↓':>9}{'sign-acc↑':>11}{'corr↑':>8}{'AUC↑':>8}{'v_std':>8}")
    print("-" * 60)
    for r in rows:
        print(f"{r['label']:<16}{r['brier']:>9.4f}{r['sign_acc_dec']:>11.3f}{r['corr']:>8.3f}{r['auc']:>8.3f}{r['v_std']:>8.3f}  {r['note']}")
    print("\n↑ higher better (discrimination), ↓ lower better (calibration). "
          "AUC = P(value ranks a win above a loss); 0.5 = blind.")


if __name__ == "__main__":
    main()
