#!/usr/bin/env python
"""Can a better value head exist from iter1_ema's base? (E22 follow-up diagnostic)

The value-head calibration diagnostic showed self-play freezes the value head at
AUC ~0.737. This asks whether that's *under-training* (the loop's fault) or a real
ceiling: fine-tune the value head DIRECTLY on known outcomes (strong supervised
gradient) and see if held-out AUC beats 0.737.

Default: freeze trunk + policy, train ONLY the value head (cheapest; a positive is
immediately deployable — keep iter1_ema's strong policy, swap a sharper value head).
--train-trunk also fine-tunes the trunk (more generous upper bound; breaks the
policy, so it's diagnostic-only).

Held-out split is CONTIGUOUS (last frac of the file) to approximate a game-level
holdout — positions from one game sit together, so a random split would leak.

Usage:
  python scripts/value_head_finetune_probe.py \
    --checkpoint models/iter1_ema_2026-05-27/iter1_ema.pt \
    --data expert_games/hvh_full_game_15ch.npz --epochs 8 --lr 1e-3
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from yinsh_ml.network.wrapper import NetworkWrapper


def auc_win_vs_loss(v, z):
    win, loss = v[z > 0], v[z < 0]
    if len(win) == 0 or len(loss) == 0:
        return float('nan')
    allv = np.concatenate([win, loss]); order = allv.argsort()
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    return (ranks[:len(win)].sum() - len(win) * (len(win) + 1) / 2) / (len(win) * len(loss))


def evaluate(net, S, Z, device):
    net.eval(); vals = []
    with torch.no_grad():
        for i in range(0, len(S), 512):
            _, v = net(torch.from_numpy(S[i:i+512]).float().to(device))
            vals.append(v.detach().cpu().float().numpy().ravel())
    vp = np.concatenate(vals); z = Z.astype(np.float32); dec = z != 0
    return {"auc": auc_win_vs_loss(vp, z), "brier": float(np.mean((vp - z) ** 2)),
            "sign_acc": float(np.mean(np.sign(vp[dec]) == np.sign(z[dec]))),
            "v_std": float(vp.std())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/iter1_ema_2026-05-27/iter1_ema.pt")
    ap.add_argument("--data", default="expert_games/hvh_full_game_15ch.npz")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--train-trunk", action="store_true", help="also fine-tune trunk (breaks policy; diagnostic only)")
    ap.add_argument("--save", default=None, help="save fine-tuned checkpoint here if AUC improves")
    args = ap.parse_args()

    d = np.load(args.data); S, Z = d["states"], d["values"].astype(np.float32)
    n_test = int(len(S) * args.test_frac)
    Str, Ztr = S[:-n_test], Z[:-n_test]
    Ste, Zte = S[-n_test:], Z[-n_test:]
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"train={len(Str)}  test={len(Ste)} (contiguous holdout)  device={device}")
    print(f"mode={'trunk+value' if args.train_trunk else 'VALUE-HEAD ONLY (trunk+policy frozen)'}  lr={args.lr} epochs={args.epochs}")

    nw = NetworkWrapper(model_path=args.checkpoint, device=device, use_enhanced_encoding=True)
    net = nw.network

    # Freeze everything, then unfreeze the targets.
    for p in net.parameters():
        p.requires_grad = False
    train_mods = [net.value_head] + ([net.conv_block] if args.train_trunk else [])
    params = []
    for m in train_mods:
        for p in m.parameters():
            p.requires_grad = True
            params.append(p)
    # Keep frozen modules (incl. their BatchNorm running stats) fixed; train targets in train mode.
    net.eval()
    for m in train_mods:
        m.train()

    base = evaluate(net, Ste, Zte, device)
    print(f"\nBASELINE (iter1_ema) on held-out test:  AUC={base['auc']:.4f}  Brier={base['brier']:.4f}  "
          f"sign-acc={base['sign_acc']:.3f}  v_std={base['v_std']:.3f}")
    print(f"(the diagnostic's full-corpus AUC was 0.737 — this contiguous-test baseline anchors before/after)\n")

    opt = torch.optim.Adam(params, lr=args.lr)
    rng = np.random.default_rng(0)
    best = dict(base); best_ep = 0
    for ep in range(1, args.epochs + 1):
        for m in train_mods:
            m.train()
        order = rng.permutation(len(Str))
        tot = 0.0; nb = 0
        for i in range(0, len(Str), args.batch):
            bi = order[i:i+args.batch]
            x = torch.from_numpy(Str[bi]).float().to(device)
            z = torch.from_numpy(Ztr[bi]).float().to(device)
            _, v = net(x)
            loss = F.mse_loss(v.view(-1), z)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        m_te = evaluate(net, Ste, Zte, device)
        flag = "  <-- best" if m_te["auc"] > best["auc"] else ""
        print(f"epoch {ep}: train_mse={tot/nb:.4f}  test AUC={m_te['auc']:.4f}  Brier={m_te['brier']:.4f}  "
              f"sign-acc={m_te['sign_acc']:.3f}  v_std={m_te['v_std']:.3f}{flag}")
        if m_te["auc"] > best["auc"]:
            best = dict(m_te); best_ep = ep
            if args.save:
                nw.save_model(args.save)

    d_auc = best["auc"] - base["auc"]
    print(f"\n=== VERDICT ===")
    print(f"baseline AUC {base['auc']:.4f} -> best AUC {best['auc']:.4f} (epoch {best_ep})  Δ={d_auc:+.4f}")
    if d_auc > 0.02:
        print("AUC IMPROVED meaningfully -> the value head was UNDER-trained by the loop, not capped.")
    else:
        print("AUC ~flat -> 0.737 is a ceiling for this base (architecture/evaluability), not under-training.")


if __name__ == "__main__":
    main()
