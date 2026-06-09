#!/usr/bin/env python3
"""Per-ply value-head eval trajectory for analysis-board games, judged by a
chosen oracle checkpoint. Shows where a game's eval crossed from losing to
winning (from the engine's POV) and flags sharp single-move swings (candidate
blunders) — without running MCTS.

    python scripts/eval_trajectory.py 0ee9d9ce fd6c49ce \
        --model models/iter1_ema_2026-05-27/iter1_ema.pt

Value is the network's own head (no search): trajectory SHAPE is reliable;
individual plies (esp. mid-capture sequences) are noisier.
"""
import argparse

import numpy as np
import torch

import lib_game_logs as L
from yinsh_ml.network.wrapper import NetworkWrapper

DEFAULT_MODEL = "models/iter1_ema_2026-05-27/iter1_ema.pt"


def load_oracle(path):
    for enh in (False, True):
        try:
            w = NetworkWrapper(model_path=path, device="cpu", use_enhanced_encoding=enh)
            print(f"  oracle loaded ({'15ch enhanced' if enh else '6ch basic'})")
            return w
        except Exception as e:
            print(f"  enhanced={enh} failed: {str(e)[:80]}")
    raise SystemExit(f"could not load oracle {path}")


def value_scalar(value):
    """Reduce the value head output to a scalar in [-1,1] (current-player POV)."""
    v = value.detach().cpu().numpy().reshape(-1)
    if v.size == 1:
        return float(v[0])
    buckets = np.linspace(-1, 1, v.size)
    p = np.exp(v - v.max())
    return float((buckets * (p / p.sum())).sum())


def trajectory(rows, eng, wrapper):
    hum = "BLACK" if eng == "WHITE" else "WHITE"
    traj = []
    for i, r in enumerate(rows):
        npos = r.get("new_position") or {}
        try:
            gs = L.build_state(npos)
        except Exception:
            continue
        with torch.no_grad():
            _, value = wrapper.predict_from_state(gs)
        v_stm = value_scalar(value)
        v_eng = v_stm if gs.current_player.name == eng else -v_stm
        sc = npos.get("scores") or {}
        traj.append({
            "i": i + 1,
            "mover": "E" if r["pre_position"]["side_to_move"] == eng else "H",
            "v": v_eng, "e": sc.get(eng, 0), "h": sc.get(hum, 0),
            "desc": (r.get("applied_move") or {}).get("description", ""),
        })
    return traj


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", help="play_session_id prefixes")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--swing", type=float, default=0.25,
                    help="min eval jump toward engine on a human move to flag as a blunder")
    ap.add_argument("--log-dir", default=L.DEFAULT_LOG_DIR)
    args = ap.parse_args()

    sessions = L.load_sessions(args.log_dir)
    print(f"Oracle: {args.model}")
    w = load_oracle(args.model)

    for pref in args.sessions:
        match = [(sid, rows) for sid, rows in sessions.items() if sid.startswith(pref)]
        if not match:
            print(f"\n[{pref}] no matching session")
            continue
        sid, rows = match[0]
        eng = L.engine_side(rows)
        if eng not in ("WHITE", "BLACK"):
            print(f"\n[{pref}] could not determine engine side")
            continue
        traj = trajectory(rows, eng, w)
        if not traj:
            continue

        print("\n" + "=" * 80)
        print(f"GAME {sid[:8]}  engine={eng}  (final {rows[-1].get('winner')} wins)")
        print("=" * 80)
        print(f"{'ply':>3} {'by':>2} {'E-eval':>7} {'score':>6}  bar")
        prev_v = 0.0
        swings = []
        for t in traj:
            v = t["v"]
            sw = v - prev_v
            if t["mover"] == "H" and sw >= args.swing:
                swings.append((t["i"], sw, t["desc"]))
            pos = int(round((v + 1) / 2 * 30))
            bar = "·" * pos + "|" + "·" * (30 - pos)
            mark = f"  <== H swing {sw:+.2f}" if (t["mover"] == "H" and sw >= args.swing) else ""
            print(f"{t['i']:>3} {t['mover']:>2} {v:>+7.2f} {t['e']}-{t['h']:<3} {bar}{mark}")
            prev_v = v

        vals = [t["v"] for t in traj]
        min_v = min(vals)
        min_ply = traj[vals.index(min_v)]["i"]
        cross = None
        for k in range(1, len(traj)):
            if traj[k-1]["v"] < 0 <= traj[k]["v"]:
                cross = traj[k]["i"]
        print(f"\n  lowest engine eval: {min_v:+.2f} (ply {min_ply})   "
              f"last losing->winning crossover: ply {cross}")
        if swings:
            print("  candidate human blunders (eval jumped toward engine on a human move):")
            for ply, sw, desc in swings:
                print(f"    ply {ply}: {sw:+.2f}  {desc}")
        else:
            print("  no sharp human blunder — eval moved gradually (engine outplayed).")


if __name__ == "__main__":
    main()
