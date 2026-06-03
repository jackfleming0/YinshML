"""Quick opening-style characterization for the analysis-board opponent roster.

For each model: look at the raw first-ring policy over the empty board
(committal vs flat = policy entropy + top-move mass), then greedily place all
10 rings (argmax policy, alternating sides) and measure each side's ring
*spread* (mean Euclidean distance from centroid — low = clustered/corner,
high = spread). Pure forward passes; no MCTS. Fast.
"""
import sys, math
import numpy as np
import torch


def _np(t):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)

from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.game.game_state import GameState
from yinsh_ml.utils.encoding import StateEncoder

MODELS = {
    "iter1_ema": "models/iter1_ema_2026-05-27/iter1_ema.pt",
    "symmetric-15ch-iter1": "models/symmetry_run/symmetric-15ch-iter1-ema.pt",
}

enc = StateEncoder()


def _xy(pos_str):
    col = "ABCDEFGHIJK".index(pos_str[0])
    row = int(pos_str[1:])
    return col, row


def _spread(positions):
    if len(positions) < 2:
        return None
    pts = [_xy(p) for p in positions]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return sum(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 for x, y in pts) / len(pts)


def characterize(name, path):
    print(f"\n=== {name}  ({path}) ===")
    w = NetworkWrapper(model_path=path)
    print(f"  channels={'15' if w.use_enhanced_encoding else '6'}")

    gs = GameState()
    moves = gs.get_valid_moves()
    policy, value = w.predict_from_state(gs)
    policy = _np(policy).reshape(-1)

    pairs = sorted(((float(policy[enc.move_to_index(m)]), m) for m in moves), key=lambda x: -x[0])
    total = sum(p for p, _ in pairs) or 1.0
    probs = [p / total for p, _ in pairs]
    entropy = -sum(q * math.log(q + 1e-12) for q in probs)
    norm_ent = entropy / math.log(len(probs))  # 0 = one move, 1 = uniform
    print(f"  first-ring: {len(moves)} legal | top mass {probs[0]*100:4.1f}% | "
          f"top-5 mass {sum(probs[:5])*100:4.1f}% | norm-entropy {norm_ent:.3f} "
          f"({'committal' if norm_ent < 0.85 else 'flat/even'})")
    print("  top first placements: " +
          ", ".join(f"{str(m.source)} {probs[i]*100:.1f}%" for i, (_, m) in enumerate(pairs[:6])))

    # Greedy 10-ring opening, alternating sides.
    white_pos, black_pos = [], []
    for ply in range(10):
        mv = gs.get_valid_moves()
        if not mv:
            break
        pol, _ = w.predict_from_state(gs)
        pol = _np(pol).reshape(-1)
        best = max(mv, key=lambda m: float(pol[enc.move_to_index(m)]))
        (white_pos if ply % 2 == 0 else black_pos).append(str(best.source))
        gs.make_move(best)
    print(f"  greedy opening  W rings {white_pos} spread={_spread(white_pos):.2f}" if white_pos else "")
    print(f"                  B rings {black_pos} spread={_spread(black_pos):.2f}" if black_pos else "")


if __name__ == "__main__":
    for n, p in MODELS.items():
        try:
            characterize(n, p)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  FAILED: {e}", file=sys.stderr)
