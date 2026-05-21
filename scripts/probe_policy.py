"""Inspect a trained checkpoint's raw policy distribution on real game states.

Diagnoses whether a "policy head learned nothing" failure (cloud_run_v1's
50% / 0% / 0% pattern) is mode-collapse, deterministic-but-wrong, or
near-uniform — each implies a different fix. See CLOUD_RUN_V1_POSTMORTEM.md
for the failure context.

Usage:
    python scripts/probe_policy.py --checkpoint <ckpt.pt> --buffer <buf.pkl.gz>

    # Or shortcut: point at a run dir, pick the latest iteration_*/checkpoint.
    python scripts/probe_policy.py --run-dir runs/20260427_000821

The probe reports four signals across N sampled states:
    - entropy of softmax(logits): low = collapsed/peaky, high (~ln(7433)≈8.91) = untrained
    - top-1 confidence: high + low entropy = peaky
    - # unique top-1 moves across the sample: low (1-5) = mode collapse
    - top-1 move index histogram: shows whether one move dominates

Maps to hypotheses in the postmortem follow-up:
    entropy ≈ 0, few unique top-1 → policy emits same handful of moves regardless of state.
    entropy ≈ 0, many unique top-1 → memorised hybrid-MCTS, can't stand alone.
    entropy ≈ ln(7433)            → head literally didn't move; gradient-flow bug.
"""

from __future__ import annotations

import argparse
import gzip
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Project imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yinsh_ml.network.wrapper import NetworkWrapper  # noqa: E402


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    """Return the highest-numbered iteration_N/checkpoint_iteration_N.pt under run_dir."""
    candidates = sorted(
        run_dir.glob("iteration_*/checkpoint_iteration_*.pt"),
        key=lambda p: int(p.parent.name.split("_")[-1]),
    )
    # Prefer non-EMA over EMA when both exist for the same iteration.
    non_ema = [p for p in candidates if not p.stem.endswith("_ema")]
    return non_ema[-1] if non_ema else (candidates[-1] if candidates else None)


def find_buffer(run_dir: Path) -> Path | None:
    for name in ("replay_buffer.pkl.gz", "replay_buffer.pkl"):
        p = run_dir / name
        if p.exists():
            return p
    return None


def load_buffer(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, help="Path to .pt checkpoint.")
    ap.add_argument("--buffer", type=Path, help="Path to replay_buffer.pkl[.gz].")
    ap.add_argument("--run-dir", type=Path, help="Shortcut: pick latest ckpt + buffer from run dir.")
    ap.add_argument("--num-states", type=int, default=64, help="States to sample.")
    ap.add_argument("--device", default="auto", help="cuda | mps | cpu | auto.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-enhanced-encoding", action="store_true",
                    help="Set if checkpoint was trained with 15-channel encoding.")
    ap.add_argument("--top-k", type=int, default=5, help="K for top-K confidence + histogram.")
    args = ap.parse_args()

    if args.run_dir:
        if args.checkpoint is None:
            args.checkpoint = find_latest_checkpoint(args.run_dir)
            if args.checkpoint is None:
                ap.error(f"No checkpoint_iteration_*.pt under {args.run_dir}")
        if args.buffer is None:
            args.buffer = find_buffer(args.run_dir)
            if args.buffer is None:
                ap.error(f"No replay_buffer.pkl[.gz] under {args.run_dir}")
    if not args.checkpoint or not args.buffer:
        ap.error("Need --checkpoint AND --buffer (or --run-dir).")

    rng = np.random.default_rng(args.seed)
    device = None if args.device == "auto" else args.device

    print(f"checkpoint: {args.checkpoint}")
    print(f"buffer:     {args.buffer}")

    nw = NetworkWrapper(
        model_path=str(args.checkpoint),
        device=device,
        use_enhanced_encoding=args.use_enhanced_encoding,
    )
    nw.network.eval()
    print(f"device:     {nw.device}, total_moves: {nw.state_encoder.total_moves}")

    buf = load_buffer(args.buffer)
    states = list(buf.get("states", []))
    phases = list(buf.get("phases", []))
    if len(states) == 0:
        ap.error("Replay buffer has no states.")

    n = min(args.num_states, len(states))
    idxs = rng.choice(len(states), size=n, replace=False)
    sampled = np.stack([np.asarray(states[i], dtype=np.float32) for i in idxs])
    sampled_phases = [phases[i] if i < len(phases) else "?" for i in idxs]
    print(f"sampled:    {n} of {len(states)} states  (phases: {Counter(sampled_phases)})")

    states_t = torch.from_numpy(sampled).to(nw.device)
    with torch.no_grad():
        logits, value = nw.network(states_t)
        probs = F.softmax(logits.float(), dim=-1)
        log_probs = probs.clamp_min(1e-12).log()
        entropy = -(probs * log_probs).sum(dim=-1)         # (N,)
        top1_conf = probs.max(dim=-1).values               # (N,)
        top1_idx = probs.argmax(dim=-1)                    # (N,)
        topk = torch.topk(probs, k=args.top_k, dim=-1)
        topk_mass = topk.values.sum(dim=-1)                # (N,)

    n_classes = probs.shape[-1]
    uniform_entropy = float(np.log(n_classes))

    # Aggregates
    H_mean = float(entropy.mean())
    H_med = float(entropy.median())
    H_min = float(entropy.min())
    H_max = float(entropy.max())
    conf_mean = float(top1_conf.mean())
    conf_med = float(top1_conf.median())
    topk_mean = float(topk_mass.mean())
    unique_top1 = sorted(set(top1_idx.tolist()))
    counts = Counter(top1_idx.tolist())
    most_common = counts.most_common(args.top_k)

    print()
    print("─── policy distribution diagnostics ─────────────────────────")
    print(f"  uniform reference entropy ≈ ln({n_classes}) = {uniform_entropy:.3f}")
    print(f"  entropy:        mean={H_mean:.3f}  median={H_med:.3f}  min={H_min:.3f}  max={H_max:.3f}")
    print(f"  entropy / ln(N) ratio:  {H_mean / uniform_entropy:.3f}  "
          f"(1.0 = uniform, 0.0 = one-hot)")
    print(f"  top-1 confidence:  mean={conf_mean:.4f}  median={conf_med:.4f}")
    print(f"  top-{args.top_k} mass:        mean={topk_mean:.4f}")
    print(f"  unique top-1 moves: {len(unique_top1)} / {n}")
    print(f"  top-{args.top_k} most-frequent argmax moves (idx, count):")
    for idx, c in most_common:
        print(f"      slot {idx:>5}  ×{c}")

    print()
    print("─── value head ──────────────────────────────────────────────")
    v = value.float().cpu().numpy()
    print(f"  pred_value:  mean={v.mean():+.3f}  std={v.std():.3f}  "
          f"min={v.min():+.3f}  max={v.max():+.3f}  |v|.mean={np.abs(v).mean():.3f}")

    print()
    print("─── interpretation ──────────────────────────────────────────")
    ratio = H_mean / uniform_entropy
    n_unique = len(unique_top1)
    if ratio > 0.85:
        print("  HEAD ESSENTIALLY UNIFORM. Entropy near ln(N); the policy head")
        print("  did not move. Suggests gradient-flow break or vanishing signal.")
    elif n_unique <= max(3, n // 20):
        print(f"  MODE COLLAPSE. Only {n_unique} unique argmax across {n} states.")
        print("  Policy emits the same handful of moves regardless of input.")
        print("  Likely cause: targets too peaky to learn (temp=0.1 + 48-sim noise).")
    elif ratio < 0.30:
        print(f"  DETERMINISTIC-BUT-DIVERSE. {n_unique} unique argmax, low entropy.")
        print("  Policy memorised something — probably hybrid-MCTS-shaped behaviour")
        print("  that can't stand alone. Test by training with evaluation_mode: pure_neural.")
    else:
        print("  No textbook signature matched. Mid-range entropy + diverse argmax.")
        print("  Worth comparing argmax to heuristic's choice on the same states.")


if __name__ == "__main__":
    main()
