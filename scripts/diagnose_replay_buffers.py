"""Compare replay buffers across runs to test the iter-3-4 drift hypotheses.

Three buffers available locally (all from 15-channel runs):
  - D.2 (2026-05-25):              buggy phase weights, 5 iters
  - B1B2B3 original (2026-05-26):  buggy phase weights, 5 iters
  - B1B2B3 RE-RUN #2 (2026-05-27): correct phase weights, 5 iters (this is the one
                                    that produced the +5 WR jump at iter 2)

The diagnostic answers two related questions:

1. **Phase mix:** how badly did the bug warp the training distribution?
   (We already know the labels were wrong; this confirms it with the FIXED
   decoder applied to the stored states.)

2. **Content quality:** does the buffer show signs of mode collapse / value
   distribution shift / game-length shift that would explain the iter-3-4
   drift OR would degrade D1-partial's value if we pretrain on it?

Per-buffer metrics:
  - True phase distribution (via fixed decode_phase_from_state)
  - move_probs sparsity (mean nonzero entries per row; sparser = more
    deterministic / peaked policies)
  - value distribution (mean, std, % at extreme ±1.0, ±0.667, ±0.333)
  - game-length proxy (move_number distribution; resets to 0 indicate
    new-game boundaries)
  - per-row policy sum (sanity — should be ~1.0 for valid distributions,
    0.0 for terminal-position dummies)

Writes a markdown report so we can review + commit findings.
"""

from __future__ import annotations

import gzip
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add project root so we can import the encoder utility.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yinsh_ml.utils.encoding import decode_phase_from_state


BUFFERS = {
    "D.2 (buggy phases)": "experiments/branchD2_run_2026-05-25/full_run_dir/20260525_041120/replay_buffer.pkl.gz",
    "B1B2B3 original (buggy phases)": "experiments/branchB1B2B3_run_2026-05-26/full_run_dir/20260525_233508/replay_buffer.pkl.gz",
    "B1B2B3 RE-RUN #2 (fixed phases)": "experiments/branchB1B2B3_rerun2_2026-05-27/full_run_dir/20260527_001626/replay_buffer.pkl.gz",
}


def load_buffer(path: str) -> Dict:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def true_phase_mix(buf: Dict, sample_size: int = 5000) -> Dict[str, float]:
    """Run the FIXED phase decoder on a sample of states. The stored
    `phases` field reflects the buggy decoder at write-time; this gives
    the ground truth."""
    n = len(buf["states"])
    idx = np.linspace(0, n - 1, min(sample_size, n), dtype=int)
    counts = Counter()
    for i in idx:
        counts[decode_phase_from_state(buf["states"][i])] += 1
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def stored_phase_mix(buf: Dict) -> Dict[str, float]:
    """Phase labels as stored in the buffer (under the buggy decoder for old runs)."""
    total = len(buf["phases"])
    counts = Counter(buf["phases"])
    return {k: v / total for k, v in counts.items()}


def policy_sparsity(buf: Dict, sample_size: int = 5000) -> Dict[str, float]:
    """How many entries per move_probs row are nonzero? Lower = more
    peaked / deterministic policy."""
    n = len(buf["move_probs"])
    idx = np.linspace(0, n - 1, min(sample_size, n), dtype=int)
    nonzero_counts = np.array([(buf["move_probs"][i] > 0).sum() for i in idx], dtype=float)
    row_sums = np.array([buf["move_probs"][i].astype(np.float32).sum() for i in idx], dtype=float)
    # Separate zero-policy rows (dummies / terminals) from real distributions
    real_mask = row_sums > 0.5
    return {
        "n_sampled": float(len(idx)),
        "frac_real": float(real_mask.mean()),
        "frac_dummy_zero": float((row_sums < 0.01).mean()),
        "nonzero_mean_real": float(nonzero_counts[real_mask].mean()) if real_mask.any() else float("nan"),
        "nonzero_median_real": float(np.median(nonzero_counts[real_mask])) if real_mask.any() else float("nan"),
        "nonzero_p25_real": float(np.percentile(nonzero_counts[real_mask], 25)) if real_mask.any() else float("nan"),
        "nonzero_p75_real": float(np.percentile(nonzero_counts[real_mask], 75)) if real_mask.any() else float("nan"),
        "row_sum_mean_real": float(row_sums[real_mask].mean()) if real_mask.any() else float("nan"),
    }


def value_distribution(buf: Dict) -> Dict[str, float]:
    """Where do value targets cluster? Drift toward extremes (±1.0) means
    games are becoming more decisive. Clustering at 0 means more draws/
    even positions."""
    vals = np.asarray(buf["values"], dtype=np.float32)
    return {
        "n": int(len(vals)),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "pct_at_pos1": float((np.abs(vals - 1.0) < 0.01).mean()),
        "pct_at_neg1": float((np.abs(vals + 1.0) < 0.01).mean()),
        "pct_at_pm_0_667": float((np.abs(np.abs(vals) - 2 / 3) < 0.01).mean()),
        "pct_at_pm_0_333": float((np.abs(np.abs(vals) - 1 / 3) < 0.01).mean()),
        "pct_at_zero": float((np.abs(vals) < 0.01).mean()),
        "pct_extreme_pm1": float((np.abs(vals) > 0.99).mean()),
    }


def game_length_stats(buf: Dict) -> Dict[str, float]:
    """Move-number distribution. The stored move_numbers reset to 0 at
    each new game start, so we can infer game lengths from the resets."""
    mn = np.asarray(buf["move_numbers"], dtype=np.int32)
    # Find game boundaries (where move_number = 0 or decreases)
    # Game starts where move_number = 0
    game_starts = np.where(mn == 0)[0]
    if len(game_starts) < 2:
        return {"n_games_inferred": float(len(game_starts)), "mean_len": float("nan")}
    game_lengths = np.diff(np.concatenate([game_starts, [len(mn)]]))
    return {
        "n_games_inferred": float(len(game_starts)),
        "len_mean": float(game_lengths.mean()),
        "len_median": float(np.median(game_lengths)),
        "len_p25": float(np.percentile(game_lengths, 25)),
        "len_p75": float(np.percentile(game_lengths, 75)),
        "len_min": float(game_lengths.min()),
        "len_max": float(game_lengths.max()),
        "move_number_max": float(mn.max()),
    }


def format_pct(d: Dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.1%}" for k, v in sorted(d.items()))


def format_float(d: Dict[str, float], precision: int = 3) -> str:
    return ", ".join(f"{k}={v:.{precision}f}" for k, v in sorted(d.items()))


def main() -> None:
    print("# Replay Buffer Diagnostics — 3-way comparison\n")
    print("Compares D.2 (buggy), B1B2B3 original (buggy), and B1B2B3 RE-RUN #2 (fixed).")
    print("All three are 15-channel runs that completed 5 iterations of self-play.\n")
    print("True phase mix is computed via the FIXED `decode_phase_from_state` utility,")
    print("which is independent of the labels stored in the buffer at write time.\n")
    print("---\n")

    results: Dict[str, Dict] = {}
    for label, path in BUFFERS.items():
        full_path = Path(path)
        if not full_path.exists():
            print(f"## {label}\n\n  ⚠️ Buffer not found at {path}\n")
            continue
        print(f"## {label}\n")
        print(f"Path: `{path}`\n")
        buf = load_buffer(path)
        n = len(buf["phases"])
        print(f"Total samples: **{n:,}**\n")

        print("### Phase distribution\n")
        stored = stored_phase_mix(buf)
        true_mix = true_phase_mix(buf)
        print("| Source | RING_PLACEMENT | MAIN_GAME | RING_REMOVAL |")
        print("|---|---|---|---|")
        print(f"| Stored (labels at write time) | "
              f"{stored.get('RING_PLACEMENT', 0):.1%} | "
              f"{stored.get('MAIN_GAME', 0):.1%} | "
              f"{stored.get('RING_REMOVAL', 0):.1%} |")
        print(f"| **True (fixed decoder)** | "
              f"**{true_mix.get('RING_PLACEMENT', 0):.1%}** | "
              f"**{true_mix.get('MAIN_GAME', 0):.1%}** | "
              f"**{true_mix.get('RING_REMOVAL', 0):.1%}** |")
        print()

        print("### Policy (move_probs) sparsity\n")
        sp = policy_sparsity(buf)
        print(f"- Sampled {int(sp['n_sampled'])} rows")
        print(f"- Dummy/zero-policy rows: **{sp['frac_dummy_zero']:.1%}** "
              f"(real distributions: {sp['frac_real']:.1%})")
        print(f"- Nonzero entries per REAL row: "
              f"mean=**{sp['nonzero_mean_real']:.1f}**, "
              f"median={sp['nonzero_median_real']:.0f}, "
              f"IQR=[{sp['nonzero_p25_real']:.0f}, {sp['nonzero_p75_real']:.0f}]")
        print(f"- Row-sum mean (real rows): {sp['row_sum_mean_real']:.3f} (should be ~1.0)\n")

        print("### Value-target distribution\n")
        v = value_distribution(buf)
        print(f"- mean={v['mean']:+.4f}, std={v['std']:.4f}")
        print(f"- at +1.0 (white-3): **{v['pct_at_pos1']:.1%}**, "
              f"at -1.0 (black-3): **{v['pct_at_neg1']:.1%}**")
        print(f"- at ±0.667 (margin of 2): {v['pct_at_pm_0_667']:.1%}")
        print(f"- at ±0.333 (margin of 1): {v['pct_at_pm_0_333']:.1%}")
        print(f"- at 0 (tied/early): {v['pct_at_zero']:.1%}")
        print(f"- extreme ±1.0 total: **{v['pct_extreme_pm1']:.1%}**\n")

        print("### Game length (inferred from move_number resets)\n")
        gl = game_length_stats(buf)
        print(f"- inferred game count: {int(gl['n_games_inferred'])}")
        if "len_mean" in gl and not np.isnan(gl["len_mean"]):
            print(f"- length: mean=**{gl['len_mean']:.1f}**, "
                  f"median={gl['len_median']:.0f}, "
                  f"IQR=[{gl['len_p25']:.0f}, {gl['len_p75']:.0f}], "
                  f"range=[{gl['len_min']:.0f}, {gl['len_max']:.0f}]")
            print(f"- move_number max: {gl['move_number_max']:.0f}\n")

        results[label] = {
            "n": n,
            "true_mix": true_mix,
            "stored_mix": stored,
            "sparsity": sp,
            "values": v,
            "game_length": gl,
        }
        print("---\n")

    # Summary comparison
    print("## Summary — what shifts between buggy and fixed?\n")
    if len(results) >= 2:
        print("Direct comparison of the three buffers' key metrics. Bold = notable shift.\n")
        labels = list(results.keys())
        rows = [
            ("Sample count", lambda r: f"{r['n']:,}"),
            ("True MAIN_GAME %", lambda r: f"{r['true_mix'].get('MAIN_GAME', 0):.1%}"),
            ("True RING_PLACEMENT %", lambda r: f"{r['true_mix'].get('RING_PLACEMENT', 0):.1%}"),
            ("Stored MAIN_GAME %", lambda r: f"{r['stored_mix'].get('MAIN_GAME', 0):.1%}"),
            ("Frac dummy/zero rows", lambda r: f"{r['sparsity']['frac_dummy_zero']:.1%}"),
            ("Policy nonzero mean (real)", lambda r: f"{r['sparsity']['nonzero_mean_real']:.1f}"),
            ("Value mean", lambda r: f"{r['values']['mean']:+.4f}"),
            ("Value std", lambda r: f"{r['values']['std']:.4f}"),
            ("Value % extreme ±1.0", lambda r: f"{r['values']['pct_extreme_pm1']:.1%}"),
            ("Value % at zero", lambda r: f"{r['values']['pct_at_zero']:.1%}"),
            ("Inferred game length (mean)", lambda r: f"{r['game_length'].get('len_mean', float('nan')):.1f}"),
        ]
        # Markdown table
        header = "| Metric | " + " | ".join(labels) + " |"
        sep = "|---|" + "|".join("---" for _ in labels) + "|"
        print(header)
        print(sep)
        for name, fn in rows:
            cells = [fn(results[lab]) for lab in labels]
            print(f"| {name} | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    main()
