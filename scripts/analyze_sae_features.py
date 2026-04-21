"""Post-hoc hand-labeling helper for Track B §8 SAE probe output.

The SAE probe (scripts/run_sae_probe.py) emits a feature_report.json with
per-feature top-K positions. This script converts those indices into
board-level summaries the reader can use to assign concept labels:

  * ring counts per player
  * marker counts per player
  * game phase distribution
  * move-number distribution
  * network-value distribution
  * any player-score differential signals

Writes two artifacts:
  * `feature_labels.json` — structured per-feature aggregates for scripting
  * `feature_labels.md` — human-readable report listing the top-N features
    sorted by "most distinctive" (features whose top-K positions are the
    most homogeneous on some board axis).

Usage:
  python scripts/analyze_sae_features.py \
    --probe-dir analysis_output/sae_probe/bakeoff_challenger_iter9_v2 \
    --top-features 40
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe-dir', type=str, required=True,
                   help='Directory containing SAE probe artifacts.')
    p.add_argument('--top-features', type=int, default=40,
                   help='How many of the most-distinctive features to report.')
    p.add_argument('--min-firing-rate', type=float, default=0.005,
                   help='Skip features that fire on < N fraction of positions.')
    return p.parse_args()


def board_stats(state: np.ndarray) -> Dict:
    """Summarize a 6-channel encoded state into human-readable counts.

    Channels (see CLAUDE.md):
      0: White rings, 1: Black rings, 2: White markers, 3: Black markers,
      4: Valid moves, 5: Game phase (scalar, broadcast to all cells).
    """
    white_rings = int((state[0] > 0).sum())
    black_rings = int((state[1] > 0).sum())
    white_markers = int((state[2] > 0).sum())
    black_markers = int((state[3] > 0).sum())
    valid_moves = int((state[4] > 0).sum())
    # Phase: mean of channel 5 (scalar, broadcast uniformly)
    phase_value = float(state[5].mean())
    if phase_value < 0.2:
        phase = 'RING_PLACEMENT'
    elif phase_value < 0.6:
        phase = 'MAIN_GAME'
    else:
        phase = 'RING_REMOVAL'
    return {
        'white_rings': white_rings,
        'black_rings': black_rings,
        'white_markers': white_markers,
        'black_markers': black_markers,
        'total_markers': white_markers + black_markers,
        'marker_diff': white_markers - black_markers,
        'valid_moves': valid_moves,
        'phase': phase,
    }


def aggregate_feature(top_indices: List[int], states: np.ndarray,
                      net_values: np.ndarray, outcomes: np.ndarray,
                      move_numbers: np.ndarray) -> Dict:
    """Compute per-feature aggregate statistics over its top-K positions."""
    per_pos = [board_stats(states[i]) for i in top_indices]
    if not per_pos:
        return {}
    n = len(per_pos)
    phase_counter = Counter(p['phase'] for p in per_pos)
    phase_frac = {k: v / n for k, v in phase_counter.items()}

    def col(key):
        return np.array([p[key] for p in per_pos], dtype=np.float32)

    def summary(a):
        return {
            'mean': float(a.mean()),
            'std': float(a.std()),
            'min': float(a.min()),
            'max': float(a.max()),
        }

    net_v = np.array([net_values[i] for i in top_indices], dtype=np.float32)
    actual_v = np.array([outcomes[i] for i in top_indices], dtype=np.float32)
    moves = np.array([move_numbers[i] for i in top_indices], dtype=np.int32)

    return {
        'n_positions': n,
        'phase_fractions': phase_frac,
        'white_rings': summary(col('white_rings')),
        'black_rings': summary(col('black_rings')),
        'white_markers': summary(col('white_markers')),
        'black_markers': summary(col('black_markers')),
        'marker_diff': summary(col('marker_diff')),
        'valid_moves': summary(col('valid_moves')),
        'move_number': summary(moves.astype(np.float32)),
        'network_value': summary(net_v),
        'actual_outcome': summary(actual_v),
    }


def distinctiveness_score(agg: Dict) -> float:
    """Heuristic for how homogeneous the top-K positions are — higher means
    the feature reliably fires on a specific phase / ring-count / etc.
    Sums inverse-std signals across the board-axis summaries."""
    if not agg:
        return 0.0
    # Peak phase fraction (1.0 = fires only on one phase)
    peak_phase = max(agg['phase_fractions'].values())
    # Low std on ring/marker counts = consistent position type
    ring_std = agg['white_rings']['std'] + agg['black_rings']['std']
    marker_std = agg['white_markers']['std'] + agg['black_markers']['std']
    # Ring counts are 0..5, so std <0.5 is very tight; markers can be 0..50
    ring_component = max(0.0, 1.0 - ring_std / 2.0)
    marker_component = max(0.0, 1.0 - marker_std / 10.0)
    return peak_phase * 0.5 + ring_component * 0.25 + marker_component * 0.25


def render_feature_md(fid: int, firing_rate: float, mean_act: float,
                      top_k: int, agg: Dict, score: float) -> List[str]:
    """Produce a human-readable markdown block for one feature."""
    lines = [
        f"### Feature #{fid}",
        f"- **Firing rate**: {firing_rate:.2%} of positions",
        f"- **Mean activation (when firing)**: {mean_act:.3f}",
        f"- **Distinctiveness**: {score:.3f} (higher = more specific)",
        f"- **Top-K analyzed**: {top_k}",
        f"- **Phase distribution**: " + ", ".join(
            f"{k}={v:.0%}" for k, v in sorted(agg['phase_fractions'].items(), key=lambda kv: -kv[1])
        ),
        f"- **Rings**: W={agg['white_rings']['mean']:.1f}±{agg['white_rings']['std']:.1f}, "
        f"B={agg['black_rings']['mean']:.1f}±{agg['black_rings']['std']:.1f}",
        f"- **Markers**: W={agg['white_markers']['mean']:.1f}±{agg['white_markers']['std']:.1f}, "
        f"B={agg['black_markers']['mean']:.1f}±{agg['black_markers']['std']:.1f}, "
        f"diff={agg['marker_diff']['mean']:+.1f}±{agg['marker_diff']['std']:.1f}",
        f"- **Valid moves**: {agg['valid_moves']['mean']:.1f}±{agg['valid_moves']['std']:.1f}",
        f"- **Move number**: {agg['move_number']['mean']:.1f}±{agg['move_number']['std']:.1f} "
        f"(range {agg['move_number']['min']:.0f}..{agg['move_number']['max']:.0f})",
        f"- **Network value**: {agg['network_value']['mean']:+.3f}±{agg['network_value']['std']:.3f}",
        f"- **Actual outcome**: {agg['actual_outcome']['mean']:+.3f}±{agg['actual_outcome']['std']:.3f}",
        f"- **Concept label (hand-fill)**: _TODO_",
        "",
    ]
    return lines


def main() -> int:
    args = parse_args()
    probe_dir = Path(args.probe_dir)

    with open(probe_dir / 'feature_report.json') as f:
        report = json.load(f)

    states = np.load(probe_dir / 'positions_states.npy')
    net_values = np.load(probe_dir / 'positions_network_values.npy')
    outcomes = np.load(probe_dir / 'positions_outcomes.npy')
    move_numbers = np.load(probe_dir / 'positions_move_numbers.npy')

    features = report['features']
    print(f"Loaded {len(features)} features; {report['dead_feature_count']} dead, "
          f"{report['sparse_feature_count']} sparse, {report['dense_feature_count']} dense.")

    # Aggregate per-feature stats; filter dead / too-rare features.
    live = []
    for f in features:
        if f['firing_rate'] < args.min_firing_rate:
            continue
        if not f['top_k_indices']:
            continue
        agg = aggregate_feature(
            f['top_k_indices'], states, net_values, outcomes, move_numbers
        )
        score = distinctiveness_score(agg)
        live.append({
            'feature_id': f['feature_id'],
            'firing_rate': f['firing_rate'],
            'mean_activation': f['mean_activation'],
            'max_activation': f['max_activation'],
            'top_k': len(f['top_k_indices']),
            'aggregate': agg,
            'distinctiveness': score,
        })

    # Sort by distinctiveness descending.
    live.sort(key=lambda x: -x['distinctiveness'])

    # Persist structured + markdown outputs.
    with open(probe_dir / 'feature_labels.json', 'w') as f:
        json.dump({
            'num_live_features': len(live),
            'features': live,
        }, f, indent=2)

    top = live[:args.top_features]
    md_lines = [
        "# SAE feature-label worksheet",
        "",
        f"**Source**: `{probe_dir}/feature_report.json`",
        f"**Total features**: {report['num_features']} "
        f"({report['dead_feature_count']} dead, "
        f"{report['sparse_feature_count']} sparse, "
        f"{report['dense_feature_count']} dense)",
        f"**Live (>= {args.min_firing_rate:.1%} firing rate)**: {len(live)}",
        f"**Shown below**: top {len(top)} by distinctiveness score",
        "",
        "## Reading the report",
        "Each feature's top-20 positions were aggregated into distributional",
        "summaries. Phase, ring counts, marker counts, and value stats are",
        "the main cues for labeling. Fill in the _Concept label_ line per",
        "feature based on the pattern — e.g. \"ring-placement early\", \"black",
        "ahead on markers\", \"endgame with no free rings\", etc.",
        "",
        "## Features",
        "",
    ]
    for feat in top:
        md_lines.extend(render_feature_md(
            feat['feature_id'],
            feat['firing_rate'],
            feat['mean_activation'],
            feat['top_k'],
            feat['aggregate'],
            feat['distinctiveness'],
        ))

    with open(probe_dir / 'feature_labels.md', 'w') as f:
        f.write('\n'.join(md_lines))

    print(f"Wrote {probe_dir/'feature_labels.json'} ({len(live)} live features)")
    print(f"Wrote {probe_dir/'feature_labels.md'} (top {len(top)} features)")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
