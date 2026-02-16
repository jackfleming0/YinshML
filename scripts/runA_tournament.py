#!/usr/bin/env python3
"""Cross-era tournament for Run A checkpoints."""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cross_era_tournament import load_model, run_match
from collections import defaultdict

def main():
    run_dir = Path('runs/20260214_094223')
    device = 'mps'
    games_per_side = 50

    # Run A checkpoints
    checkpoints = {
        'iter_9': run_dir / 'iteration_9' / 'checkpoint_iteration_9.pt',
        'iter_12': run_dir / 'iteration_12' / 'checkpoint_iteration_12.pt',
        'iter_33': run_dir / 'iteration_33' / 'checkpoint_iteration_33.pt',
        'iter_39': run_dir / 'iteration_39' / 'checkpoint_iteration_39.pt',
        'iter_45': run_dir / 'iteration_45' / 'checkpoint_iteration_45.pt',
    }

    # Load models
    print("Loading models...")
    models = {}
    for name, path in checkpoints.items():
        if path.exists():
            print(f"  Loading {name}...")
            models[name] = load_model(str(path), device)
        else:
            print(f"  Missing: {name}")

    early = ['iter_9', 'iter_12']
    late = ['iter_33', 'iter_39', 'iter_45']

    early = [m for m in early if m in models]
    late = [m for m in late if m in models]

    print(f"\nEarly: {early}")
    print(f"Late: {late}")

    results = []

    print("\n" + "="*60)
    print("CROSS-ERA TOURNAMENT (Run A)")
    print("="*60)

    for e in early:
        for l in late:
            print(f"\n--- {e} vs {l} ---")
            r1, r2 = run_match(models[e], models[l], e, l, games_per_side)
            results.append(r1)
            results.append(r2)

    # Aggregate
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    model_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0})
    for r in results:
        model_stats[r.white_model]['wins'] += r.white_wins
        model_stats[r.white_model]['losses'] += r.black_wins
        model_stats[r.white_model]['draws'] += r.draws
        model_stats[r.black_model]['wins'] += r.black_wins
        model_stats[r.black_model]['losses'] += r.white_wins
        model_stats[r.black_model]['draws'] += r.draws

    print(f"\n{'Model':<12} {'Wins':>6} {'Losses':>6} {'Draws':>6} {'Win%':>8}")
    print("-" * 44)
    for model in sorted(model_stats.keys(), key=lambda m: int(m.split('_')[1])):
        stats = model_stats[model]
        total = stats['wins'] + stats['losses'] + stats['draws']
        win_rate = stats['wins'] / total * 100 if total > 0 else 0
        print(f"{model:<12} {stats['wins']:>6} {stats['losses']:>6} {stats['draws']:>6} {win_rate:>7.1f}%")

    # Head-to-head
    early_wins = late_wins = draws = 0
    for r in results:
        is_early_white = r.white_model in early
        is_late_black = r.black_model in late
        is_late_white = r.white_model in late
        is_early_black = r.black_model in early

        if is_early_white and is_late_black:
            early_wins += r.white_wins
            late_wins += r.black_wins
            draws += r.draws
        elif is_late_white and is_early_black:
            late_wins += r.white_wins
            early_wins += r.black_wins
            draws += r.draws

    total = early_wins + late_wins + draws
    print("\n" + "="*60)
    print("HEAD-TO-HEAD (Early vs Late)")
    print("="*60)
    if total > 0:
        print(f"\nEarly (iter 9, 12): {early_wins} wins ({early_wins/total*100:.1f}%)")
        print(f"Late (iter 33, 39, 45): {late_wins} wins ({late_wins/total*100:.1f}%)")
        print(f"Draws: {draws} ({draws/total*100:.1f}%)")

        if late_wins > early_wins:
            print(f"\n✅ Late models show +{(late_wins-early_wins)/total*100:.1f}% improvement")
        elif early_wins > late_wins:
            print(f"\n⚠️ Early models ahead by +{(early_wins-late_wins)/total*100:.1f}%")
        else:
            print(f"\n⚖️ No significant difference")

if __name__ == '__main__':
    main()
