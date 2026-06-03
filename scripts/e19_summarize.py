#!/usr/bin/env python3
"""E19 dual-arm slope summary.

Reads the color-balanced H2H JSONs written by e19_dualarm.sh (pairs of
<tag>_as_white.json / <tag>_as_black.json) and reports, per arm, the
challenger's combined win-rate vs the FROZEN iter1_ema yardstick at each
iteration — plus a least-squares slope across iterations.

The decision metric (E19): compare the SLOPES. Arm A (seed=iter1_ema) starts
near 50% vs frozen-self; Arm B (seed=sym15-iter1-ema) near 27%. A steeper
B-slope => the symmetric/15ch architecture is the better substrate.

Usage: python scripts/e19_summarize.py h2h_e19
"""
import sys, json, re, glob, os
from collections import defaultdict


def wr_vs_frozen(d_white, d_black):
    """Challenger WR vs frozen, combining the two color assignments.

    d_white: JSON where challenger played WHITE (challenger wins = wins['white']).
    d_black: JSON where challenger played BLACK (challenger wins = wins['black']).
    Draws count as 0.5.
    """
    w = d_white['wins']; b = d_black['wins']
    chal_wins = w['white'] + b['black']
    draws = w['draw'] + b['draw']
    total = sum(w.values()) + sum(b.values())
    if total == 0:
        return None, 0
    return (chal_wins + 0.5 * draws) / total, total


def linfit_slope(xs, ys):
    n = len(xs)
    if n < 2:
        return float('nan')
    mx = sum(xs) / n; my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else float('nan')


def main(d):
    pat = re.compile(r'(arm[AB])_iter(\d+)_as_(white|black)\.json$')
    # arm -> iter -> {'white': data, 'black': data}
    data = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(os.path.join(d, '*.json')):
        m = pat.search(os.path.basename(f))
        if not m:
            continue
        arm, it, color = m.group(1), int(m.group(2)), m.group(3)
        with open(f) as fh:
            data[arm][it][color] = json.load(fh)

    if not data:
        print(f'No H2H JSONs found in {d}/')
        return

    for arm in sorted(data):
        print(f'\n=== {arm} vs frozen iter1_ema (color-balanced) ===')
        xs, ys = [], []
        for it in sorted(data[arm]):
            pair = data[arm][it]
            if 'white' not in pair or 'black' not in pair:
                print(f'  iter{it}: INCOMPLETE (missing a color run)')
                continue
            wr, total = wr_vs_frozen(pair['white'], pair['black'])
            if wr is None:
                continue
            xs.append(it); ys.append(wr)
            bar = '#' * round(wr * 40)
            print(f'  iter{it}: {wr*100:5.1f}%  (n={total:3d})  {bar}')
        if len(xs) >= 2:
            slope = linfit_slope(xs, ys)
            print(f'  SLOPE: {slope*100:+.1f} pp / iteration   (iters {xs[0]}->{xs[-1]})')

    print('\nDecision: a steeper Arm-B slope => symmetric/15ch is the better substrate.')
    print('Any arm whose curve crosses >55% vs frozen iter1_ema => the depth lever wins (proceed to E20).')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'h2h_e19')
