#!/usr/bin/env python3
"""E24 Phase 1a — per-arm H2H-vs-frozen-iter1_ema slope summary.

Parses the color-balanced H2H JSONs written by scripts/e24_phase1a_sweep.sh
(pairs of  <arm>_iter<N>_as_white.json / <arm>_iter<N>_as_black.json , where
arm = lr3e-5 / lr1e-4 / lr3e-4 / ... ) and reports, per arm, the challenger's
win-rate vs the FROZEN iter1_ema yardstick at each iteration — with a Wilson 95%
CI on the decisive games and a least-squares slope across iterations.

NB: this exists because scripts/e19_summarize.py hardcodes `arm[AB]` filename
tags and won't match E24's `lr<tag>` tags.

Reading (E24): this is the CONFIRMATORY signal — the value-head AUC trend on the
human corpus is PRIMARY (and thin H2H at 3 iters ~ 1 point/arm).
  rising, crossing ~50% and holding >= 45%  -> consistent with GREEN
  declining / gate-reverting                -> degradation
  flat near a low level                     -> no movement

Usage:
  python scripts/e24_summarize_h2h.py h2h_e24
  python scripts/e24_summarize_h2h.py h2h_e24 --arm lr1e-4
"""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

PAT = re.compile(r'(lr.+?)_iter(\d+)_as_(white|black)\.json$')


def challenger_tally(d_white, d_black):
    """(ckpt_wins, frozen_wins, draws, total), combining both color assignments.

    d_white: challenger played WHITE (its wins = wins['white']);
    d_black: challenger played BLACK (its wins = wins['black']).
    """
    w, b = d_white['wins'], d_black['wins']
    ckpt = w['white'] + b['black']
    frozen = w['black'] + b['white']
    draws = w['draw'] + b['draw']
    return ckpt, frozen, draws, ckpt + frozen + draws


def wilson(k, n, z=1.96):
    """Wilson score interval for k successes in n decisive trials."""
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return c - h, c + h


def linfit_slope(xs, ys):
    n = len(xs)
    if n < 2:
        return float('nan')
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else float('nan')


def collect(d):
    """dir -> {arm: {iter: {'white': data, 'black': data}}}."""
    data = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(os.path.join(d, '*.json')):
        m = PAT.search(os.path.basename(f))
        if not m:
            continue
        arm, it, color = m.group(1), int(m.group(2)), m.group(3)
        with open(f) as fh:
            data[arm][it][color] = json.load(fh)
    return data


def summarize(data, only_arm=None):
    out = []
    for arm in sorted(data):
        if only_arm and arm != only_arm:
            continue
        out.append(f"\n=== {arm} vs frozen iter1_ema (color-balanced) ===")
        out.append(f"  {'iter':>4} {'score%':>7} {'decWR%':>7} {'dec 95% CI':>13} {'n':>4}  W-L-D")
        xs, ys = [], []
        for it in sorted(data[arm]):
            pair = data[arm][it]
            if 'white' not in pair or 'black' not in pair:
                out.append(f"  {it:>4}   INCOMPLETE (missing a color run)")
                continue
            ck, fr, dr, tot = challenger_tally(pair['white'], pair['black'])
            if tot == 0:
                continue
            score = (ck + 0.5 * dr) / tot
            dec = ck / (ck + fr) if (ck + fr) else float('nan')
            lo, hi = wilson(ck, ck + fr)
            xs.append(it)
            ys.append(score)
            out.append(f"  {it:>4} {score*100:7.1f} {dec*100:7.1f} "
                       f"[{lo*100:4.0f},{hi*100:4.0f}] {tot:>4}  {ck}-{fr}-{dr}")
        if len(xs) >= 2:
            sl = linfit_slope(xs, ys)
            arrow = "climbing" if sl > 0.01 else ("declining" if sl < -0.01 else "flat")
            out.append(f"  SLOPE: {sl*100:+.1f} pp/iter  ({xs[0]}->{xs[-1]})  [{arrow}]")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", nargs="?", default="h2h_e24")
    ap.add_argument("--arm", default=None, help="only this arm (e.g. lr1e-4)")
    args = ap.parse_args(argv)

    data = collect(args.dir)
    if not data:
        print(f"No E24 H2H JSONs (lr*_iter*_as_white/black.json) found in {args.dir}/")
        return 1
    print(summarize(data, args.arm))
    print("\nscore% = draws-as-0.5; decWR% = wins/(wins+losses) with its Wilson CI.")
    print("PRIMARY signal is the value-head AUC trend (human corpus); this H2H is confirmatory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
