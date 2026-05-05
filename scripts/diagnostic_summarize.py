#!/usr/bin/env python3
"""Generate summary.md from diagnostic protocol artifacts.

Reads the eval JSONs and training metrics from a diagnostic run and emits
a structured markdown report that walks the H1-H5 hypothesis table from
INVESTIGATION_PLAN.md with the actual run's data, emitting a verdict per
hypothesis.

Usage:
    python scripts/diagnostic_summarize.py \\
        --output-dir diagnostic_output/<TIMESTAMP> \\
        > diagnostic_output/<TIMESTAMP>/summary.md
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"WARN: failed to load {path}: {e}", file=sys.stderr)
        return None


def per_color_dominance(pair: dict) -> float:
    """Return |a_W_wins - a_B_wins| / games_per_side. >0.7 means white-wins-pattern."""
    n_per_side = (pair['a_wins'] + pair['b_wins'] + pair['draws']) // 2
    if n_per_side == 0:
        return 0.0
    a_w = pair.get('a_white_wins', 0)
    a_b = pair.get('a_black_wins', 0)
    return abs((a_w - a_b) / n_per_side)


def fraction_white_wins_patterns(eval_json: dict) -> tuple[int, int]:
    """Return (count_with_pattern, total_pairs)."""
    pairs = eval_json.get("pairs", [])
    flagged = sum(1 for p in pairs if per_color_dominance(p) > 0.7)
    return flagged, len(pairs)


def best_iter_vs_final(eval_json: dict) -> Optional[dict]:
    """Find the most asymmetric pair where smaller iter beats larger one — the
    'iter-3 crushes iter-19' pattern from B1."""
    pairs = eval_json.get("pairs", [])
    candidates = []
    for p in pairs:
        try:
            a_iter = int(p['a_label'].split('_')[1])
            b_iter = int(p['b_label'].split('_')[1])
        except (ValueError, IndexError):
            continue
        # Looking for a (smaller, larger) pair where a (smaller) crushes b (larger)
        if a_iter < b_iter and p['a_score'] >= 0.7:
            candidates.append(p)
    candidates.sort(key=lambda p: -p['a_score'])
    return candidates[0] if candidates else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory containing eval_h2h_*.json + heuristic_sanity.log")
    args = parser.parse_args()

    out = args.output_dir
    h2h_t0 = load_json(out / "eval_h2h_temp0.json")
    h2h_t05 = load_json(out / "eval_h2h_temp0.5.json")
    h2h_mcts = load_json(out / "eval_h2h_mcts.json")

    print("# Diagnostic Run Summary")
    print()
    print(f"Output dir: `{out}`")
    print()

    # ---------- Hypothesis table ----------
    print("## Hypothesis verdicts")
    print()
    print("| ID | Hypothesis | Evidence | Verdict |")
    print("|---|---|---|---|")

    # H1 — heuristic-vs-heuristic
    sanity_log = out / "heuristic_sanity.log"
    h1_evidence = "log not found"
    h1_verdict = "?"
    if sanity_log.exists():
        text = sanity_log.read_text()
        try:
            # Parse the SUMMARY block: "WHITE wins: N | BLACK wins: M | DRAW/inconclusive: K"
            w = int(text.split("WHITE wins:")[1].split("|")[0].strip())
            b = int(text.split("BLACK wins:")[1].split("|")[0].strip())
            total_decisive = w + b
            black_share = b / total_decisive if total_decisive else 0
            h1_evidence = f"heuristic-vs-heuristic: {w}W / {b}B (of {total_decisive} decisive)"
            h1_verdict = "REFUTED (heuristic plays balanced)" if black_share >= 0.3 else \
                         "SUPPORTED (heuristic is white-biased)"
        except Exception:
            h1_evidence = "parse error"
    print(f"| H1 | Heuristic itself is offense-only | {h1_evidence} | {h1_verdict} |")

    # H2 — temp=0.5 fixes white-wins
    if h2h_t0 and h2h_t05:
        flagged_t0, total_t0 = fraction_white_wins_patterns(h2h_t0)
        flagged_t05, total_t05 = fraction_white_wins_patterns(h2h_t05)
        h2_evidence = f"temp=0: {flagged_t0}/{total_t0} pairs flagged | temp=0.5: {flagged_t05}/{total_t05} pairs flagged"
        if flagged_t0 > 0 and flagged_t05 == 0:
            h2_verdict = "SUPPORTED (deterministic-argmax brittleness)"
        elif flagged_t0 == 0:
            h2_verdict = "N/A (no pattern observed)"
        else:
            h2_verdict = "REFUTED (pattern persists with temperature)"
    else:
        h2_evidence = "missing eval JSONs"
        h2_verdict = "?"
    print(f"| H2 | White-wins is deterministic-argmax artifact | {h2_evidence} | {h2_verdict} |")

    # H3 — pattern persists at temp=0.5 (real policy collapse)
    if h2h_t05:
        flagged, total = fraction_white_wins_patterns(h2h_t05)
        h3_evidence = f"temp=0.5: {flagged}/{total} pairs flagged"
        h3_verdict = "SUPPORTED (real policy bias)" if flagged > total // 2 else "REFUTED"
    else:
        h3_evidence = "missing"
        h3_verdict = "?"
    print(f"| H3 | Bias persists with stochastic policy | {h3_evidence} | {h3_verdict} |")

    # H4 — pattern persists with MCTS (production-realistic)
    if h2h_mcts:
        flagged, total = fraction_white_wins_patterns(h2h_mcts)
        h4_evidence = f"MCTS: {flagged}/{total} pairs flagged"
        h4_verdict = "SUPPORTED (production play affected)" if flagged > 0 else \
                     "REFUTED (production play OK)"
    else:
        h4_evidence = "missing"
        h4_verdict = "?"
    print(f"| H4 | Bias persists under MCTS | {h4_evidence} | {h4_verdict} |")

    # H5 — early iter crushes later
    if h2h_mcts:
        crusher = best_iter_vs_final(h2h_mcts)
        if crusher:
            h5_evidence = (f"{crusher['a_label']} beats {crusher['b_label']} "
                           f"{crusher['a_wins']}/{crusher['a_wins']+crusher['b_wins']+crusher['draws']} under MCTS")
            h5_verdict = "SUPPORTED (network regresses over training)"
        else:
            h5_evidence = "no early-crushes-late pair found under MCTS"
            h5_verdict = "REFUTED"
    else:
        h5_evidence = "missing"
        h5_verdict = "?"
    print(f"| H5 | Network regresses over training | {h5_evidence} | {h5_verdict} |")

    # ---------- Detailed tables ----------
    for label, ej in [("temp=0.0 (deterministic argmax)", h2h_t0),
                       ("temp=0.5 (stochastic)", h2h_t05),
                       ("MCTS", h2h_mcts)]:
        print()
        print(f"## H2H: {label}")
        if not ej:
            print("(missing)")
            continue
        pairs = ej.get("pairs", [])
        if not pairs:
            print("(no pairs)")
            continue
        print()
        print("| pair | a_score | a_W_wins | a_B_wins | per-color flag |")
        print("|---|---|---|---|---|")
        for p in pairs:
            n_per_side = (p['a_wins'] + p['b_wins'] + p['draws']) // 2
            a_w = p.get('a_white_wins', 0)
            a_b = p.get('a_black_wins', 0)
            dom = per_color_dominance(p)
            flag = "⚠ WHITE-WINS" if dom > 0.7 else ""
            print(f"| {p['a_label']} vs {p['b_label']} | {p['a_score']:.3f} | "
                  f"{a_w}/{n_per_side} | {a_b}/{n_per_side} | {flag} |")

    # ---------- Recommendation ----------
    print()
    print("## Recommendation")
    print()
    if h2h_mcts:
        flagged_mcts, _ = fraction_white_wins_patterns(h2h_mcts)
        if flagged_mcts > 0:
            print("- **HIGH PRIORITY**: White-wins pattern persists under MCTS. "
                  "Production play is affected. Investigate value-head architecture "
                  "and training pipeline for color-asymmetric signal before further tuning.")
        elif h2h_t0:
            flagged_t0, _ = fraction_white_wins_patterns(h2h_t0)
            if flagged_t0 > 0:
                print("- **MEDIUM PRIORITY**: White-wins is a deterministic-argmax artifact "
                      "but does not affect MCTS-based play. Continue tuning, but always "
                      "evaluate with MCTS or temperature>0; flag any tooling that uses "
                      "raw-policy argmax as unreliable.")
            else:
                print("- **LOW PRIORITY / NO REPRODUCTION**: No white-wins pattern observed. "
                      "B1/B2 pattern may have been transient. Resume tuning at full scale.")
    else:
        print("- (MCTS eval missing — cannot conclude)")


if __name__ == "__main__":
    main()
