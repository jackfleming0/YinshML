"""Head-to-head comparison of two measurement runs over the SAME positions.

For each position present in both files, compare key metrics and surface:
- aggregate deltas per sim budget
- per-phase breakdown of how alignment shifted
- positions where the two models materially disagree (high-EV inspection)

Usage:
    python analysis_board/loop/compare.py \
        --a analysis_board/loop/runs/<ts>/measurements_anchor.jsonl \
        --b analysis_board/loop/runs/<ts>/measurements_iter4.jsonl \
        --label-a anchor --label-b iter4 \
        --out-dir analysis_board/loop/runs/<ts>/compare_report/
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("compare")


def load(path: Path) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("ok") or r.get("id") is None:
                continue
            by_id[r["id"]] = r
    return by_id


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, type=Path)
    p.add_argument("--b", required=True, type=Path)
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = args.out_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    a = load(args.a)
    b = load(args.b)
    shared = sorted(set(a.keys()) & set(b.keys()))
    log.info("%d positions in A, %d in B, %d shared", len(a), len(b), len(shared))
    if not shared:
        log.error("no shared positions")
        return

    # Recover sim budgets (assume both runs share them).
    sims = sorted(int(k) for k in a[shared[0]]["mcts"].keys())
    top_sim = max(sims)
    log.info("sim budgets: %s  top=%d", sims, top_sim)

    rows_a = [a[i] for i in shared]
    rows_b = [b[i] for i in shared]

    # --- Aggregate per-budget table ---
    def stats(rows: List[Dict[str, Any]], sim: int) -> Tuple[float, float, float, float, float]:
        key = str(sim)
        ranks = [r["mcts"][key]["rank_of_final_best"] for r in rows
                 if key in r["mcts"] and "error" not in r["mcts"][key]]
        gains = [r["mcts"][key].get("value_gain_over_raw") for r in rows
                 if key in r["mcts"] and "error" not in r["mcts"][key]]
        gains = [g for g in gains if g is not None]
        bvs = [r["mcts"][key]["best_move_value"] for r in rows
               if key in r["mcts"] and "error" not in r["mcts"][key]
               and r["mcts"][key]["best_move_value"] is not None]
        misaligned = sum(1 for r in ranks if r >= 3) / len(ranks) if ranks else 0.0
        costly = sum(1 for g in gains if g >= 0.1) / len(gains) if gains else 0.0
        mean_rank = float(np.mean(ranks)) if ranks else 0.0
        mean_gain = float(np.mean(gains)) if gains else 0.0
        mean_bv = float(np.mean(bvs)) if bvs else 0.0
        return mean_rank, misaligned, mean_gain, costly, mean_bv

    overview_lines = [
        f"## Overview (N={len(shared)})",
        "",
        f"| Budget | mean rank | %misaligned (≥3) | mean value_gain | %costly (≥0.1) | mean best_value |",
        f"|---:|---|---|---|---|---|",
    ]
    for sim in sims:
        ra, ma, ga, ca, ba = stats(rows_a, sim)
        rb, mb, gb, cb, bb = stats(rows_b, sim)
        d_rank = rb - ra
        d_mis = mb - ma
        d_gain = gb - ga
        d_costly = cb - ca
        d_bv = bb - ba
        arrow = lambda x, good_dir="down": (
            f" {'↓' if (x < 0) == (good_dir == 'down') else '↑'}{abs(x):.2f}" if abs(x) > 1e-6 else ""
        )
        overview_lines.append(
            f"| **{sim}** | "
            f"{args.label_a} {ra:.2f} → {args.label_b} {rb:.2f}{arrow(d_rank)} | "
            f"{ma * 100:.1f}% → {mb * 100:.1f}%{arrow(d_mis * 100)} | "
            f"{ga:+.3f} → {gb:+.3f}{arrow(d_gain, 'down')} | "
            f"{ca * 100:.1f}% → {cb * 100:.1f}%{arrow(d_costly * 100)} | "
            f"{ba:+.3f} → {bb:+.3f}{arrow(d_bv, 'up')} |"
        )

    # --- Per-phase rank-of-final-best at top sim ---
    by_phase_a: Dict[str, List[int]] = defaultdict(list)
    by_phase_b: Dict[str, List[int]] = defaultdict(list)
    for r in rows_a:
        key = str(top_sim)
        if key in r["mcts"] and "error" not in r["mcts"][key]:
            by_phase_a[r["phase"]].append(r["mcts"][key]["rank_of_final_best"])
    for r in rows_b:
        key = str(top_sim)
        if key in r["mcts"] and "error" not in r["mcts"][key]:
            by_phase_b[r["phase"]].append(r["mcts"][key]["rank_of_final_best"])

    phase_lines = [
        f"## rank-of-final-best by phase at {top_sim} sims",
        "",
        f"| Phase | N | {args.label_a} mean | {args.label_b} mean | Δ | {args.label_a} %top1 | {args.label_b} %top1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for phase in sorted(set(by_phase_a) | set(by_phase_b)):
        a_ranks = by_phase_a.get(phase, [])
        b_ranks = by_phase_b.get(phase, [])
        n = min(len(a_ranks), len(b_ranks))
        if n == 0:
            continue
        ma = float(np.mean(a_ranks))
        mb = float(np.mean(b_ranks))
        pa1 = sum(1 for r in a_ranks if r == 0) / len(a_ranks) * 100
        pb1 = sum(1 for r in b_ranks if r == 0) / len(b_ranks) * 100
        d = mb - ma
        d_str = f"{d:+.2f}"
        phase_lines.append(
            f"| {phase} | {n} | {ma:.2f} | {mb:.2f} | {d_str} | {pa1:.0f}% | {pb1:.0f}% |"
        )

    # --- Per-position deltas: which positions did B improve / regress on? ---
    top_key = str(top_sim)
    deltas = []
    for pid in shared:
        ra = a[pid]
        rb = b[pid]
        if top_key not in ra["mcts"] or top_key not in rb["mcts"]:
            continue
        if "error" in ra["mcts"][top_key] or "error" in rb["mcts"][top_key]:
            continue
        d_rank = rb["mcts"][top_key]["rank_of_final_best"] - ra["mcts"][top_key]["rank_of_final_best"]
        ga = ra["mcts"][top_key].get("value_gain_over_raw")
        gb = rb["mcts"][top_key].get("value_gain_over_raw")
        d_gain = (gb - ga) if (ga is not None and gb is not None) else None
        deltas.append({
            "id": pid,
            "phase": ra["phase"],
            "move_number": ra.get("meta", {}).get("move_number"),
            "n_legal": ra["n_legal_moves"],
            "rank_a": ra["mcts"][top_key]["rank_of_final_best"],
            "rank_b": rb["mcts"][top_key]["rank_of_final_best"],
            "d_rank": d_rank,
            "gain_a": ga,
            "gain_b": gb,
            "d_gain": d_gain,
        })

    # B improved most over A (rank dropped, i.e., d_rank negative AND was misaligned)
    improvements = [d for d in deltas if d["rank_a"] >= 3 and d["d_rank"] <= -3]
    regressions = [d for d in deltas if d["rank_a"] <= 1 and d["d_rank"] >= 5]

    improvements.sort(key=lambda d: d["d_rank"])
    regressions.sort(key=lambda d: -d["d_rank"])

    def render_top(rows, label):
        out = [
            f"## {label}",
            "",
            f"| id | phase | move # | n_legal | rank ({args.label_a}) | rank ({args.label_b}) | Δ rank | gain ({args.label_a}) | gain ({args.label_b}) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for d in rows[:10]:
            ga = f"{d['gain_a']:+.3f}" if isinstance(d['gain_a'], (int, float)) else "—"
            gb = f"{d['gain_b']:+.3f}" if isinstance(d['gain_b'], (int, float)) else "—"
            out.append(
                f"| `{d['id']}` | {d['phase']} | {d['move_number'] or '—'} | "
                f"{d['n_legal']} | {d['rank_a']} | {d['rank_b']} | "
                f"{d['d_rank']:+d} | {ga} | {gb} |"
            )
        return "\n".join(out)

    # --- Charts ---
    # Scatter: rank_a vs rank_b
    fig, ax = plt.subplots(figsize=(6, 6))
    xa = [d["rank_a"] for d in deltas]
    xb = [d["rank_b"] for d in deltas]
    ax.scatter(xa, xb, alpha=0.5, s=22, color="#2563eb")
    lim = max(max(xa, default=0), max(xb, default=0)) + 1
    ax.plot([0, lim], [0, lim], color="#999", linestyle="--", linewidth=1, label="no change")
    ax.set_xlabel(f"rank_of_final_best — {args.label_a}")
    ax.set_ylabel(f"rank_of_final_best — {args.label_b}")
    ax.set_title(f"Per-position alignment shift at {top_sim} sims")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts_dir / "rank_scatter.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Histograms of rank-of-final-best, overlaid
    fig, ax = plt.subplots(figsize=(8, 4))
    ranks_a_top = [r["mcts"][top_key]["rank_of_final_best"] for r in rows_a
                   if top_key in r["mcts"] and "error" not in r["mcts"][top_key]]
    ranks_b_top = [r["mcts"][top_key]["rank_of_final_best"] for r in rows_b
                   if top_key in r["mcts"] and "error" not in r["mcts"][top_key]]
    bins = np.arange(0, max(max(ranks_a_top, default=0), max(ranks_b_top, default=0)) + 2) - 0.5
    ax.hist(ranks_a_top, bins=bins, alpha=0.6, color="#2563eb", label=args.label_a)
    ax.hist(ranks_b_top, bins=bins, alpha=0.6, color="#b91c1c", label=args.label_b)
    ax.axvline(2.5, color="#b45309", linestyle="--", linewidth=1, label="rank≥3 (misaligned)")
    ax.set_xlabel("rank_of_final_best")
    ax.set_ylabel("# positions")
    ax.set_title(f"Alignment distribution at {top_sim} sims")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts_dir / "rank_distribution.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Value gain histograms, overlaid
    fig, ax = plt.subplots(figsize=(8, 4))
    gains_a_top = [r["mcts"][top_key]["value_gain_over_raw"] for r in rows_a
                   if top_key in r["mcts"] and "error" not in r["mcts"][top_key]
                   and r["mcts"][top_key]["value_gain_over_raw"] is not None]
    gains_b_top = [r["mcts"][top_key]["value_gain_over_raw"] for r in rows_b
                   if top_key in r["mcts"] and "error" not in r["mcts"][top_key]
                   and r["mcts"][top_key]["value_gain_over_raw"] is not None]
    bins = np.linspace(
        min(min(gains_a_top), min(gains_b_top)),
        max(max(gains_a_top), max(gains_b_top)),
        25,
    )
    ax.hist(gains_a_top, bins=bins, alpha=0.6, color="#2563eb", label=args.label_a)
    ax.hist(gains_b_top, bins=bins, alpha=0.6, color="#b91c1c", label=args.label_b)
    ax.axvline(0.1, color="#b45309", linestyle="--", linewidth=1, label="costly threshold")
    ax.axvline(0, color="#999", linestyle="-", linewidth=0.7)
    ax.set_xlabel("value_gain_over_raw")
    ax.set_ylabel("# positions")
    ax.set_title(f"Cost-of-misalignment distribution at {top_sim} sims")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts_dir / "gain_distribution.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- Compose markdown report ---
    md = [
        f"# Comparison: {args.label_a} vs {args.label_b}",
        "",
        f"- A: `{args.a}` ({args.label_a})",
        f"- B: `{args.b}` ({args.label_b})",
        f"- Shared positions: {len(shared)}",
        f"- Sim budgets: {sims}",
        f"- Top budget: {top_sim}",
        "",
        *overview_lines,
        "",
        *phase_lines,
        "",
        f"## {args.label_b} improves on {args.label_a} (top 10)",
        f"",
        f"Positions where {args.label_a} was misaligned (rank≥3) AND {args.label_b}'s rank dropped by ≥3.",
        "",
        render_top(improvements, f"{args.label_b} fixed these"),
        "",
        f"## {args.label_b} regresses from {args.label_a} (top 10)",
        "",
        f"Positions where {args.label_a} was aligned (rank≤1) AND {args.label_b}'s rank jumped by ≥5.",
        "",
        render_top(regressions, f"{args.label_b} broke these"),
        "",
        "## Charts",
        "",
        "![rank scatter](charts/rank_scatter.png)",
        "",
        "![rank distribution](charts/rank_distribution.png)",
        "",
        "![gain distribution](charts/gain_distribution.png)",
        "",
    ]

    out_path = args.out_dir / "report.md"
    out_path.write_text("\n".join(md))
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
