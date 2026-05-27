"""Aggregate measure.py output into a markdown report + charts.

Reads ``measurements.jsonl`` produced by measure.py and produces:
- ``report.md`` — narrative summary with key tables
- ``charts/*.png`` — matplotlib plots (entropy distribution, rank-of-final-best
  histogram, value-vs-sims convergence)

Usage:
    python analysis_board/loop/report.py \
        --measurements analysis_board/loop/runs/<ts>/measurements.jsonl \
        --out-dir analysis_board/loop/runs/<ts>/
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("report")


def load(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return [r for r in rows if r.get("ok")]


def fmt_row(values: Iterable, widths: Iterable[int]) -> str:
    return " | ".join(str(v).ljust(w) for v, w in zip(values, widths))


def overview_table(rows: List[Dict[str, Any]], sims: List[int]) -> str:
    """Per-budget summary including value-cost-of-misalignment."""
    lines = [
        "| Budget | N | mean root_q | mean best_move_value | mean rank_of_final_best | %misaligned (rank≥3) | mean value_gain_over_raw | %costly-misalignment (gain≥0.1) | mean wall (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    # Raw policy row first
    raw_values = [r["raw_policy"]["value"] for r in rows]
    raw_best = [r["raw_policy"]["best_move_value"] for r in rows
                if r["raw_policy"]["best_move_value"] is not None]
    raw_best_mean = f"{np.mean(raw_best):+.3f}" if raw_best else "—"
    lines.append(
        f"| raw policy | {len(rows)} | {np.mean(raw_values):+.3f} | {raw_best_mean} | — | — | — | — | — |"
    )
    for s in sims:
        if s <= 0:
            continue
        key = str(s)
        budget_rows = [r for r in rows if key in r["mcts"] and "error" not in r["mcts"][key]]
        if not budget_rows:
            continue
        rqs = [r["mcts"][key]["root_q"] for r in budget_rows]
        bvs = [r["mcts"][key]["best_move_value"] for r in budget_rows
               if r["mcts"][key]["best_move_value"] is not None]
        ranks = [r["mcts"][key]["rank_of_final_best"] for r in budget_rows]
        misaligned = sum(1 for r in ranks if r >= 3) / len(ranks)
        gaps = [r["mcts"][key]["gap_1_2"] for r in budget_rows]
        walls = [r["mcts"][key]["wall_seconds"] for r in budget_rows]
        bv_mean = f"{np.mean(bvs):+.3f}" if bvs else "—"
        gains = [r["mcts"][key].get("value_gain_over_raw") for r in budget_rows]
        gains = [g for g in gains if g is not None]
        gain_mean = f"{np.mean(gains):+.3f}" if gains else "—"
        # "costly misalignment": value loss ≥ 0.1 from following raw policy
        costly = sum(1 for g in gains if g >= 0.1) / len(gains) if gains else 0.0
        lines.append(
            f"| {s} sims | {len(budget_rows)} | "
            f"{np.mean(rqs):+.3f} | {bv_mean} | "
            f"{np.mean(ranks):4.2f} | {misaligned * 100:.1f}% | "
            f"{gain_mean} | {costly * 100:.1f}% | "
            f"{np.mean(walls):.2f} |"
        )
    return "\n".join(lines)


def rank_histogram_per_phase(rows: List[Dict[str, Any]], top_sim: int) -> str:
    """Distribution of rank-of-final-best at the highest sim budget, by phase."""
    by_phase = defaultdict(list)
    for r in rows:
        key = str(top_sim)
        if key not in r["mcts"] or "error" in r["mcts"][key]:
            continue
        by_phase[r["phase"]].append(r["mcts"][key]["rank_of_final_best"])

    if not by_phase:
        return "_(no MCTS data at top budget)_"

    lines = [
        "| Phase | N | mean | median | %top1 | %top3 | %rank≥5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for phase, ranks in sorted(by_phase.items()):
        n = len(ranks)
        mean = float(np.mean(ranks))
        median = float(np.median(ranks))
        pct_top1 = sum(1 for r in ranks if r == 0) / n * 100
        pct_top3 = sum(1 for r in ranks if r < 3) / n * 100
        pct_misaligned = sum(1 for r in ranks if r >= 5) / n * 100
        lines.append(
            f"| {phase} | {n} | {mean:.2f} | {median:.1f} | {pct_top1:.1f}% | {pct_top3:.1f}% | {pct_misaligned:.1f}% |"
        )
    return "\n".join(lines)


def misaligned_examples(rows: List[Dict[str, Any]], top_sim: int, k: int = 10) -> str:
    """Top-K worst-misaligned positions for hand-inspection."""
    key = str(top_sim)
    candidates = []
    for r in rows:
        if key not in r["mcts"] or "error" in r["mcts"][key]:
            continue
        rk = r["mcts"][key]["rank_of_final_best"]
        if rk >= 1:  # not the top-1 in raw policy
            candidates.append((rk, r))
    candidates.sort(key=lambda x: -x[0])
    candidates = candidates[:k]
    if not candidates:
        return "_(no misaligned positions at this budget)_"
    lines = [
        "| id | phase | move # | n_legal | raw entropy | rank | raw prob | mcts prob | mcts root_q | best_value |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rk, r in candidates:
        meta = r.get("meta", {})
        raw = r["raw_policy"]
        m = r["mcts"][key]
        # Find raw-prob of the move that MCTS picked
        mcts_top_idx = m["top"][0]["idx"]
        raw_prob_of_mcts_top = next(
            (t["prob"] for t in raw["top"] if t["idx"] == mcts_top_idx),
            "—",
        )
        if isinstance(raw_prob_of_mcts_top, float):
            raw_prob_str = f"{raw_prob_of_mcts_top * 100:.1f}%"
        else:
            raw_prob_str = "<top-K"
        bmv = m.get("best_move_value")
        bmv_str = f"{bmv:+.3f}" if isinstance(bmv, (int, float)) else "—"
        lines.append(
            f"| `{r['id']}` | {r['phase']} | {meta.get('move_number', '—')} | "
            f"{r['n_legal_moves']} | {raw['entropy']:.2f} | {rk} | "
            f"{raw_prob_str} | {m['top'][0]['prob'] * 100:.1f}% | "
            f"{m['root_q']:+.3f} | {bmv_str} |"
        )
    return "\n".join(lines)


def value_gain_chart(rows: List[Dict[str, Any]], top_sim: int, out_path: Path) -> None:
    """Histogram of value_gain_over_raw at the top sim budget."""
    key = str(top_sim)
    gains = [r["mcts"][key].get("value_gain_over_raw") for r in rows
             if key in r["mcts"] and "error" not in r["mcts"][key]]
    gains = [g for g in gains if g is not None]
    if not gains:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(gains, bins=25, color="#15803d", edgecolor="#166534")
    ax.axvline(0, color="#999", linestyle="--", linewidth=1, label="no value cost")
    ax.axvline(0.1, color="#b45309", linestyle="--", linewidth=1, label="costly threshold")
    ax.set_xlabel(f"value_gain_over_raw at {top_sim} sims  (best_value(MCTS top) − best_value(raw policy top))")
    ax.set_ylabel("# positions")
    ax.set_title("Cost of policy misalignment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main_game_breakdown(
    rows: List[Dict[str, Any]],
    top_sim: int,
    bucket_key: str,
    bucket_edges: List[int],
    label: str,
) -> str:
    """Bin MAIN_GAME-only positions by some integer key (move_number,
    n_legal_moves) and report rank-of-final-best stats per bucket.

    Restricted to MAIN_GAME because the per-phase split already shows
    RING_REMOVAL / ROW_COMPLETION at 87% top-1 — including them here would
    create an artifact where the "late-move" bucket looks well-aligned just
    because capture sequences happen late. Stripping the phase signal lets
    us isolate the "competence vs game time" question Jack raised.
    """
    key = str(top_sim)
    filtered = [r for r in rows
                if r["phase"] == "MAIN_GAME"
                and key in r["mcts"]
                and "error" not in r["mcts"][key]]

    def bucket_label(value: int) -> str:
        for i in range(len(bucket_edges) - 1):
            lo, hi = bucket_edges[i], bucket_edges[i + 1]
            if lo <= value < hi:
                return f"{lo}–{hi - 1}"
        return f"{bucket_edges[-1]}+"

    def value_for(row: Dict[str, Any]) -> Optional[int]:
        if bucket_key == "move_number":
            v = row.get("meta", {}).get("move_number")
            return int(v) if isinstance(v, (int, float)) else None
        if bucket_key == "n_legal_moves":
            return int(row.get("n_legal_moves", 0))
        return None

    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in filtered:
        v = value_for(r)
        if v is None:
            continue
        by_bucket[bucket_label(v)].append(r)

    if not by_bucket:
        return f"_(no MAIN_GAME data for {label} breakdown)_"

    lines = [
        f"| {label} | N | mean rank | %top1 | %misaligned (≥3) | mean value_gain | %costly |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    # Sort by the numeric start of the bucket label for natural ordering.
    def sort_key(bucket: str) -> int:
        digits = bucket.split("–")[0].rstrip("+")
        try:
            return int(digits)
        except ValueError:
            return 0

    for bucket in sorted(by_bucket.keys(), key=sort_key):
        bucket_rows = by_bucket[bucket]
        ranks = [r["mcts"][key]["rank_of_final_best"] for r in bucket_rows]
        gains = [r["mcts"][key].get("value_gain_over_raw") for r in bucket_rows]
        gains = [g for g in gains if g is not None]
        n = len(bucket_rows)
        mean_rank = float(np.mean(ranks))
        pct_top1 = sum(1 for r in ranks if r == 0) / n * 100
        pct_misaligned = sum(1 for r in ranks if r >= 3) / n * 100
        mean_gain = float(np.mean(gains)) if gains else 0.0
        pct_costly = sum(1 for g in gains if g >= 0.1) / len(gains) * 100 if gains else 0.0
        lines.append(
            f"| {bucket} | {n} | {mean_rank:.2f} | {pct_top1:.1f}% | "
            f"{pct_misaligned:.1f}% | {mean_gain:+.3f} | {pct_costly:.1f}% |"
        )
    return "\n".join(lines)


def main_game_scatter(
    rows: List[Dict[str, Any]],
    top_sim: int,
    x_key: str,
    x_label: str,
    out_path: Path,
) -> None:
    """Scatter of rank-of-final-best vs an integer per-row key (MAIN_GAME only),
    with binned-mean overlay so the trend (if any) is visible through the noise."""
    key = str(top_sim)
    pts: List[Tuple[float, float]] = []
    for r in rows:
        if r["phase"] != "MAIN_GAME":
            continue
        if key not in r["mcts"] or "error" in r["mcts"][key]:
            continue
        if x_key == "move_number":
            xv = r.get("meta", {}).get("move_number")
        elif x_key == "n_legal_moves":
            xv = r.get("n_legal_moves")
        else:
            continue
        if xv is None:
            continue
        pts.append((float(xv), float(r["mcts"][key]["rank_of_final_best"])))
    if not pts:
        return

    xs, ys = zip(*pts)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(xs, ys, alpha=0.4, s=22, color="#2563eb")
    # Binned mean
    nbins = min(8, max(3, len(set(int(x) for x in xs)) // 3))
    bins = np.linspace(min(xs), max(xs), nbins + 1)
    bin_idx = np.digitize(xs, bins)
    bm_x, bm_y = [], []
    for b in range(1, len(bins)):
        in_bin = [y for x, y, i in zip(xs, ys, bin_idx) if i == b]
        if in_bin:
            bm_x.append((bins[b - 1] + bins[b]) / 2)
            bm_y.append(np.mean(in_bin))
    ax.plot(bm_x, bm_y, "o-", color="#b45309", linewidth=2, label="bin mean")
    ax.axhline(2.5, color="#999", linestyle="--", linewidth=1, label="rank≥3 (misaligned)")
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"rank_of_final_best at {top_sim} sims")
    ax.set_title(f"MAIN_GAME alignment vs {x_label.lower()}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def value_drift_chart(rows: List[Dict[str, Any]], sims: List[int], out_path: Path) -> None:
    """Two-panel chart: mean root_q vs sims (left), mean best_move_value vs sims (right)."""
    xs = [0] + [s for s in sims if s > 0]
    raw_values = [r["raw_policy"]["value"] for r in rows]
    raw_best = [r["raw_policy"]["best_move_value"] for r in rows
                if r["raw_policy"]["best_move_value"] is not None]
    means_rq = [float(np.mean(raw_values))]
    means_bv = [float(np.mean(raw_best))]
    for s in sims:
        if s <= 0:
            continue
        key = str(s)
        rqs = [r["mcts"][key]["root_q"] for r in rows
               if key in r["mcts"] and "error" not in r["mcts"][key]]
        bvs = [r["mcts"][key]["best_move_value"] for r in rows
               if key in r["mcts"] and "error" not in r["mcts"][key]
               and r["mcts"][key]["best_move_value"] is not None]
        means_rq.append(float(np.mean(rqs)) if rqs else 0.0)
        means_bv.append(float(np.mean(bvs)) if bvs else 0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(xs, means_rq, "o-", color="#2563eb")
    ax1.axhline(0, color="#999", linewidth=0.7, linestyle="--")
    ax1.set_xlabel("MCTS sims (0 = raw policy)")
    ax1.set_ylabel("mean root_q")
    ax1.set_title("Search-averaged value (drifts to 0 with sims)")
    ax1.set_xscale("symlog", linthresh=1)

    ax2.plot(xs, means_bv, "o-", color="#15803d")
    ax2.axhline(0, color="#999", linewidth=0.7, linestyle="--")
    ax2.set_xlabel("MCTS sims (0 = raw policy)")
    ax2.set_ylabel("mean best_move_value")
    ax2.set_title("Best-move value (closer to 'is this winning')")
    ax2.set_xscale("symlog", linthresh=1)

    fig.suptitle("Value convergence vs sim budget", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def rank_histogram_chart(rows: List[Dict[str, Any]], top_sim: int, out_path: Path) -> None:
    """Histogram of rank-of-final-best at the highest sim budget."""
    key = str(top_sim)
    ranks = [r["mcts"][key]["rank_of_final_best"] for r in rows
             if key in r["mcts"] and "error" not in r["mcts"][key]]
    if not ranks:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    max_rank = max(ranks)
    bins = np.arange(0, max(max_rank + 2, 10))
    ax.hist(ranks, bins=bins - 0.5, color="#2563eb", edgecolor="#1d4ed8")
    ax.set_xlabel(f"rank of MCTS-{top_sim} top move within raw policy ordering")
    ax.set_ylabel("# positions")
    ax.set_title("Network/search misalignment distribution")
    ax.set_xticks(range(0, max(max_rank + 1, 10)))
    ax.axvline(2.5, color="#b45309", linestyle="--", linewidth=1, label="rank≥3 (misaligned)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def entropy_vs_rank_chart(rows: List[Dict[str, Any]], top_sim: int, out_path: Path) -> None:
    """Scatter: does high policy entropy predict high rank-of-final-best?"""
    key = str(top_sim)
    pts = [
        (r["raw_policy"]["entropy"], r["mcts"][key]["rank_of_final_best"])
        for r in rows
        if key in r["mcts"] and "error" not in r["mcts"][key]
    ]
    if not pts:
        return
    xs, ys = zip(*pts)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(xs, ys, alpha=0.4, s=20, color="#2563eb")
    ax.set_xlabel("raw policy entropy (nats)")
    ax.set_ylabel("rank of MCTS top move in raw policy")
    ax.set_title("Does flat policy predict misalignment?")
    # Simple bin means
    bins = np.linspace(min(xs), max(xs), 8)
    bin_idx = np.digitize(xs, bins)
    bin_means_x, bin_means_y = [], []
    for b in range(1, len(bins)):
        in_bin = [y for x, y, i in zip(xs, ys, bin_idx) if i == b]
        if in_bin:
            bin_means_x.append((bins[b - 1] + bins[b]) / 2)
            bin_means_y.append(np.mean(in_bin))
    ax.plot(bin_means_x, bin_means_y, "o-", color="#b45309", linewidth=2, label="bin mean")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--measurements", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = args.out_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    rows = load(args.measurements)
    log.info("loaded %d OK rows from %s", len(rows), args.measurements)
    if not rows:
        log.error("no OK rows to report on")
        return

    # Recover sim budgets from the first row
    first = rows[0]
    sims = sorted(int(k) for k in first["mcts"].keys())
    top_sim = max(sims) if sims else 0

    log.info("sim budgets observed: %s", sims)

    # Generate charts
    value_drift_chart(rows, sims, charts_dir / "value_drift.png")
    if top_sim > 0:
        rank_histogram_chart(rows, top_sim, charts_dir / "rank_histogram.png")
        entropy_vs_rank_chart(rows, top_sim, charts_dir / "entropy_vs_rank.png")
        value_gain_chart(rows, top_sim, charts_dir / "value_gain.png")
        # MAIN_GAME-only breakdowns separating the "competence over game time"
        # signal from the phase / branching-factor confounds.
        main_game_scatter(rows, top_sim, "move_number", "Move number",
                          charts_dir / "rank_vs_move_number.png")
        main_game_scatter(rows, top_sim, "n_legal_moves", "Legal moves",
                          charts_dir / "rank_vs_n_legal.png")

    # Compose markdown
    md_lines = [
        f"# Analysis-board loop report",
        "",
        f"- Measurements: `{args.measurements.name}`",
        f"- Positions: {len(rows)}",
        f"- Sim budgets: {sims}",
        f"- Top budget for headline metrics: {top_sim}",
        "",
        "## Overview",
        "",
        "Per-budget aggregates. **rank_of_final_best** is the rank of the MCTS-chosen move within the raw network policy's ordering — 0 means the policy head's top-1 was the right answer. **%misaligned** counts positions where that rank ≥ 3, the threshold where the policy head is materially wrong about the best move.",
        "",
        overview_table(rows, sims),
        "",
    ]
    if top_sim > 0:
        # Move-number buckets — chosen to span early/mid/late MAIN_GAME without
        # overlapping the RING_PLACEMENT range. Adjust if game lengths shift.
        move_buckets = [11, 20, 30, 45, 60, 80]
        legal_buckets = [5, 15, 25, 35, 45, 55, 70]
        md_lines.extend([
            f"## rank-of-final-best at {top_sim} sims, by phase",
            "",
            rank_histogram_per_phase(rows, top_sim),
            "",
            "## MAIN_GAME alignment vs move number",
            "",
            "Stripped of RING_PLACEMENT / RING_REMOVAL / ROW_COMPLETION (which have their own per-phase numbers above) to isolate the *competence-over-game-time* signal from phase artifacts.",
            "",
            main_game_breakdown(rows, top_sim, "move_number", move_buckets, "Move #"),
            "",
            "## MAIN_GAME alignment vs branching factor",
            "",
            "Possible confound for the move-number breakdown — late-game positions have fewer rings and tighter board state, which lowers the legal-move count. Binning by `n_legal_moves` directly separates the two.",
            "",
            main_game_breakdown(rows, top_sim, "n_legal_moves", legal_buckets, "n_legal"),
            "",
            f"## 10 worst-misaligned positions at {top_sim} sims",
            "",
            "These are positions where the MCTS-chosen move was buried deep in the raw policy. Highest-EV training data: the network's policy was wrong by definition.",
            "",
            misaligned_examples(rows, top_sim, k=10),
            "",
            "## Charts",
            "",
            "![Value drift vs sims](charts/value_drift.png)",
            "",
            "![Rank-of-final-best distribution](charts/rank_histogram.png)",
            "",
            "![Entropy vs rank](charts/entropy_vs_rank.png)",
            "",
            "![Value cost of misalignment](charts/value_gain.png)",
            "",
            "![MAIN_GAME alignment vs move number](charts/rank_vs_move_number.png)",
            "",
            "![MAIN_GAME alignment vs legal-move count](charts/rank_vs_n_legal.png)",
            "",
        ])

    report_path = args.out_dir / "report.md"
    report_path.write_text("\n".join(md_lines))
    log.info("wrote %s", report_path)


if __name__ == "__main__":
    main()
