"""N-way comparison of measurement runs over the same set of positions.

Generalizes the previous 2-way compare.py: pass `--input <label> <path>`
repeatedly for as many models as you want to compare. The report shows
one overview table per metric (rows = sim budgets, cols = models), a
per-phase breakdown per metric at the top budget, pairwise position-level
improvements/regressions for every (A, B) pair, and overlaid charts.

Usage:
    python analysis_board/loop/compare.py \
        --input anchor analysis_board/loop/runs/<ts>/measurements_anchor.jsonl \
        --input iter1  analysis_board/loop/runs/<ts>/measurements_iter1.jsonl \
        --input iter4  analysis_board/loop/runs/<ts>/measurements_iter4.jsonl \
        --out-dir analysis_board/loop/runs/<ts>/compare_report/
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("compare")


# ---------------------------------------------------------------------------
# Loading + filtering
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("ok") or r.get("id") is None:
                continue
            out[r["id"]] = r
    return out


# ---------------------------------------------------------------------------
# Per-row metric extractors (factored so any metric can be plugged in)
# ---------------------------------------------------------------------------

def _at_budget(row: Dict[str, Any], sim: int) -> Optional[Dict[str, Any]]:
    """Get the mcts block at this sim budget, or None if missing/errored."""
    key = str(sim)
    cell = row.get("mcts", {}).get(key)
    if not cell or "error" in cell:
        return None
    return cell


_METRICS: List[Tuple[str, Callable[[Dict[str, Any]], Optional[float]], str, Optional[str]]] = [
    # (label, extractor, fmt, direction).
    # Direction interpretation for highlighting the "best" cell:
    #   "lower"  → lower value is better (less misalignment, less divergence)
    #   "higher" → higher value is better (rare here — most metrics measure error)
    #   None     → no monotonic direction (e.g., value_gain — closer to zero is
    #              the alignment signal, but both extremes are informative failure
    #              modes, so we don't bold either side)
    ("mean rank_of_final_best", lambda c: c.get("rank_of_final_best"), "{:.2f}", "lower"),
    ("% misaligned (rank≥3)",   lambda c: 1.0 if (c.get("rank_of_final_best") or 0) >= 3 else 0.0,
                                 "{:.1%}", "lower"),
    # value_gain: policy-vs-MCTS value delta. Positive = policy missed value
    # MCTS found; negative = MCTS picks land in positions the value head
    # rates worse than raw-policy picks (interesting failure mode). Closer
    # to zero = better policy/value/MCTS alignment. No monotonic "better".
    ("mean value_gain_over_raw", lambda c: c.get("value_gain_over_raw"), "{:+.3f}", None),
    # % costly: fraction of positions where the policy head missed ≥0.1 of
    # value vs what search found. Lower = policy head is more often making
    # value-equivalent picks to MCTS.
    ("% costly (gain≥0.1)",     lambda c: 1.0 if (c.get("value_gain_over_raw") or 0) >= 0.1 else 0.0,
                                 "{:.1%}", "lower"),
    ("% opposite-sign divergence", lambda c: float(c["opposite_sign_divergence"])
                                              if c.get("opposite_sign_divergence") is not None else None,
                                 "{:.1%}", "lower"),
    ("mean best_move_value",    lambda c: c.get("best_move_value"), "{:+.3f}", None),
    ("mean root_q",             lambda c: c.get("root_q"), "{:+.3f}", None),
]


def aggregate(rows: List[Dict[str, Any]], sim: int, extractor) -> Optional[float]:
    vals = []
    for r in rows:
        cell = _at_budget(r, sim)
        if cell is None:
            continue
        v = extractor(cell)
        if v is None:
            continue
        vals.append(float(v))
    return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def metric_table(
    rows_by_label: Dict[str, List[Dict[str, Any]]],
    sims: List[int],
    label_text: str,
    extractor,
    fmt: str,
    direction: Optional[str],
) -> str:
    """One table per metric: rows = sim budgets, cols = labels."""
    labels = list(rows_by_label.keys())
    direction_tag = ""
    if direction == "lower":
        direction_tag = " _(lower is better)_"
    elif direction == "higher":
        direction_tag = " _(higher is better)_"

    header = "| Budget | " + " | ".join(labels) + " |"
    sep = "|---:|" + "".join(["---:|" for _ in labels])
    lines = [f"### {label_text}{direction_tag}", "", header, sep]

    # Find the best per row for highlighting (bold the cell)
    for sim in sims:
        cells = []
        raw_vals: List[Optional[float]] = []
        for label in labels:
            v = aggregate(rows_by_label[label], sim, extractor)
            raw_vals.append(v)
        # Determine the "best" value to bold, if direction is set
        best_idx = None
        valid = [(i, v) for i, v in enumerate(raw_vals) if v is not None]
        if valid and direction in ("lower", "higher"):
            if direction == "lower":
                best_idx = min(valid, key=lambda x: x[1])[0]
            else:
                best_idx = max(valid, key=lambda x: x[1])[0]
        for i, v in enumerate(raw_vals):
            if v is None:
                cells.append("—")
            else:
                txt = fmt.format(v)
                if i == best_idx:
                    txt = f"**{txt}**"
                cells.append(txt)
        lines.append(f"| {sim} sims | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def per_phase_at_top(
    rows_by_label: Dict[str, List[Dict[str, Any]]],
    top_sim: int,
) -> str:
    """For each phase, show mean rank_of_final_best per label at the top budget."""
    labels = list(rows_by_label.keys())
    # Collect by phase
    by_phase: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for label, rows in rows_by_label.items():
        for r in rows:
            cell = _at_budget(r, top_sim)
            if cell is None:
                continue
            v = cell.get("rank_of_final_best")
            if v is None:
                continue
            by_phase[r["phase"]][label].append(float(v))
    if not by_phase:
        return "_(no MCTS data at top budget)_"

    header = "| Phase | N | " + " | ".join(f"mean rank ({l})" for l in labels) + " |"
    sep = "|---|---:|" + "".join(["---:|" for _ in labels])
    lines = [header, sep]
    for phase in sorted(by_phase.keys()):
        ns = [len(by_phase[phase].get(l, [])) for l in labels]
        n = min(ns) if ns else 0
        if n == 0:
            continue
        cells = [phase, str(n)]
        # Best per row
        means = [float(np.mean(by_phase[phase].get(l, [0]))) if by_phase[phase].get(l) else None
                 for l in labels]
        valid = [(i, m) for i, m in enumerate(means) if m is not None]
        best_idx = min(valid, key=lambda x: x[1])[0] if valid else None
        for i, m in enumerate(means):
            if m is None:
                cells.append("—")
            else:
                txt = f"{m:.2f}"
                if i == best_idx:
                    txt = f"**{txt}**"
                cells.append(txt)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pairwise_position_deltas(
    rows_by_label: Dict[str, List[Dict[str, Any]]],
    top_sim: int,
    label_a: str,
    label_b: str,
    k: int = 10,
) -> str:
    """Per-position rank deltas for one (a, b) pair."""
    by_id_a = {r["id"]: r for r in rows_by_label[label_a]}
    by_id_b = {r["id"]: r for r in rows_by_label[label_b]}
    shared = sorted(set(by_id_a) & set(by_id_b))
    deltas = []
    key = str(top_sim)
    for pid in shared:
        ra = by_id_a[pid]
        rb = by_id_b[pid]
        ca = _at_budget(ra, top_sim)
        cb = _at_budget(rb, top_sim)
        if not ca or not cb:
            continue
        rank_a = ca.get("rank_of_final_best", 0)
        rank_b = cb.get("rank_of_final_best", 0)
        deltas.append({
            "id": pid,
            "phase": ra["phase"],
            "rank_a": rank_a,
            "rank_b": rank_b,
            "d_rank": rank_b - rank_a,
            "gain_a": ca.get("value_gain_over_raw"),
            "gain_b": cb.get("value_gain_over_raw"),
        })

    # B improves on A: A was misaligned (rank≥3), B's rank dropped by ≥3
    improvements = sorted(
        [d for d in deltas if d["rank_a"] >= 3 and d["d_rank"] <= -3],
        key=lambda d: d["d_rank"],
    )[:k]
    # B regresses from A: A was aligned (rank≤1), B's rank jumped by ≥5
    regressions = sorted(
        [d for d in deltas if d["rank_a"] <= 1 and d["d_rank"] >= 5],
        key=lambda d: -d["d_rank"],
    )[:k]

    def fmt_block(rows, title):
        out = [f"**{title}**", "",
               f"| id | phase | rank ({label_a}) | rank ({label_b}) | Δ rank | gain ({label_a}) | gain ({label_b}) |",
               "|---|---|---:|---:|---:|---:|---:|"]
        for d in rows:
            ga = f"{d['gain_a']:+.3f}" if isinstance(d['gain_a'], (int, float)) else "—"
            gb = f"{d['gain_b']:+.3f}" if isinstance(d['gain_b'], (int, float)) else "—"
            out.append(f"| `{d['id']}` | {d['phase']} | {d['rank_a']} | {d['rank_b']} | "
                       f"{d['d_rank']:+d} | {ga} | {gb} |")
        return "\n".join(out)

    n_imp = sum(1 for d in deltas if d["rank_a"] >= 3 and d["d_rank"] <= -3)
    n_reg = sum(1 for d in deltas if d["rank_a"] <= 1 and d["d_rank"] >= 5)

    return "\n\n".join([
        f"#### {label_a} → {label_b}",
        f"_(N shared = {len(shared)}, improvements = {n_imp}, regressions = {n_reg})_",
        fmt_block(improvements, f"{label_b} improved on {label_a} (top {min(k, len(improvements))})"),
        fmt_block(regressions, f"{label_b} regressed from {label_a} (top {min(k, len(regressions))})"),
    ])


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

_PALETTE = ["#2563eb", "#b91c1c", "#15803d", "#7c3aed", "#b45309", "#0891b2"]


def trajectory_chart(
    rows_by_label: Dict[str, List[Dict[str, Any]]],
    sims: List[int],
    extractor,
    metric_name: str,
    out_path: Path,
    direction: Optional[str] = None,
) -> None:
    """One line per model showing the metric across sim budgets."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xs = [s for s in sims if s > 0]
    for i, (label, rows) in enumerate(rows_by_label.items()):
        ys = [aggregate(rows, s, extractor) for s in xs]
        ys = [y if y is not None else float("nan") for y in ys]
        ax.plot(xs, ys, "o-", color=_PALETTE[i % len(_PALETTE)], label=label, linewidth=2)
    ax.set_xlabel("MCTS sims")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} vs sim budget")
    ax.set_xscale("symlog", linthresh=100)
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def rank_distribution_overlay(
    rows_by_label: Dict[str, List[Dict[str, Any]]],
    top_sim: int,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    all_ranks: List[int] = []
    by_label: Dict[str, List[int]] = {}
    for label, rows in rows_by_label.items():
        ranks = []
        for r in rows:
            cell = _at_budget(r, top_sim)
            if cell is None:
                continue
            v = cell.get("rank_of_final_best")
            if v is not None:
                ranks.append(int(v))
        by_label[label] = ranks
        all_ranks.extend(ranks)
    if not all_ranks:
        return
    bins = np.arange(0, max(all_ranks) + 2) - 0.5
    for i, (label, ranks) in enumerate(by_label.items()):
        ax.hist(ranks, bins=bins, alpha=0.45, color=_PALETTE[i % len(_PALETTE)],
                label=label, edgecolor="white", linewidth=0.5)
    ax.axvline(2.5, color="#b45309", linestyle="--", linewidth=1, label="rank≥3 (misaligned)")
    ax.set_xlabel(f"rank_of_final_best at {top_sim} sims")
    ax.set_ylabel("# positions")
    ax.set_title("Alignment distribution comparison")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def gain_distribution_overlay(
    rows_by_label: Dict[str, List[Dict[str, Any]]],
    top_sim: int,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    by_label: Dict[str, List[float]] = {}
    all_gains: List[float] = []
    for label, rows in rows_by_label.items():
        gains = []
        for r in rows:
            cell = _at_budget(r, top_sim)
            if cell is None:
                continue
            v = cell.get("value_gain_over_raw")
            if v is not None:
                gains.append(float(v))
        by_label[label] = gains
        all_gains.extend(gains)
    if not all_gains:
        return
    bins = np.linspace(min(all_gains) - 0.05, max(all_gains) + 0.05, 30)
    for i, (label, gains) in enumerate(by_label.items()):
        ax.hist(gains, bins=bins, alpha=0.45, color=_PALETTE[i % len(_PALETTE)],
                label=label, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="#999", linestyle="-", linewidth=0.7)
    ax.axvline(0.1, color="#b45309", linestyle="--", linewidth=1, label="costly threshold")
    ax.set_xlabel(f"value_gain_over_raw at {top_sim} sims")
    ax.set_ylabel("# positions")
    ax.set_title("Cost-of-misalignment distribution comparison")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True,
        help="repeatable: --input <label> <path-to-measurements.jsonl>. Provide 2 or more.",
    )
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()

    if len(args.input) < 2:
        raise SystemExit("Need at least 2 --input entries to compare.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = args.out_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    # Load all
    raw_by_label: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for label, path in args.input:
        rows = load_jsonl(Path(path))
        raw_by_label[label] = rows
        log.info("loaded %d rows for %s from %s", len(rows), label, path)

    # Intersect IDs so every model has the same position set (apples-to-apples)
    id_sets = [set(d.keys()) for d in raw_by_label.values()]
    shared_ids = sorted(set.intersection(*id_sets))
    log.info("shared positions across all inputs: %d", len(shared_ids))
    if not shared_ids:
        raise SystemExit("No positions are common across all inputs.")

    rows_by_label: Dict[str, List[Dict[str, Any]]] = {
        label: [d[i] for i in shared_ids] for label, d in raw_by_label.items()
    }

    # Recover sim budgets from the first model's first row
    first_label = next(iter(rows_by_label))
    sims = sorted(int(k) for k in rows_by_label[first_label][0]["mcts"].keys())
    top_sim = max(sims) if sims else 0
    log.info("sim budgets: %s  top=%d", sims, top_sim)

    # --- Charts ---
    for label_text, extractor, fmt, direction in _METRICS:
        # Trajectory chart per metric
        slug = label_text.lower().replace(" ", "_").replace("%", "pct").replace("≥", "ge").replace("(", "").replace(")", "")
        trajectory_chart(rows_by_label, sims, extractor, label_text,
                         charts_dir / f"trajectory_{slug}.png", direction)
    rank_distribution_overlay(rows_by_label, top_sim, charts_dir / "rank_distribution.png")
    gain_distribution_overlay(rows_by_label, top_sim, charts_dir / "gain_distribution.png")

    # --- Markdown report ---
    md = [
        "# N-way comparison report",
        "",
        f"- Inputs: {len(args.input)}",
        f"- Shared positions: {len(shared_ids)}",
        f"- Sim budgets: {sims}",
        f"- Top budget for headline metrics: {top_sim}",
        "",
        "## Models",
        "",
    ]
    for label, path in args.input:
        md.append(f"- **{label}**: `{path}`")
    md.extend(["", "## Per-metric overview", "",
               "Each table shows the metric at each sim budget, one column per model. Best value in each row is bolded."])
    for label_text, extractor, fmt, direction in _METRICS:
        md.append("")
        md.append(metric_table(rows_by_label, sims, label_text, extractor, fmt, direction))
        md.append("")

    md.extend([f"## Per-phase rank_of_final_best at {top_sim} sims", "",
               per_phase_at_top(rows_by_label, top_sim), ""])

    md.extend(["## Pairwise position-level changes",
               "",
               f"Each pair shows positions where the *second* model materially improved or regressed vs the first, at {top_sim} sims. 'Improved' = first was misaligned (rank≥3) and second dropped rank by ≥3. 'Regressed' = first was aligned (rank≤1) and second jumped rank by ≥5.",
               ""])
    labels = list(rows_by_label.keys())
    for a, b in combinations(labels, 2):
        md.append(pairwise_position_deltas(rows_by_label, top_sim, a, b, k=10))
        md.append("")

    md.append("## Charts")
    md.append("")
    md.append("![Rank distribution](charts/rank_distribution.png)")
    md.append("")
    md.append("![Gain distribution](charts/gain_distribution.png)")
    md.append("")
    for label_text, _, _, _ in _METRICS:
        slug = label_text.lower().replace(" ", "_").replace("%", "pct").replace("≥", "ge").replace("(", "").replace(")", "")
        md.append(f"### Trajectory: {label_text}")
        md.append(f"![{label_text}](charts/trajectory_{slug}.png)")
        md.append("")

    out_path = args.out_dir / "report.md"
    out_path.write_text("\n".join(md))
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
