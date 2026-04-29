#!/usr/bin/env python3
"""Head-to-head bake-off between two checkpoints.

The polished-vs-baseline comparison that Decision Gate #1 asks for. Plays a
fixed number of games in each direction (White/Black alternated) with a
pinned seed and the Wilson-CI reporting the rest of the pipeline already
uses, then emits a Markdown report.

Usage:
    python scripts/run_bakeoff.py \\
        --challenger runs/bakeoff_challenger/.../iteration_50/checkpoint_iteration_50.pt \\
        --baseline  runs/bakeoff_baseline/.../iteration_50/checkpoint_iteration_50.pt \\
        --games 200 \\
        --seed 20260416 \\
        --output analysis_output/bakeoffs/track_a.md

Games are split in half per direction (e.g. `--games 200` → 100 as White,
100 as Black for each model), so White-advantage bias cancels out of the
aggregated win rate. The Wilson CI is computed over decisive games only
(draws excluded from the denominator); draws are reported separately.

ELO delta estimated as `400 · log10(wr / (1 − wr))` where wr is the
challenger's win rate over decisive games.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# Make the repo importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yinsh_ml.utils.stats import wilson_bounds
from yinsh_ml.utils.tournament import ModelTournament


def _resolve_device(request: str) -> str:
    """Mirror the run_training.py 'auto' resolution."""
    if request != "auto":
        return request
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _elo_delta(win_rate: float) -> float:
    """ELO delta implied by a head-to-head win rate. 50% → 0; clamps at ±800
    to avoid log(0) blowups at 0%/100% (common with small sample sizes)."""
    eps = 1e-3
    wr = max(eps, min(1.0 - eps, win_rate))
    return 400.0 * math.log10(wr / (1.0 - wr))


def run_bakeoff(
    challenger_path: Path,
    baseline_path: Path,
    games: int,
    eval_seed: int,
    device: str,
    temperature: float,
    use_ema_for_eval: bool,
) -> dict:
    """Run the head-to-head and return a stats dict. Split games in half per
    direction so White-advantage cancels. Uses a throwaway ModelTournament
    so we reuse `_load_model` (EMA preference) and `_play_match`
    (deterministic-eval seeding) without duplicating their logic here."""
    games_per_direction = games // 2
    if games_per_direction * 2 != games:
        raise ValueError(f"games ({games}) must be even so White/Black split evenly")

    # training_dir is unused when we drive _play_match ourselves; point it at
    # /tmp so the tournament's own housekeeping files don't pollute the repo.
    tournament = ModelTournament(
        training_dir=Path("/tmp"),
        device=device,
        games_per_match=games_per_direction,
        temperature=temperature,
        eval_seed=eval_seed,
        use_ema_for_eval=use_ema_for_eval,
    )

    logging.info(
        "Loading challenger: %s (EMA preference: %s)",
        challenger_path, use_ema_for_eval,
    )
    challenger_model = tournament._load_model(challenger_path)
    logging.info("Loading baseline:   %s", baseline_path)
    baseline_model = tournament._load_model(baseline_path)

    logging.info(
        "Playing %d games as challenger-White, %d as challenger-Black (total %d)",
        games_per_direction, games_per_direction, games,
    )

    t0 = time.time()
    # Direction 1: challenger = White, baseline = Black.
    fwd = tournament._play_match(
        challenger_model, baseline_model, "challenger", "baseline",
    )
    logging.info(
        "Forward direction: challenger-White wins=%d, baseline-Black wins=%d, draws=%d",
        fwd.white_wins, fwd.black_wins, fwd.draws,
    )
    # Direction 2: challenger = Black, baseline = White.
    bwd = tournament._play_match(
        baseline_model, challenger_model, "baseline", "challenger",
    )
    logging.info(
        "Reverse direction: baseline-White wins=%d, challenger-Black wins=%d, draws=%d",
        bwd.white_wins, bwd.black_wins, bwd.draws,
    )
    elapsed_s = time.time() - t0

    # Aggregate from challenger's perspective.
    challenger_wins = fwd.white_wins + bwd.black_wins
    baseline_wins = fwd.black_wins + bwd.white_wins
    draws = fwd.draws + bwd.draws
    decisive = challenger_wins + baseline_wins

    # Wilson CI on challenger's proportion of decisive games (draws excluded
    # from the denominator — reported separately). If every game is a draw,
    # Wilson returns (0, 1) to flag "no signal" honestly.
    lb, ub = wilson_bounds(challenger_wins, decisive)
    win_rate = challenger_wins / decisive if decisive > 0 else 0.5

    # Also compute per-direction Wilson CIs so White-advantage asymmetry is
    # visible — if the forward direction says "challenger wins 80% as white"
    # but the reverse says "55% as black," that's a strong signal that the
    # aggregate number is hiding a color-dependent effect.
    fwd_decisive = fwd.white_wins + fwd.black_wins
    fwd_ci = wilson_bounds(fwd.white_wins, fwd_decisive)  # challenger=white in fwd
    bwd_decisive = bwd.white_wins + bwd.black_wins
    bwd_ci = wilson_bounds(bwd.black_wins, bwd_decisive)  # challenger=black in bwd

    return {
        "challenger_path": str(challenger_path),
        "baseline_path": str(baseline_path),
        "games": games,
        "eval_seed": eval_seed,
        "temperature": temperature,
        "device": device,
        "elapsed_seconds": elapsed_s,
        "aggregate": {
            "challenger_wins": challenger_wins,
            "baseline_wins": baseline_wins,
            "draws": draws,
            "decisive": decisive,
            "win_rate": win_rate,
            "wilson_lower": lb,
            "wilson_upper": ub,
            "elo_delta": _elo_delta(win_rate) if decisive > 0 else 0.0,
        },
        "forward_direction": {
            "description": "challenger plays White",
            "challenger_wins": fwd.white_wins,
            "baseline_wins": fwd.black_wins,
            "draws": fwd.draws,
            "wilson_ci": list(fwd_ci),
        },
        "reverse_direction": {
            "description": "challenger plays Black",
            "challenger_wins": bwd.black_wins,
            "baseline_wins": bwd.white_wins,
            "draws": bwd.draws,
            "wilson_ci": list(bwd_ci),
        },
    }


def render_report(stats: dict) -> str:
    """Markdown report. Top-of-file verdict so a skimmer sees it immediately."""
    agg = stats["aggregate"]
    fwd = stats["forward_direction"]
    bwd = stats["reverse_direction"]
    lb, ub = agg["wilson_lower"], agg["wilson_upper"]
    # Promotion-gate-style straddle check: if CI crosses 0.5, the result is
    # statistical noise (consistent with either side being better).
    straddles = lb <= 0.5 <= ub
    verdict = (
        "❌ **Inconclusive** — 95% CI straddles 50% win rate"
        if straddles else
        ("✅ **Challenger wins**" if agg["win_rate"] > 0.5 else "⚠️  **Baseline wins**")
    )
    elapsed_min = stats["elapsed_seconds"] / 60.0

    lines = [
        "# Bake-off report",
        "",
        f"- **Challenger**: `{stats['challenger_path']}`",
        f"- **Baseline**:  `{stats['baseline_path']}`",
        f"- **Games**: {stats['games']} (split {stats['games']//2}/{stats['games']//2} per direction)",
        f"- **Eval seed**: {stats['eval_seed']} (deterministic — rerunning reproduces exactly)",
        f"- **Device / temp**: {stats['device']} / T={stats['temperature']}",
        f"- **Wall-clock**: {elapsed_min:.1f} min",
        f"- **Generated**: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Aggregate",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Challenger wins | {agg['challenger_wins']} |",
        f"| Baseline wins | {agg['baseline_wins']} |",
        f"| Draws | {agg['draws']} |",
        f"| Decisive games | {agg['decisive']} |",
        f"| Challenger win rate (decisive) | {agg['win_rate']:.3f} |",
        f"| Wilson 95% CI | [{lb:.3f}, {ub:.3f}] |",
        f"| ELO Δ (challenger − baseline) | {agg['elo_delta']:+.1f} |",
        "",
        "## Per-direction breakdown",
        "",
        f"_White-advantage sanity check — if these diverge wildly, the aggregate hides a color-dependent effect._",
        "",
        f"| Direction | Challenger wins | Baseline wins | Draws | Wilson 95% CI |",
        f"|---|---|---|---|---|",
        f"| Challenger plays White | {fwd['challenger_wins']} | {fwd['baseline_wins']} | {fwd['draws']} | [{fwd['wilson_ci'][0]:.3f}, {fwd['wilson_ci'][1]:.3f}] |",
        f"| Challenger plays Black | {bwd['challenger_wins']} | {bwd['baseline_wins']} | {bwd['draws']} | [{bwd['wilson_ci'][0]:.3f}, {bwd['wilson_ci'][1]:.3f}] |",
        "",
        "## Raw stats (JSON)",
        "",
        "```json",
        json.dumps(stats, indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Head-to-head bake-off between two YINSH checkpoints."
    )
    parser.add_argument("--challenger", type=Path, required=True,
                        help="Path to challenger checkpoint (.pt).")
    parser.add_argument("--baseline", type=Path, required=True,
                        help="Path to baseline checkpoint (.pt).")
    parser.add_argument("--games", type=int, default=400,
                        help="Total games played (split in half per direction). Must be even.")
    parser.add_argument("--seed", type=int, default=20260416,
                        help="eval_seed for deterministic play. Pin this if you need to reproduce a run.")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Move-selection temperature (match training default for a fair eval).")
    parser.add_argument("--device", type=str, default="auto",
                        help="cuda, mps, cpu, or auto.")
    parser.add_argument("--use-ema", action="store_true", default=True,
                        help="Prefer <ckpt>_ema.pt sibling when present (default: True).")
    parser.add_argument("--no-ema", dest="use_ema", action="store_false",
                        help="Force-disable EMA preference.")
    parser.add_argument("--output", type=Path,
                        default=Path("analysis_output/bakeoffs/bakeoff.md"),
                        help="Report path. Parent dirs are created.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    device = _resolve_device(args.device)

    for label, path in (("challenger", args.challenger), ("baseline", args.baseline)):
        if not path.exists():
            logging.error("%s checkpoint not found: %s", label, path)
            return 1

    stats = run_bakeoff(
        challenger_path=args.challenger,
        baseline_path=args.baseline,
        games=args.games,
        eval_seed=args.seed,
        device=device,
        temperature=args.temperature,
        use_ema_for_eval=args.use_ema,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(stats))
    logging.info("Wrote report to %s", args.output)

    # Pretty-print the verdict for scripting/CI consumers.
    agg = stats["aggregate"]
    print(
        f"\nChallenger {agg['challenger_wins']}-{agg['baseline_wins']}-{agg['draws']} baseline "
        f"| wr={agg['win_rate']:.3f} (95% CI [{agg['wilson_lower']:.3f}, {agg['wilson_upper']:.3f}]) "
        f"| ΔElo≈{agg['elo_delta']:+.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
