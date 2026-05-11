#!/usr/bin/env python3
"""End-of-run absolute-strength eval: candidate vs HeuristicAgent.

Plays N games between a candidate checkpoint and HeuristicAgent at a given
depth, with the candidate playing via pure-neural MCTS at deployment-realistic
budget. Reuses tournament.run_anchor_eval to keep the eval path identical to
what the supervisor uses during training (just standalone, not wired to a
training loop).

Usage:
    python scripts/eval_vs_heuristic.py \\
        --checkpoint models/supervised_seed/best_supervised.pt \\
        --num-games 100 \\
        --depth 3 \\
        --mcts-simulations 400 \\
        --device cuda
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.utils.tournament import ModelTournament

logger = logging.getLogger("eval_vs_heuristic")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to candidate checkpoint .pt")
    parser.add_argument("--num-games", type=int, default=100,
                        help="Total games (split half white / half black)")
    parser.add_argument("--depth", type=int, default=3,
                        help="HeuristicAgent search depth")
    parser.add_argument("--mcts-simulations", type=int, default=400,
                        help="MCTS sim budget (deployment 'hard' preset = 400)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument(
        "--time-limit-per-move",
        type=float,
        default=0.0,
        help=(
            "Per-move wall-clock cap (seconds) on HeuristicAgent's alpha-beta "
            "search. 0.0 (default) = no limit. STRONGLY RECOMMENDED for "
            "depth=3: pass --time-limit-per-move 30 to prevent indefinite "
            "hangs on pathological network-produced positions. With iterative "
            "deepening on, the agent returns the deepest-completed depth's "
            "best move when the budget is hit."
        ),
    )
    parser.add_argument("--label", type=str, default=None,
                        help="Display label (defaults to checkpoint stem)")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--use-mcts", action="store_true", default=True,
                        help="(default) Candidate plays via MCTS")
    parser.add_argument("--no-mcts", dest="use_mcts", action="store_false",
                        help="Candidate plays via raw policy argmax instead")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Move-selection temperature for the candidate. 0.0 (default) "
                             "is argmax — deterministic, fragile to side-of-board artifacts. "
                             "Use 0.5–1.0 for stochastic reads of true strength.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = args.device
    logger.info(f"Device: {device}")

    if not args.checkpoint.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    label = args.label or args.checkpoint.stem
    logger.info(f"Loading checkpoint as '{label}': {args.checkpoint}")

    # Pass model_path to constructor so the auto-detect (input_channels,
    # num_channels, num_blocks) runs against the checkpoint state_dict.
    # Calling load_model AFTER construction misses the auto-detect path
    # and breaks on 15-channel enhanced-encoding checkpoints.
    net = NetworkWrapper(model_path=str(args.checkpoint), device=device)

    # Use the supervisor's tournament infra so eval path matches what training
    # uses — just instantiated standalone here.
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp())
    mgr = ModelTournament(training_dir=tmp_dir, device=device, games_per_match=1)

    mode = "mcts" if args.use_mcts else "raw_policy"
    logger.info(
        f"Eval: {label} ({mode}) vs HeuristicAgent(depth={args.depth}) × "
        f"{args.num_games} games (mcts_sims={args.mcts_simulations if args.use_mcts else 0})"
    )
    t0 = time.time()
    if args.depth >= 3 and args.time_limit_per_move <= 0.0:
        logger.warning(
            "Running depth=3 with --time-limit-per-move=0 (no cap). "
            "Alpha-beta is known to blow up on some network-produced "
            "positions (see WARMSTART_PHASE_LOG.md §4b/§5b). Consider "
            "--time-limit-per-move 30."
        )
    result = mgr.run_anchor_eval(
        candidate_network=net,
        candidate_label=label,
        num_games=args.num_games,
        depth=args.depth,
        seed=args.seed,
        max_moves_per_game=args.max_moves,
        use_mcts=args.use_mcts,
        mcts_simulations=args.mcts_simulations,
        heuristic_time_limit_seconds=args.time_limit_per_move,
        candidate_temperature=args.temperature,
    )
    elapsed = time.time() - t0

    games = result.get("games_played", 0)
    wins = result.get("candidate_wins", 0)
    losses = result.get("anchor_wins", 0)
    draws = result.get("draws", 0)
    rate = result.get("win_rate", 0.0)

    # Wilson 95% CI
    if games > 0:
        import numpy as np
        z = 1.96
        denom = 1 + z * z / games
        centre = (rate + z * z / (2 * games)) / denom
        half = (z * np.sqrt(rate * (1 - rate) / games + z * z / (4 * games * games))) / denom
        ci_lo, ci_hi = centre - half, centre + half
    else:
        ci_lo, ci_hi = 0.0, 1.0

    print("\n" + "=" * 72)
    print(f"Result: {label} ({mode}) vs HeuristicAgent(depth={args.depth})")
    print("=" * 72)
    print(f"  Games played:   {games}")
    print(f"  Candidate wins: {wins}  ({rate:.1%})")
    print(f"  Anchor wins:    {losses}")
    print(f"  Draws:          {draws}")
    print(f"  Win rate:       {rate:.3f}  CI95=[{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  Avg game length: {result.get('avg_game_length', 0):.1f} moves")
    print(f"  Wall-clock:     {elapsed:.0f}s ({elapsed/max(games,1):.1f}s/game)")
    # Per-side breakdown + deterministic-collapse flag
    per_side = result.get("per_side") or {}
    for side, ss in per_side.items():
        print(
            f"  side={side}: cand_wins={ss['cand_wins']}/{ss['games']} "
            f"({ss['cand_win_rate']:.1%}); game_length "
            f"min/avg/max = {ss['game_length_min']}/{ss['avg_game_length']:.1f}/"
            f"{ss['game_length_max']}"
        )
    det_sides = result.get("deterministic_sides") or []
    if det_sides:
        print(
            f"  ⚠️  DETERMINISTIC-COLLAPSE on {', '.join(det_sides)} — every "
            f"game on those sides replayed the same line. The win rate "
            f"measures side-coverage, not skill. Re-run with --temperature 0.5+ "
            f"or with MCTS to break determinism."
        )
    print("=" * 72)

    # Verdict line — friendly for the "intermediate player?" question.
    if ci_lo >= 0.65:
        verdict = "STRONG: clears 'intermediate player' bar (≥65% lower CI)."
    elif ci_lo >= 0.55:
        verdict = "PROMISING: beats heuristic with margin; below 65% bar."
    elif ci_hi >= 0.5:
        verdict = "WEAK: roughly even with heuristic; below intermediate."
    else:
        verdict = "FAILS: candidate consistently loses to heuristic."
    print(f"\nVerdict: {verdict}")

    if args.output_json:
        out = {
            "label": label,
            "checkpoint": str(args.checkpoint),
            "mode": mode,
            "depth": args.depth,
            "temperature": args.temperature,
            "mcts_simulations": args.mcts_simulations if args.use_mcts else 0,
            "num_games": games,
            "candidate_wins": wins,
            "anchor_wins": losses,
            "draws": draws,
            "win_rate": rate,
            "ci95": [ci_lo, ci_hi],
            "elapsed_seconds": elapsed,
            "verdict": verdict,
            "per_side": result.get("per_side"),
            "deterministic_sides": result.get("deterministic_sides"),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
