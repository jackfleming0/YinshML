#!/usr/bin/env python3
"""Evaluate a trained model against expert (BGA) games.

Replays each expert game move-by-move and asks the model what it would have
played. Reports policy agreement (top-1, top-3) and value MSE, bucketed by
game phase. The expert games are kept as a permanent holdout — never train on
them — so this metric stays comparable across training runs.

Usage:
    python scripts/eval_vs_expert.py \\
        --model checkpoints/iter_42.pt \\
        --games-dir expert_games/bga/parsed/ \\
        --out eval_reports/iter_42_vs_bga.json
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from yinsh_ml.data.converter import GameConverter
from yinsh_ml.game.constants import Player, Position
from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.types import GamePhase, Move, MoveType
from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.utils.encoding import StateEncoder
from yinsh_ml.utils.enhanced_encoding import EnhancedStateEncoder

logger = logging.getLogger(__name__)


def replay_to_eval_positions(
    game_data: dict, encoder: StateEncoder
) -> Optional[List[Dict]]:
    """Replay one expert game and emit a per-position eval record.

    Returns None if the game can't be parsed cleanly — we drop the whole game
    rather than emit partial records, so a single bad parse doesn't pollute
    the metric.
    """
    result = game_data.get("result")
    moves = game_data.get("moves", [])
    if not moves:
        return None
    # `result == "unknown"` happens on BGA games abandoned mid-play (no
    # player captured 3 rings; the gameEnd notification carries an empty
    # player_id). The moves themselves are still real expert moves, so we
    # keep these positions for policy agreement and just skip them when
    # computing value MSE (outcome_value=None).
    if result not in ("white", "black", "draw", "unknown"):
        return None

    converter = GameConverter(encoder=encoder)
    gs = GameState()
    records: List[Dict] = []

    for move_data in moves:
        move = converter._parse_move(move_data, gs)
        if move is None:
            return None

        try:
            expert_idx = encoder.move_to_index(move)
        except (ValueError, IndexError):
            return None

        valid_moves = gs.get_valid_moves()
        valid_indices: List[int] = []
        for m in valid_moves:
            try:
                valid_indices.append(encoder.move_to_index(m))
            except (ValueError, IndexError):
                continue
        if expert_idx not in valid_indices:
            # Expert move isn't in the legal-move enumeration — engine
            # disagreement; safer to drop than to score a phantom move.
            return None

        records.append(
            {
                "state": encoder.encode_state(gs),
                "expert_idx": expert_idx,
                "valid_indices": valid_indices,
                "outcome_value": (
                    None if result == "unknown"
                    else GameConverter._outcome_value(result, gs.current_player)
                ),
                "phase": gs.phase.name,
                "side_to_move": "white" if gs.current_player == Player.WHITE else "black",
                "move_num": len(gs.move_history),
                "game_id": game_data.get("game_id", ""),
            }
        )

        if not gs.make_move(move):
            return None

    return records


def load_expert_positions(
    games_dir: str, encoder: StateEncoder, min_rating: int = 0
) -> List[Dict]:
    path = Path(games_dir)
    out: List[Dict] = []
    n_games = 0
    n_dropped = 0
    for json_file in sorted(path.glob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            n_dropped += 1
            continue

        games = data if isinstance(data, list) else [data]
        for game in games:
            if min_rating > 0:
                players = game.get("players", {})
                w = players.get("white", {}).get("rating", 0)
                b = players.get("black", {}).get("rating", 0)
                if w < min_rating or b < min_rating:
                    continue

            recs = replay_to_eval_positions(game, encoder)
            if recs is None:
                n_dropped += 1
                continue
            out.extend(recs)
            n_games += 1

    logger.info(
        f"Loaded {len(out)} positions from {n_games} games "
        f"({n_dropped} games dropped)"
    )
    return out


def evaluate(
    model: NetworkWrapper,
    positions: List[Dict],
    batch_size: int = 256,
) -> Dict:
    device = model.device
    policy_size = model.state_encoder.total_moves

    buckets: Dict[str, Dict] = defaultdict(
        lambda: {"n": 0, "top1": 0, "top3": 0,
                 "n_value": 0, "value_sq_err": 0.0}
    )

    for start in range(0, len(positions), batch_size):
        chunk = positions[start : start + batch_size]
        states = np.stack([p["state"] for p in chunk]).astype(np.float32)
        state_tensor = torch.from_numpy(states).to(device)

        # Build a per-row valid-move mask so top-k is over legal moves only.
        # Without this, the model can "agree" by ranking an illegal move above
        # the expert, which is meaningless for a self-play-trained agent.
        mask = torch.zeros((len(chunk), policy_size), dtype=torch.bool, device=device)
        for i, p in enumerate(chunk):
            mask[i, p["valid_indices"]] = True

        probs, values = model.predict(state_tensor, move_mask=mask, temperature=1.0)
        # `probs` already has -inf at invalid slots → softmax → 0, so argmax/top-k
        # naturally pick legal moves.
        top3 = torch.topk(probs, k=min(3, policy_size), dim=1).indices.cpu().numpy()
        top1 = top3[:, 0]
        values_np = values.detach().cpu().numpy().reshape(-1)

        for i, p in enumerate(chunk):
            expert = p["expert_idx"]
            outcome = p["outcome_value"]
            for key in ("ALL", p["phase"], f"side_{p['side_to_move']}"):
                b = buckets[key]
                b["n"] += 1
                if top1[i] == expert:
                    b["top1"] += 1
                if expert in top3[i]:
                    b["top3"] += 1
                if outcome is not None:
                    b["n_value"] += 1
                    b["value_sq_err"] += float((values_np[i] - outcome) ** 2)

    report: Dict = {}
    for key, b in buckets.items():
        if b["n"] == 0:
            continue
        report[key] = {
            "n": b["n"],
            "top1_acc": b["top1"] / b["n"],
            "top3_acc": b["top3"] / b["n"],
            "n_value": b["n_value"],
            "value_mse": (
                b["value_sq_err"] / b["n_value"] if b["n_value"] > 0 else None
            ),
        }
    return report


def format_report(report: Dict) -> str:
    phase_order = ["ALL", "RING_PLACEMENT", "MAIN_GAME", "ROW_COMPLETION",
                   "RING_REMOVAL", "GAME_OVER", "side_white", "side_black"]
    ordered = [k for k in phase_order if k in report]
    ordered += [k for k in report if k not in phase_order]

    lines = [
        f"{'bucket':<20} {'n':>7} {'top1':>8} {'top3':>8} "
        f"{'n_val':>7} {'val_mse':>9}",
        "-" * 64,
    ]
    for k in ordered:
        r = report[k]
        val_mse = (
            f"{r['value_mse']:>9.4f}" if r["value_mse"] is not None else f"{'—':>9}"
        )
        lines.append(
            f"{k:<20} {r['n']:>7d} {r['top1_acc']:>8.3f} "
            f"{r['top3_acc']:>8.3f} {r['n_value']:>7d} {val_mse}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--games-dir",
        default="expert_games/bga/parsed",
        help="Directory of parsed expert game JSON files",
    )
    parser.add_argument("--use-enhanced-encoding", action="store_true")
    parser.add_argument(
        "--value-mode", choices=["classification", "regression"], default="classification"
    )
    parser.add_argument("--min-rating", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--out", default=None, help="Optional path to write JSON report"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    encoder = (
        EnhancedStateEncoder() if args.use_enhanced_encoding else StateEncoder()
    )
    positions = load_expert_positions(args.games_dir, encoder, args.min_rating)
    if not positions:
        logger.error("No positions loaded — aborting.")
        sys.exit(1)

    model = NetworkWrapper(
        model_path=args.model,
        device=args.device,
        value_mode=args.value_mode,
        use_enhanced_encoding=args.use_enhanced_encoding,
    )

    report = evaluate(model, positions, batch_size=args.batch_size)
    print(format_report(report))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "model": args.model,
                    "games_dir": args.games_dir,
                    "min_rating": args.min_rating,
                    "use_enhanced_encoding": args.use_enhanced_encoding,
                    "report": report,
                },
                f,
                indent=2,
            )
        logger.info(f"Wrote report to {out_path}")


if __name__ == "__main__":
    main()
