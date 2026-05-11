#!/usr/bin/env python3
"""Calibration baselines for the expert-agreement metric.

Runs the same per-position expert-move-agreement scoring used in
eval_vs_expert.py, but for non-network agents:

  - random: uniform random over legal moves
  - heuristic d=1, d=2, d=3: HeuristicAgent picks at search depth N

This anchors the 5% MAIN top-1 ceiling we've been seeing in supervised
runs: how much of that 5% is "trivial" (matched by any reasonable agent),
and what's the actual signal floor we should compare against?

Usage:
    python scripts/eval_baselines_vs_expert.py \\
        --games-dir expert_games/boardspace_human_holdout \\
        --out baselines_holdout.json
"""

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from yinsh_ml.agents.heuristic_agent import HeuristicAgent, HeuristicAgentConfig
from yinsh_ml.data.converter import GameConverter
from yinsh_ml.game.game_state import GameState
from yinsh_ml.utils.encoding import StateEncoder

logger = logging.getLogger(__name__)


def replay_positions(game_data: dict) -> Optional[List[Dict]]:
    """Replay a game and emit per-position expert-move records.

    Records a GameState (deep-copied at each step) so non-network agents can
    re-evaluate the position. Matches the same drop-policy as eval_vs_expert
    (any parse failure → drop the whole game).
    """
    result = game_data.get("result")
    moves = game_data.get("moves", [])
    if not moves:
        return None
    if result not in ("white", "black", "draw", "unknown"):
        return None

    encoder = StateEncoder()
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
        valid_indices = []
        for m in valid_moves:
            try:
                valid_indices.append(encoder.move_to_index(m))
            except (ValueError, IndexError):
                continue
        if expert_idx not in valid_indices:
            return None

        # Keep a deepcopy snapshot — agents are stateful w.r.t. transposition
        # tables but expect to be handed an arbitrary position.
        records.append({
            "state": gs.copy(),
            "expert_idx": expert_idx,
            "expert_move": move,
            "valid_moves": valid_moves,
            "phase": gs.phase.name,
            "move_num": len(gs.move_history),
        })
        if not gs.make_move(move):
            return None
    return records


def load_all_positions(games_dir: str) -> List[Dict]:
    out = []
    n_games = 0
    n_dropped = 0
    for json_file in sorted(Path(games_dir).glob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            n_dropped += 1
            continue
        games = data if isinstance(data, list) else [data]
        for game in games:
            recs = replay_positions(game)
            if recs is None:
                n_dropped += 1
                continue
            for r in recs:
                r["game_id"] = game.get("game_id", "")
            out.extend(recs)
            n_games += 1
    logger.info(f"Loaded {len(out)} positions from {n_games} games "
                f"({n_dropped} dropped)")
    return out


def score_agent(name: str, positions: List[Dict], agent_callable) -> Dict:
    """Iterate positions; ask agent for its move; check vs expert."""
    encoder = StateEncoder()
    buckets = defaultdict(lambda: {"n": 0, "top1": 0})
    t0 = time.time()

    for i, rec in enumerate(positions):
        if i and i % 500 == 0:
            rate = i / (time.time() - t0)
            logger.info(f"  [{name}] {i}/{len(positions)} ({rate:.0f} pos/s)")
        try:
            agent_move = agent_callable(rec["state"], rec["valid_moves"])
        except Exception as e:
            logger.warning(f"  [{name}] agent failed at pos {i}: {e}")
            continue
        try:
            agent_idx = encoder.move_to_index(agent_move)
        except (ValueError, IndexError):
            continue
        match = int(agent_idx == rec["expert_idx"])
        for key in ("ALL", rec["phase"]):
            b = buckets[key]
            b["n"] += 1
            b["top1"] += match

    report = {}
    for k, b in buckets.items():
        if b["n"] == 0:
            continue
        report[k] = {"n": b["n"], "top1_acc": b["top1"] / b["n"]}
    elapsed = time.time() - t0
    logger.info(f"  [{name}] done in {elapsed:.1f}s")
    return {"agent": name, "elapsed_s": round(elapsed, 1), "report": report}


def make_random_agent(seed: int):
    rng = random.Random(seed)
    def pick(state, valid_moves):
        return rng.choice(valid_moves)
    return pick


def make_heuristic_agent(depth: int):
    cfg = HeuristicAgentConfig(
        max_depth=depth,
        time_limit_seconds=5.0,
        use_transposition_table=True,
    )
    agent = HeuristicAgent(cfg)
    def pick(state, valid_moves):
        return agent.select_move(state)
    return pick


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-dir", default="expert_games/boardspace_human_holdout")
    parser.add_argument("--out", default="baselines_holdout.json")
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["random", "heuristic_d1", "heuristic_d3"],
        help="Subset of {random, heuristic_d1, heuristic_d2, heuristic_d3}",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    positions = load_all_positions(args.games_dir)
    if not positions:
        logger.error("No positions loaded")
        sys.exit(1)

    results = []
    for agent_name in args.agents:
        logger.info(f"Scoring {agent_name} on {len(positions)} positions...")
        if agent_name == "random":
            cb = make_random_agent(args.seed)
        elif agent_name.startswith("heuristic_d"):
            depth = int(agent_name.split("d")[-1])
            cb = make_heuristic_agent(depth)
        else:
            logger.error(f"unknown agent: {agent_name}")
            continue
        results.append(score_agent(agent_name, positions, cb))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {"games_dir": args.games_dir, "num_positions": len(positions),
             "results": results},
            f,
            indent=2,
        )
    logger.info(f"Wrote {out_path}")

    # Print summary
    print("\n" + "=" * 66)
    print(f"{'agent':<22} {'ALL n':>10} {'ALL top-1':>10} {'MAIN n':>10} {'MAIN top-1':>10}")
    print("-" * 66)
    for r in results:
        rpt = r["report"]
        a = rpt.get("ALL", {})
        m = rpt.get("MAIN_GAME", {})
        print(f"{r['agent']:<22} {a.get('n', 0):>10d} {a.get('top1_acc', 0):>10.3f} "
              f"{m.get('n', 0):>10d} {m.get('top1_acc', 0):>10.3f}")
    print("=" * 66)


if __name__ == "__main__":
    main()
