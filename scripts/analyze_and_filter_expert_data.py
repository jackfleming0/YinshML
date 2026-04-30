#!/usr/bin/env python3
"""Analyze the supervised expert-games dataset and produce filtered subsets.

The "validated" Boardspace games include a lot of bot games (Dumbot, WeakBot,
etc.) and anonymous guests, which are weak opponents. The supervised model
trained on this data inherits weak play patterns (chain shuffling without
deliberate planning, no marker-flip strategy).

This script:
  1. Inventories the data (player counts, bot fraction, game lengths, etc.)
  2. Reports how many games survive various filters (no-bots, humans-only,
     regulars-only, etc.) so we can pick a filter that trades dataset size
     against quality.
  3. Optionally writes a filtered .npz that `run_supervised_pretraining.py`
     can train on with `--data <path>`.

Usage:
    # Just analyze:
    python scripts/analyze_and_filter_expert_data.py

    # Analyze + write a filtered npz with the "humans-only" filter:
    python scripts/analyze_and_filter_expert_data.py \\
        --filter humans_only \\
        --output expert_games/filtered_humans_only/training_data.npz
"""

import argparse
import collections
import json
import logging
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


# Known Boardspace AI players. These are deterministic bots of varying
# strength but all of them are well below human-intermediate play.
KNOWN_BOTS = frozenset({
    "Dumbot", "WeakBot", "SmartBot", "BestBot",
    "Bot", "Robot", "Computer",
})

# "guest" is the anonymous Boardspace login — usually one-off players,
# mostly very weak. Drop them.
ANONYMOUS = frozenset({"guest", "Guest", "GUEST"})


def player_name(game: dict, color: str) -> str:
    return game.get("players", {}).get(color, {}).get("name", "?")


def is_bot(name: str) -> bool:
    if name in KNOWN_BOTS:
        return True
    lname = name.lower()
    return any(kw in lname for kw in ("bot", " ai", "robot", "computer"))


def is_anonymous(name: str) -> bool:
    return name.lower() in {n.lower() for n in ANONYMOUS}


# ---- Filters ----

FILTERS: dict[str, tuple[str, Callable[[dict], bool]]] = {
    "all": (
        "All validated games (current default — what produced iter_0)",
        lambda g: True,
    ),
    "no_bots": (
        "Drop games where either player is a known bot",
        lambda g: not (is_bot(player_name(g, "white")) or is_bot(player_name(g, "black"))),
    ),
    "no_anonymous": (
        "Drop games where either player is the anonymous 'guest' account",
        lambda g: not (is_anonymous(player_name(g, "white")) or is_anonymous(player_name(g, "black"))),
    ),
    "humans_only": (
        "Drop bots AND anonymous guests — both players must have a real handle",
        lambda g: not (
            is_bot(player_name(g, "white"))
            or is_bot(player_name(g, "black"))
            or is_anonymous(player_name(g, "white"))
            or is_anonymous(player_name(g, "black"))
        ),
    ),
    "regulars_only": (
        "Both players must have appeared in ≥10 games (computed below)",
        None,  # populated dynamically with the regulars set
    ),
    "decisive_only": (
        "Drop draws and very short games (<30 moves) — keeps decisive intermediate-length games",
        lambda g: g.get("result") in ("white", "black") and len(g.get("moves", [])) >= 30,
    ),
    "humans_decisive": (
        "humans_only ∩ decisive_only — likely the strongest filter",
        None,  # composed below
    ),
}


def build_dynamic_filters(games: list[dict], min_appearances: int = 10) -> None:
    """Populate filters whose definitions depend on dataset stats."""
    counts = collections.Counter()
    for g in games:
        for color in ("white", "black"):
            counts[player_name(g, color)] += 1
    regulars = {n for n, c in counts.items()
                if c >= min_appearances and not is_bot(n) and not is_anonymous(n)}
    FILTERS["regulars_only"] = (
        f"Both players appear in ≥{min_appearances} games AND are humans ({len(regulars)} qualifying handles)",
        lambda g: (
            player_name(g, "white") in regulars
            and player_name(g, "black") in regulars
        ),
    )
    humans_filter = FILTERS["humans_only"][1]
    decisive_filter = FILTERS["decisive_only"][1]
    FILTERS["humans_decisive"] = (
        FILTERS["humans_decisive"][0],
        lambda g: humans_filter(g) and decisive_filter(g),
    )


def report(games: list[dict]) -> None:
    """Inventory the dataset and print survival counts per filter."""
    print(f"Total games: {len(games)}")
    print(f"Total moves (sum): {sum(len(g.get('moves', [])) for g in games)}")

    # Player name frequency
    counts = collections.Counter()
    for g in games:
        for color in ("white", "black"):
            counts[player_name(g, color)] += 1
    print(f"Unique player names: {len(counts)}")

    bot_slots = sum(c for n, c in counts.items() if is_bot(n))
    anon_slots = sum(c for n, c in counts.items() if is_anonymous(n))
    human_slots = sum(c for n, c in counts.items() if not is_bot(n) and not is_anonymous(n))
    total_slots = sum(counts.values())
    print(f"\nPlayer slots (each game has 2):")
    print(f"  Bot:       {bot_slots:5d}  ({bot_slots/total_slots:.1%})")
    print(f"  Anonymous: {anon_slots:5d}  ({anon_slots/total_slots:.1%})")
    print(f"  Human:     {human_slots:5d}  ({human_slots/total_slots:.1%})")

    print("\nTop 10 player names by appearance:")
    for n, c in counts.most_common(10):
        flag = " (BOT)" if is_bot(n) else (" (anon)" if is_anonymous(n) else "")
        print(f"  {n:20s} {c:5d}{flag}")

    print("\nFilter survival:")
    print(f"{'filter':<22} {'games':>6} {'positions':>10}  description")
    print("-" * 100)
    for name, (desc, fn) in FILTERS.items():
        if fn is None:
            print(f"  {name:<20} (skipped — dynamic filter not built)")
            continue
        survivors = [g for g in games if fn(g)]
        positions = sum(len(g.get("moves", [])) for g in survivors)
        pct = 100 * len(survivors) / len(games) if games else 0
        print(f"{name:<22} {len(survivors):>6} {positions:>10}  {desc} ({pct:.0f}%)")


# ---- Optional npz writer ----

def _convert_and_save(games: list[dict], output_path: Path,
                     use_enhanced_encoding: bool = False) -> None:
    """Convert filtered games to (states, policies, values) and save as .npz."""
    from yinsh_ml.data.converter import GameConverter
    from yinsh_ml.utils.encoding import StateEncoder
    from yinsh_ml.utils.enhanced_encoding import EnhancedStateEncoder

    encoder = EnhancedStateEncoder() if use_enhanced_encoding else StateEncoder()
    converter = GameConverter(encoder=encoder)

    pairs = []
    for game in games:
        try:
            pairs.extend(converter.convert_game(game))
        except Exception as e:
            logger.warning(f"convert_game failed: {e}")

    if not pairs:
        logger.error("No training pairs produced from filtered games.")
        return

    states = np.array([p["state"] for p in pairs], dtype=np.float32)
    policies = np.array([p["policy"] for p in pairs], dtype=np.float32)
    values = np.array([p["value"] for p in pairs], dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path,
                        states=states, policies=policies, values=values)
    logger.info(
        f"Wrote {len(pairs):,} training positions from {len(games):,} games to {output_path}"
    )
    logger.info(
        f"  states: {states.shape}, policies: {policies.shape}, values: {values.shape}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path("expert_games/validated/all_games.json"),
        help="Path to validated games JSON",
    )
    parser.add_argument(
        "--filter", type=str, default=None, choices=list(FILTERS.keys()),
        help="If set, write a filtered .npz instead of just analyzing",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Where to write the filtered .npz (required if --filter is set)",
    )
    parser.add_argument(
        "--min-appearances", type=int, default=10,
        help="Threshold for the regulars_only filter (default: 10)",
    )
    parser.add_argument(
        "--use-enhanced-encoding", action="store_true",
        help="Use 15-channel encoding instead of 6-channel (must match training config)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if not args.input.exists():
        logger.error(f"Input not found: {args.input}")
        sys.exit(1)

    logger.info(f"Loading {args.input}...")
    with open(args.input) as f:
        games = json.load(f)

    build_dynamic_filters(games, min_appearances=args.min_appearances)
    report(games)

    if args.filter is None:
        return

    if args.output is None:
        parser.error("--output is required when --filter is given")

    if FILTERS[args.filter][1] is None:
        parser.error(f"Filter {args.filter!r} could not be built (no qualifying data?)")

    survivors = [g for g in games if FILTERS[args.filter][1](g)]
    print(f"\nApplying filter '{args.filter}' → {len(survivors):,} games surviving "
          f"({100*len(survivors)/len(games):.0f}% of input)")

    if not survivors:
        logger.error("Filter produced 0 games. Aborting.")
        sys.exit(1)

    _convert_and_save(survivors, args.output, args.use_enhanced_encoding)


if __name__ == "__main__":
    main()
