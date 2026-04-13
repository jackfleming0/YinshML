#!/usr/bin/env python3
"""Gather expert YINSH games from all available sources.

Orchestrates scraping from Boardspace.net, CodinGame, and Board Game Arena,
validates all games through GameState, and converts to training data.

Usage:
    # Scrape from Boardspace only (no auth needed)
    python scripts/gather_expert_games.py --sources boardspace

    # Scrape from all sources
    python scripts/gather_expert_games.py --sources boardspace codingame bga

    # Limit number of games per source
    python scripts/gather_expert_games.py --sources boardspace --max-boardspace 500

    # BGA requires credentials
    python scripts/gather_expert_games.py --sources bga --bga-email user@example.com --bga-password pass

    # Validate and convert only (skip scraping)
    python scripts/gather_expert_games.py --validate-only --output-dir expert_games
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from yinsh_ml.data.scrapers.boardspace import BoardspaceScraper
from yinsh_ml.data.scrapers.codingame import CodinGameScraper
from yinsh_ml.data.scrapers.bga import BGAScraper
from yinsh_ml.data.validator import GameValidator
from yinsh_ml.data.converter import GameConverter

logger = logging.getLogger(__name__)


def scrape_boardspace(args) -> list:
    """Scrape games from Boardspace.net."""
    logger.info("=== Boardspace.net ===")
    scraper = BoardspaceScraper(
        output_dir=str(Path(args.output_dir) / 'boardspace'),
        delay=args.delay,
    )
    games = scraper.scrape_all(max_games=args.max_boardspace)
    logger.info(f"Boardspace: {len(games)} games scraped")
    return games


def scrape_codingame(args) -> list:
    """Scrape games from CodinGame."""
    logger.info("=== CodinGame ===")
    scraper = CodinGameScraper(delay=args.delay)
    games = scraper.scrape_leaderboard_games(
        top_n=args.cg_top_n,
        max_per_agent=args.max_codingame // max(1, args.cg_top_n),
    )
    logger.info(f"CodinGame: {len(games)} games scraped")
    return games


def scrape_bga(args) -> list:
    """Scrape games from Board Game Arena."""
    logger.info("=== Board Game Arena ===")
    scraper = BGAScraper(delay=args.delay)

    if args.bga_cookies:
        if not scraper.load_cookies(args.bga_cookies):
            logger.error("BGA cookie-based auth failed")
            return []
    elif args.bga_email and args.bga_password:
        if not scraper.login(args.bga_email, args.bga_password):
            logger.error("BGA login failed")
            return []
    else:
        logger.error("BGA requires --bga-cookies OR --bga-email + --bga-password")
        return []

    games = scraper.scrape_top_player_games(
        top_n=args.bga_top_n,
        max_per_player=args.max_bga // max(1, args.bga_top_n),
    )
    logger.info(f"BGA: {len(games)} games scraped")
    return games


def validate_games(games: list, strict: bool = False) -> list:
    """Validate all games through GameState. Returns only valid games."""
    logger.info(f"Validating {len(games)} games...")
    validator = GameValidator(strict=strict)
    valid = []
    invalid_reasons = {}

    for game in games:
        result = validator.validate_game(game)
        if result.valid:
            valid.append(game)
        else:
            reason = result.error_message[:80]
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1

    logger.info(f"Validation: {len(valid)}/{len(games)} valid "
                f"({len(games) - len(valid)} rejected)")

    if invalid_reasons:
        logger.info("Top rejection reasons:")
        for reason, count in sorted(invalid_reasons.items(),
                                     key=lambda x: -x[1])[:10]:
            logger.info(f"  {count:4d}x: {reason}")

    return valid


def convert_to_training_data(games: list, output_path: str):
    """Convert validated games to .npz training data."""
    logger.info(f"Converting {len(games)} games to training data...")
    converter = GameConverter()
    all_pairs = []

    for game in games:
        pairs = converter.convert_game(game)
        all_pairs.extend(pairs)

    if all_pairs:
        converter.save_training_data(all_pairs, output_path)
        logger.info(f"Saved {len(all_pairs)} training positions to {output_path}")
    else:
        logger.warning("No training pairs generated")

    return len(all_pairs)


def save_games_json(games: list, output_dir: str, source: str):
    """Save games as JSON files."""
    out_path = Path(output_dir) / source / 'json'
    out_path.mkdir(parents=True, exist_ok=True)

    for game in games:
        gid = game.get('game_id', 'unknown')
        json_path = out_path / f"{gid}.json"
        with open(json_path, 'w') as f:
            json.dump(game, f, indent=2)


def save_stats(all_games: list, valid_games: list, n_positions: int,
               output_dir: str):
    """Save summary statistics."""
    stats = {
        'total_games_scraped': len(all_games),
        'valid_games': len(valid_games),
        'invalid_games': len(all_games) - len(valid_games),
        'total_positions': n_positions,
        'by_source': {},
        'by_quality': {},
    }

    for game in valid_games:
        source = game.get('source', 'unknown')
        tier = game.get('quality_tier', 'unknown')
        stats['by_source'][source] = stats['by_source'].get(source, 0) + 1
        stats['by_quality'][tier] = stats['by_quality'].get(tier, 0) + 1

    stats_path = Path(output_dir) / 'stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Total games scraped: {stats['total_games_scraped']}")
    logger.info(f"Valid games:         {stats['valid_games']}")
    logger.info(f"Training positions:  {stats['total_positions']}")
    logger.info(f"By source: {stats['by_source']}")
    logger.info(f"By quality: {stats['by_quality']}")


def load_existing_games(output_dir: str) -> list:
    """Load already-scraped games from the output directory."""
    games = []
    for source_dir in Path(output_dir).iterdir():
        json_dir = source_dir / 'json'
        if not json_dir.is_dir():
            continue
        for json_file in json_dir.glob('*.json'):
            try:
                with open(json_file) as f:
                    game = json.load(f)
                games.append(game)
            except (json.JSONDecodeError, OSError):
                pass
    return games


def main():
    parser = argparse.ArgumentParser(
        description='Gather expert YINSH games from online sources'
    )

    # Sources
    parser.add_argument('--sources', nargs='+',
                       default=['boardspace'],
                       choices=['boardspace', 'codingame', 'bga', 'all'],
                       help='Sources to scrape (default: boardspace)')

    # Limits per source
    parser.add_argument('--max-boardspace', type=int, default=None,
                       help='Max Boardspace games (default: all)')
    parser.add_argument('--max-codingame', type=int, default=500,
                       help='Max CodinGame games (default: 500)')
    parser.add_argument('--max-bga', type=int, default=500,
                       help='Max BGA games (default: 500)')

    # Source-specific options
    parser.add_argument('--cg-top-n', type=int, default=30,
                       help='CodinGame: top N bots to scrape from')
    parser.add_argument('--bga-top-n', type=int, default=20,
                       help='BGA: top N players to scrape from')
    parser.add_argument('--bga-email', type=str, default=None,
                       help='BGA login email')
    parser.add_argument('--bga-password', type=str, default=None,
                       help='BGA login password (preferred: --bga-cookies)')
    parser.add_argument('--bga-cookies', type=str, default=None,
                       help='Path to JSON file with BGA session cookies '
                            '(see .bga_cookies.json format in TODO.md)')

    # General options
    parser.add_argument('--output-dir', type=str, default='expert_games',
                       help='Output directory (default: expert_games)')
    parser.add_argument('--delay', type=float, default=2.0,
                       help='Delay between HTTP requests (default: 2.0)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Skip scraping, only validate+convert existing data')
    parser.add_argument('--skip-convert', action='store_true',
                       help='Skip conversion to .npz')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if 'all' in args.sources:
        args.sources = ['boardspace', 'codingame', 'bga']

    # Scrape or load existing
    all_games = []
    if args.validate_only:
        all_games = load_existing_games(str(output_dir))
        logger.info(f"Loaded {len(all_games)} existing games")
    else:
        t0 = time.time()
        if 'boardspace' in args.sources:
            games = scrape_boardspace(args)
            save_games_json(games, str(output_dir), 'boardspace')
            all_games.extend(games)

        if 'codingame' in args.sources:
            games = scrape_codingame(args)
            save_games_json(games, str(output_dir), 'codingame')
            all_games.extend(games)

        if 'bga' in args.sources:
            games = scrape_bga(args)
            save_games_json(games, str(output_dir), 'bga')
            all_games.extend(games)

        elapsed = time.time() - t0
        logger.info(f"Scraping complete: {len(all_games)} games in {elapsed:.0f}s")

    # Validate
    valid_games = validate_games(all_games)

    # Save validated games
    validated_dir = output_dir / 'validated'
    validated_dir.mkdir(parents=True, exist_ok=True)
    validated_path = validated_dir / 'all_games.json'
    with open(validated_path, 'w') as f:
        json.dump(valid_games, f)
    logger.info(f"Saved {len(valid_games)} validated games to {validated_path}")

    # Convert to training data
    n_positions = 0
    if not args.skip_convert and valid_games:
        training_path = str(output_dir / 'training_data')
        n_positions = convert_to_training_data(valid_games, training_path)

    # Save stats
    save_stats(all_games, valid_games, n_positions, str(output_dir))


if __name__ == '__main__':
    main()
