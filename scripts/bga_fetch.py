#!/usr/bin/env python3
"""Bulk crawler for BGA YINSH replays — network side only.

Walks top-ranked players, fetches their finished-game table list, and
persists each new replay's raw JSON to `expert_games/bga/raw/{tid}.json`.
Parsing is deferred to `scripts/bga_parse.py` so parser fixes can be
re-applied without re-hitting the network (BGA's per-account cap is ~200
fetches/day, so every raw response we save is precious).

Exits cleanly when BGA signals the daily replay cap has been hit;
`seen.json` persists state so the next session resumes where we left off.

Usage:
    python scripts/bga_fetch.py --cookies .bga_cookies.json
    python scripts/bga_fetch.py --cookies .bga_cookies.json \
        --raw-dir expert_games/bga/raw --max-fetches 150
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from yinsh_ml.data.scrapers.bga import BGAScraper, BGACapHit


def _load_seen(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logging.warning(f"could not read {path}: {e} — starting from empty")
        return {}


def _save_seen(path: Path, seen: dict) -> None:
    tmp = path.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(seen, f, indent=2, sort_keys=True)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cookies', required=True, help='Path to .bga_cookies.json')
    ap.add_argument('--raw-dir', default='expert_games/bga/raw',
                    help='Where to write raw/{tid}.json (default: '
                         'expert_games/bga/raw/)')
    ap.add_argument('--seen-file', default=None,
                    help='seen.json path (default: <raw-dir>/../seen.json)')
    ap.add_argument('--top-n', type=int, default=50,
                    help='Number of top-ranked players to walk (default: 50)')
    ap.add_argument('--max-per-player', type=int, default=50,
                    help='Max table IDs to pull per player (default: 50)')
    ap.add_argument('--max-fetches', type=int, default=200,
                    help='Hard cap on replay fetches this session '
                         '(default: 200, matches BGA daily limit)')
    ap.add_argument('--delay', type=float, default=8.0,
                    help='Seconds between requests (default: 8.0)')
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    log = logging.getLogger('bga_fetch')

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    seen_path = Path(args.seen_file) if args.seen_file else raw_dir.parent / 'seen.json'
    seen = _load_seen(seen_path)
    log.info(f"raw dir: {raw_dir}")
    log.info(f"seen file: {seen_path} ({len(seen)} known tids)")

    scraper = BGAScraper(delay=args.delay)
    if not scraper.load_cookies(args.cookies):
        sys.exit('cookie load / probe failed')

    players = scraper.get_top_players(top_n=args.top_n)
    if not players:
        sys.exit('no top players returned — session may be stale')

    fetched = 0
    cap_hit = False

    try:
        for player in players:
            if fetched >= args.max_fetches or cap_hit:
                break
            pid = player['id']
            tables = scraper.get_player_tables(pid, args.max_per_player)
            new_tables = [t for t in tables if str(t) not in seen]
            log.info(f"player {player['name']} (elo {player['elo']}): "
                     f"{len(new_tables)} new / {len(tables)} total")
            for tid in new_tables:
                if fetched >= args.max_fetches:
                    log.info(f"reached --max-fetches={args.max_fetches}; stopping")
                    break
                try:
                    raw = scraper.fetch_raw(tid)
                except BGACapHit as e:
                    log.warning(f"BGA cap hit on tid={tid}: {e} — exiting clean")
                    cap_hit = True
                    break
                fetched += 1
                if raw is None:
                    seen[str(tid)] = 'failed'
                    continue
                raw_path = raw_dir / f'{tid}.json'
                with open(raw_path, 'w') as f:
                    json.dump(raw, f)
                seen[str(tid)] = 'ok'
                if fetched % 10 == 0:
                    _save_seen(seen_path, seen)
                    log.info(f"progress: {fetched}/{args.max_fetches} fetched, "
                             f"{sum(1 for v in seen.values() if v == 'ok')} total on disk")
    finally:
        _save_seen(seen_path, seen)

    ok_total = sum(1 for v in seen.values() if v == 'ok')
    log.info(f"session done: fetched={fetched}, cap_hit={cap_hit}, "
             f"raw files on disk={ok_total}")


if __name__ == '__main__':
    main()
