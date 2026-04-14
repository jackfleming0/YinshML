#!/usr/bin/env python3
"""One-off probe: fetch a single BGA replay, dump raw notifications + parsed result.

Used to discover the actual notification schema (especially undo handling)
before extending the BGA parser. Not part of the regular pipeline.

Usage:
    python scripts/probe_bga_replay.py --cookies .bga_cookies.json
    python scripts/probe_bga_replay.py --cookies .bga_cookies.json --table 123456
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from yinsh_ml.data.scrapers.bga import (
    BGAScraper,
    _parse_bga_notifications,
    _extract_bga_result,
    _extract_bga_players_from_payload,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cookies', required=True, help='Path to .bga_cookies.json')
    ap.add_argument('--table', type=int, default=None,
                    help='Specific table ID. If omitted, picks the first '
                         'finished game from the top-ranked player.')
    ap.add_argument('--out-dir', default='bga_probe',
                    help='Where to write raw + parsed dumps (default: bga_probe/)')
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scraper = BGAScraper(delay=2.0)
    if not scraper.load_cookies(args.cookies):
        sys.exit('cookie load / probe failed')

    # Pick a table if not specified
    table_id = args.table
    if table_id is None:
        players = scraper.get_top_players(top_n=5)
        if not players:
            sys.exit('no top players returned')
        for p in players:
            tables = scraper.get_player_tables(p['id'], max_tables=5)
            if tables:
                table_id = tables[0]
                print(f'using table {table_id} from player {p["name"]} '
                      f'(elo {p["elo"]})')
                break
        if table_id is None:
            sys.exit('no tables found from top players')

    # Trigger archive generation
    scraper._fetch(
        f'https://boardgamearena.com/gamereview/gamereview/'
        f'requestTableArchive.html?table={table_id}'
    )

    # Fetch raw replay JSON
    raw = scraper._fetch_json(
        f'https://boardgamearena.com/archive/archive/'
        f'logs.html?table={table_id}&translated=true'
    )
    if raw is None:
        sys.exit('replay fetch returned None')

    raw_path = out_dir / f'raw_table_{table_id}.json'
    with open(raw_path, 'w') as f:
        json.dump(raw, f, indent=2)
    print(f'wrote raw response to {raw_path} '
          f'({raw_path.stat().st_size:,} bytes)')

    # Real structure: data.logs[i].data is the per-step notification list
    payload = raw.get('data') if isinstance(raw, dict) else None
    if not isinstance(payload, dict):
        sys.exit(f'unexpected response shape: {raw!r}')
    notifications = []
    for entry in payload.get('logs') or []:
        if isinstance(entry, dict) and isinstance(entry.get('data'), list):
            notifications.extend(entry['data'])
    players_raw = payload.get('players') or []

    # Build the same player_id -> color map the parser uses
    color_by_pid = {}
    for p in players_raw:
        if not isinstance(p, dict):
            continue
        pid = str(p.get('id', ''))
        color = (p.get('color') or '').lower().lstrip('#')
        if color in ('ffffff', 'fff', 'white'):
            color_by_pid[pid] = 'white'
        elif color in ('000000', '000', 'black'):
            color_by_pid[pid] = 'black'

    print(f'\n=== {len(notifications)} notifications extracted ===')
    print(f'players: {color_by_pid}')

    # Inventory the notification types
    type_counts = Counter()
    for n in notifications:
        if isinstance(n, dict):
            type_counts[n.get('type', '<no type>')] += 1
    print('\nNotification type histogram:')
    for t, c in sorted(type_counts.items(), key=lambda kv: -kv[1]):
        print(f'  {c:4d}  {t}')

    # Show a representative sample of each type (first occurrence)
    seen = set()
    samples = []
    for n in notifications:
        if not isinstance(n, dict):
            continue
        t = n.get('type', '<no type>')
        if t in seen:
            continue
        seen.add(t)
        samples.append(n)
    samples_path = out_dir / f'samples_table_{table_id}.json'
    with open(samples_path, 'w') as f:
        json.dump(samples, f, indent=2)
    print(f'\nwrote one-of-each-type samples to {samples_path}')

    # Run the parser and dump what it recognized
    parsed_moves = _parse_bga_notifications(notifications, color_by_pid)
    result = _extract_bga_result(notifications, color_by_pid)
    players = _extract_bga_players_from_payload(players_raw, color_by_pid)
    parsed = {
        'table_id': table_id,
        'result': result,
        'players': players,
        'n_moves_parsed': len(parsed_moves),
        'moves': parsed_moves,
    }
    parsed_path = out_dir / f'parsed_table_{table_id}.json'
    with open(parsed_path, 'w') as f:
        json.dump(parsed, f, indent=2)
    print(f'\nparser recognized {len(parsed_moves)} moves; wrote {parsed_path}')

    # Quick sanity: show first 5 + last 5 parsed moves
    if parsed_moves:
        print('\nFirst 5 parsed moves:')
        for m in parsed_moves[:5]:
            print(f'  {m}')
        if len(parsed_moves) > 10:
            print('...')
            print('Last 5 parsed moves:')
            for m in parsed_moves[-5:]:
                print(f'  {m}')


if __name__ == '__main__':
    main()
