#!/usr/bin/env python3
"""Minimal BGA test: login, fetch top players, fetch one game replay.

Dumps the raw notification types encountered so we can refine the parser.
"""
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from yinsh_ml.data.scrapers.bga import BGAScraper

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)

email = os.environ['BGA_EMAIL']
pw = os.environ['BGA_PASS']

s = BGAScraper(delay=2.5)
if not s.login(email, pw):
    sys.exit("login failed")

print("[top players]")
players = s.get_top_players(top_n=5)
for p in players:
    print(f"  {p}")

if not players:
    sys.exit("no players")

pid = players[0]['id']
tables = s.get_player_tables(pid, max_tables=3)
print(f"[tables for player {pid}]: {tables}")

if not tables:
    sys.exit("no tables")

# Fetch raw replay for inspection
tid = tables[0]
archive_url = f"https://boardgamearena.com/gamereview/gamereview/requestTableArchive.html?table={tid}"
s._fetch(archive_url)
logs_url = f"https://boardgamearena.com/archive/archive/logs.html?table={tid}&translated=true"
data = s._fetch_json(logs_url)

if data is None:
    sys.exit("no replay data")

# Save the raw response for analysis
out = Path('expert_games/bga_debug')
out.mkdir(parents=True, exist_ok=True)
with open(out / f'raw_{tid}.json', 'w') as f:
    json.dump(data, f, indent=2)
print(f"[saved raw] expert_games/bga_debug/raw_{tid}.json")

# Walk the nested response
def walk(d, depth=0):
    if depth > 6:
        return
    if isinstance(d, dict):
        keys = list(d.keys())[:20]
        print(f"{'  '*depth}dict keys: {keys}")
        for k in keys:
            walk(d[k], depth+1)
    elif isinstance(d, list):
        print(f"{'  '*depth}list len={len(d)}")
        if d:
            walk(d[0], depth+1)

walk(data)

# Look for notifications with 'type' key
def find_types(d, types):
    if isinstance(d, dict):
        if 'type' in d and isinstance(d['type'], str):
            types.add(d['type'])
        for v in d.values():
            find_types(v, types)
    elif isinstance(d, list):
        for x in d:
            find_types(x, types)

types = set()
find_types(data, types)
print(f"[notification types found]: {sorted(types)}")

# Try parsing
parsed = s._parse_replay(data, tid)
if parsed:
    print(f"[parsed] moves={len(parsed['moves'])} result={parsed['result']}")
    print(f"  first 5 moves: {parsed['moves'][:5]}")
else:
    print("[parsed] None")
