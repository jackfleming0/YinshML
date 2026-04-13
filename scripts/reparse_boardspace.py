#!/usr/bin/env python3
"""Re-parse all raw Boardspace SGF files with the current parser.

Overwrites the per-game JSONs. Much faster than re-scraping since files
are already downloaded.
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from yinsh_ml.data.parsers.boardspace_sgf import parse_boardspace_sgf

logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')
logger = logging.getLogger(__name__)

RAW = Path('expert_games/boardspace/raw')
JSON_DIR = Path('expert_games/boardspace/json')
JSON_DIR.mkdir(parents=True, exist_ok=True)

sgfs = list(RAW.rglob('*.sgf'))
print(f'Re-parsing {len(sgfs)} SGFs...', flush=True)

ok = 0
fail = 0
for i, sgf in enumerate(sgfs):
    try:
        text = sgf.read_text(encoding='utf-8', errors='replace')
        game = parse_boardspace_sgf(text)
        if game:
            out = JSON_DIR / f'{sgf.stem}.json'
            with open(out, 'w') as f:
                json.dump(game, f, indent=2)
            ok += 1
        else:
            fail += 1
    except Exception as e:
        fail += 1
        if fail <= 5:
            print(f'error on {sgf}: {e}')

    if (i + 1) % 2000 == 0:
        print(f'  {i+1}/{len(sgfs)}: {ok} ok, {fail} failed', flush=True)

print(f'Done: {ok} parsed, {fail} failed')
