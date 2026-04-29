#!/usr/bin/env python3
"""Offline parser for BGA YINSH raw replay dumps.

Reads every `expert_games/bga/raw/*.json` produced by `bga_fetch.py` and
writes the standardized parsed form to `expert_games/bga/parsed/{tid}.json`.
Always overwrites — when the parser changes, re-run to refresh every
parsed/*.json without re-fetching (BGA has a daily cap; raw dumps are
precious).

Usage:
    python scripts/bga_parse.py
    python scripts/bga_parse.py --raw-dir expert_games/bga/raw \
        --parsed-dir expert_games/bga/parsed
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from yinsh_ml.data.scrapers.bga import BGAScraper


def _tid_from_stem(stem: str) -> int:
    try:
        return int(stem)
    except ValueError:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', default='expert_games/bga/raw',
                    help='Source directory of raw/{tid}.json '
                         '(default: expert_games/bga/raw/)')
    ap.add_argument('--parsed-dir', default='expert_games/bga/parsed',
                    help='Destination for parsed/{tid}.json '
                         '(default: expert_games/bga/parsed/)')
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    log = logging.getLogger('bga_parse')

    raw_dir = Path(args.raw_dir)
    parsed_dir = Path(args.parsed_dir)
    if not raw_dir.exists():
        sys.exit(f'raw dir does not exist: {raw_dir}')
    parsed_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(raw_dir.glob('*.json'))
    log.info(f"found {len(raw_files)} raw replay files in {raw_dir}")

    ok = 0
    failed = 0
    for raw_path in raw_files:
        tid = _tid_from_stem(raw_path.stem)
        if tid < 0:
            log.debug(f"skipping non-numeric filename: {raw_path.name}")
            continue
        try:
            with open(raw_path) as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log.warning(f"bad raw file {raw_path}: {e}")
            failed += 1
            continue
        parsed = BGAScraper.parse_raw(raw, tid)
        if parsed is None:
            failed += 1
            continue
        out_path = parsed_dir / f'{tid}.json'
        with open(out_path, 'w') as f:
            json.dump(parsed, f, indent=2)
        ok += 1

    log.info(f"parsed {ok} games, {failed} failures, wrote to {parsed_dir}")


if __name__ == '__main__':
    main()
