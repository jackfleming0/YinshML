#!/usr/bin/env python3
"""Variant probe: skip the /player warmup, hit /archive directly.

The default load_cookies() path calls /player which issues Set-Cookie
headers that clear TournoiEnLigneidt + tkt — leaving us in a partial
session that /archive rejects. This script tries the original 5 cookies
straight against /archive.
"""

import argparse
import json
import logging
import re
import ssl
import sys
from http.cookiejar import Cookie, CookieJar
from pathlib import Path
from urllib.request import (
    HTTPCookieProcessor, HTTPSHandler, Request, build_opener,
)

import certifi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cookies', required=True)
    ap.add_argument('--table', type=int, required=True,
                    help='Specific table ID to fetch')
    ap.add_argument('--out-dir', default='bga_probe')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.cookies) as f:
        cookie_map = json.load(f)

    jar = CookieJar()
    for name, value in cookie_map.items():
        c = Cookie(
            version=0, name=name, value=str(value),
            port=None, port_specified=False,
            domain='.boardgamearena.com', domain_specified=True,
            domain_initial_dot=True,
            path='/', path_specified=True,
            secure=True, expires=None, discard=False,
            comment=None, comment_url=None, rest={}, rfc2109=False,
        )
        jar.set_cookie(c)

    opener = build_opener(
        HTTPCookieProcessor(jar),
        HTTPSHandler(context=ssl.create_default_context(cafile=certifi.where())),
    )
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
        'X-Requested-With': 'XMLHttpRequest',
    }

    # Step 1: We need a request token. The cleanest place to grab it without
    # triggering the cookie-clearing /player redirect is the gamereview page,
    # which renders an HTML wrapper around the archive.
    print('=== fetching /gamereview to get request token ===')
    review_url = (
        f'https://boardgamearena.com/gamereview?table={args.table}'
    )
    req = Request(review_url, headers=headers)
    resp = opener.open(req, timeout=30)
    body = resp.read().decode('utf-8', errors='replace')
    print(f'  status={resp.status}, {len(body)} bytes')
    print(f'  cookies after fetch: {[c.name for c in jar]}')

    m = re.search(r"requestToken:\s*'([^']+)'", body)
    if m:
        token = m.group(1)
        headers['X-Request-Token'] = token
        print(f'  got request token: {token[:20]}...')
    else:
        print('  WARN: no requestToken found in /gamereview body')

    # Step 2: Trigger archive build
    print('\n=== triggering archive generation ===')
    arc_url = (
        f'https://boardgamearena.com/gamereview/gamereview/'
        f'requestTableArchive.html?table={args.table}'
    )
    req = Request(arc_url, headers=headers)
    resp = opener.open(req, timeout=30)
    body = resp.read().decode('utf-8', errors='replace')
    print(f'  status={resp.status}, {len(body)} bytes')
    print(f'  body[:200]: {body[:200]}')

    # Step 3: Fetch the actual log JSON
    print('\n=== fetching archive logs ===')
    log_url = (
        f'https://boardgamearena.com/archive/archive/'
        f'logs.html?table={args.table}&translated=true'
    )
    req = Request(log_url, headers=headers)
    resp = opener.open(req, timeout=30)
    body = resp.read().decode('utf-8', errors='replace')
    print(f'  status={resp.status}, {len(body)} bytes')
    print(f'  body[:300]: {body[:300]}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f'raw_table_{args.table}_nowarmup.json'
    with open(raw_path, 'w') as f:
        f.write(body)
    print(f'\nwrote {raw_path}')


if __name__ == '__main__':
    main()
