"""Scraper for Board Game Arena YINSH game replays.

BGA has ~22,000 YINSH games played by human players, making it the
highest-quality source of game data. Replays are accessed via
authenticated HTTP endpoints.

BGA does not have a public API. This scraper uses internal endpoints
discovered from the BGA developer community. Use responsibly.

Authentication requires a BGA account (verified, >24h old, >2 games played).
There is a per-account daily limit on replay access.

Usage:
    scraper = BGAScraper()
    scraper.login('email@example.com', 'password')
    games = scraper.scrape_top_player_games(top_n=20, max_per_player=50)
"""

import json
import logging
import re
import time
import json as _json
import ssl
from http.cookiejar import Cookie, CookieJar
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urlencode
from urllib.request import (
    HTTPCookieProcessor, HTTPSHandler, Request, build_opener, urlopen,
)
from urllib.error import HTTPError, URLError

import certifi

from ...game.constants import Position, is_valid_position
from ..parsers.utils import positions_on_line

_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

logger = logging.getLogger(__name__)

YINSH_GAME_ID = 2423
BGA_BASE = "https://boardgamearena.com"
DEFAULT_DELAY = 3.0  # Generous delay to be respectful


class BGAScraper:
    """Scraper for Board Game Arena YINSH replays.

    Requires a BGA account for authentication. Replay access is
    rate-limited per account.
    """

    def __init__(self, delay: float = DEFAULT_DELAY):
        self.delay = delay
        self._last_request_time = 0.0
        self._cookie_jar = CookieJar()
        self._opener = build_opener(
            HTTPCookieProcessor(self._cookie_jar),
            HTTPSHandler(context=_SSL_CONTEXT),
        )
        self._logged_in = False
        self._request_token: Optional[str] = None

    def load_cookies(self, cookies_path: str) -> bool:
        """Load a session by importing cookies exported from a browser.

        BGA migrated their login to a JS-rendered flow that rejects
        urllib-based form posts. The practical workaround is to sign in
        with a real browser, then export the session cookies.

        Args:
            cookies_path: Path to a JSON file like
                {"PHPSESSID": "...", "TournoiEnLigneidt": "...", ...}
                Cookie names to include (set whichever you have):
                    PHPSESSID                  - session (short-lived)
                    TournoiEnLigneidt          - "remember me" cookie
                    TournoiEnLignetkt          - auth token
                    TournoiEnLigne_sso_id      - SSO session
                    TournoiEnLigne_sso_user    - SSO user hint
        Returns:
            True if cookies were loaded and a simple authenticated probe
            succeeded.
        """
        try:
            with open(cookies_path) as f:
                cookie_map = _json.load(f)
        except OSError as e:
            logger.error(f"Failed to read cookies file {cookies_path}: {e}")
            return False

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
            self._cookie_jar.set_cookie(c)

        # Probe: fetch /player, expect no "warn/redirect" bounce.
        # This also primes self._request_token for subsequent API calls.
        probe = self._fetch(f"{BGA_BASE}/player")
        if probe is None:
            logger.error("BGA cookie probe failed (request error)")
            return False

        if 'not_logged_user' in probe and 'logged_user' not in probe:
            logger.error("BGA cookies did not authenticate — session expired?")
            return False

        if self._request_token is None:
            logger.warning("BGA logged in but no requestToken found — "
                           "API calls may fail with CSRF errors")

        self._logged_in = True
        logger.info("BGA session loaded from cookies")
        return True

    def login(self, email: str, password: str) -> bool:
        """Authenticate with BGA.

        Args:
            email: BGA account email.
            password: BGA account password.

        Returns:
            True if login succeeded.
        """
        # Step 1: Get CSRF token from login page
        login_page = self._fetch(f"{BGA_BASE}/account")
        if not login_page:
            logger.error("Failed to fetch BGA login page")
            return False

        csrf_match = re.search(
            r'id=["\']csrf_token["\'][^>]*value=["\']([^"\']+)["\']',
            login_page
        )
        if not csrf_match:
            # Try alternate format
            csrf_match = re.search(
                r'name=["\']csrf_token["\'][^>]*value=["\']([^"\']+)["\']',
                login_page
            )
        if not csrf_match:
            logger.error("Could not extract CSRF token from BGA login page")
            return False

        csrf_token = csrf_match.group(1)

        # Step 2: POST login credentials
        login_data = urlencode({
            'email': email,
            'password': password,
            'rememberme': 'on',
            'redirect': 'join',
            'form_id': 'loginform',
            'csrf_token': csrf_token,
        }).encode('utf-8')

        login_url = f"{BGA_BASE}/account/account/login.html"
        try:
            req = Request(login_url, data=login_data, headers={
                'Content-Type': 'application/x-www-form-urlencoded',
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
                'Referer': f"{BGA_BASE}/account",
            })
            self._opener.open(req, timeout=30)
            self._last_request_time = time.time()
            self._logged_in = True
            logger.info("BGA login successful")
            return True
        except (HTTPError, URLError) as e:
            # A 302 redirect to /welcome is actually success
            if hasattr(e, 'code') and e.code in (301, 302):
                self._logged_in = True
                logger.info("BGA login successful (redirect)")
                return True
            logger.error(f"BGA login failed: {e}")
            return False

    def get_top_players(self, top_n: int = 50) -> List[Dict]:
        """Get top-ranked YINSH players from the hall of fame.

        Returns:
            List of player dicts with 'id' and 'name'.
        """
        if not self._logged_in:
            logger.error("Must login before scraping")
            return []

        params = urlencode({
            'game': YINSH_GAME_ID,
            'mode': 'arena',
            'start': 1,
        })
        url = f"{BGA_BASE}/halloffame/halloffame/getRanking.html?{params}"
        data = self._fetch_json(url)

        if not data or 'data' not in data:
            return []

        ranks = data['data'].get('ranks', [])
        players = []
        for rank in ranks[:top_n]:
            players.append({
                'id': rank.get('id'),
                'name': rank.get('name', 'unknown'),
                'elo': rank.get('elo', 0),
                'rank': rank.get('rank', 0),
            })

        logger.info(f"Found {len(players)} top YINSH players")
        return players

    def get_player_tables(self, player_id: int,
                          max_tables: int = 100) -> List[int]:
        """Get completed YINSH table IDs for a player.

        Args:
            player_id: BGA player ID.
            max_tables: Maximum tables to retrieve.

        Returns:
            List of table IDs.
        """
        table_ids = []
        page = 1

        while len(table_ids) < max_tables:
            params = urlencode({
                'player': player_id,
                'game_id': YINSH_GAME_ID,
                'finished': 1,
                'updateStats': 0,
                'page': page,
            })
            url = f"{BGA_BASE}/gamestats/gamestats/getGames.html?{params}"
            data = self._fetch_json(url)

            if not data or 'data' not in data:
                break

            tables = data['data'].get('tables', [])
            if not tables:
                break

            for table in tables:
                tid = table.get('table_id')
                if tid:
                    table_ids.append(int(tid))

            page += 1

        return table_ids[:max_tables]

    def scrape_game(self, table_id: int) -> Optional[Dict]:
        """Scrape a single game replay by table ID.

        Args:
            table_id: BGA table ID.

        Returns:
            Standardized game dict, or None on failure.
        """
        if not self._logged_in:
            logger.error("Must login before scraping")
            return None

        # Request archive generation (prerequisite)
        archive_url = (
            f"{BGA_BASE}/gamereview/gamereview/"
            f"requestTableArchive.html?table={table_id}"
        )
        self._fetch(archive_url)

        # Fetch replay logs
        logs_url = (
            f"{BGA_BASE}/archive/archive/"
            f"logs.html?table={table_id}&translated=true"
        )
        data = self._fetch_json(logs_url)

        if not data:
            return None

        # Check for error responses
        if isinstance(data, dict) and data.get('status') == 0:
            error = data.get('error', '')
            if 'limit' in error.lower():
                logger.warning(f"BGA replay limit reached: {error}")
            else:
                logger.warning(f"BGA error for table {table_id}: {error}")
            return None

        return self._parse_replay(data, table_id)

    def scrape_top_player_games(self, top_n: int = 20,
                                 max_per_player: int = 50,
                                 existing_tables: Optional[Set[int]] = None
                                 ) -> List[Dict]:
        """Scrape games from top-ranked YINSH players.

        Args:
            top_n: Number of top players to scrape from.
            max_per_player: Max games per player.
            existing_tables: Set of table IDs to skip.

        Returns:
            List of standardized game dicts.
        """
        seen = set(existing_tables or set())
        games = []

        players = self.get_top_players(top_n)
        for player in players:
            pid = player['id']
            tables = self.get_player_tables(pid, max_per_player)
            new_tables = [t for t in tables if t not in seen]
            logger.info(f"Player {player['name']} (ELO {player['elo']}): "
                        f"{len(new_tables)} new games")

            for tid in new_tables:
                seen.add(tid)
                game = self.scrape_game(tid)
                if game is not None:
                    games.append(game)

        logger.info(f"Scraped {len(games)} BGA YINSH games total")
        return games

    def _parse_replay(self, data: dict, table_id: int) -> Optional[Dict]:
        """Parse BGA replay log data into standardized game format.

        BGA replays are notification-based. The move data is nested at
        data.data.data.data (triple-nested 'data' key).
        """
        try:
            # Navigate the nested response structure
            log_data = data
            for _key in range(3):
                if isinstance(log_data, dict) and 'data' in log_data:
                    log_data = log_data['data']

            if not isinstance(log_data, (list, dict)):
                logger.warning(f"Unexpected replay data type for table {table_id}")
                return None

            # Extract notifications
            notifications = log_data if isinstance(log_data, list) else []
            if isinstance(log_data, dict):
                notifications = log_data.get('notifications', log_data.get('data', []))

            moves = _parse_bga_notifications(notifications)
            result = _extract_bga_result(notifications)
            players = _extract_bga_players(notifications, data)

            if not moves:
                logger.warning(f"No moves parsed from BGA table {table_id}")
                return None

            return {
                'source': 'bga',
                'game_id': f'bga_{table_id}',
                'players': players,
                'result': result,
                'moves': moves,
                'quality_tier': 'human',
            }

        except (KeyError, TypeError, IndexError) as e:
            logger.error(f"Failed to parse BGA replay {table_id}: {e}")
            return None

    def _fetch(self, url: str) -> Optional[str]:
        """Fetch URL as text with rate limiting and session cookies.

        Attaches BGA's X-Request-Token header (CSRF). The token is scraped
        from the first HTML page we load; API endpoints reject calls that
        omit it with "Invalid session information for this action".
        """
        self._rate_limit()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
        }
        if self._request_token:
            headers['X-Request-Token'] = self._request_token
            headers['X-Requested-With'] = 'XMLHttpRequest'
        try:
            req = Request(url, headers=headers)
            resp = self._opener.open(req, timeout=30)
            self._last_request_time = time.time()
            body = resp.read().decode('utf-8', errors='replace')
            # Opportunistically refresh the request token from any HTML body
            if self._request_token is None and '<html' in body[:200].lower():
                m = re.search(r"requestToken:\s*'([^']+)'", body)
                if m:
                    self._request_token = m.group(1)
            return body
        except (HTTPError, URLError) as e:
            logger.error(f"BGA fetch failed {url}: {e}")
            return None

    def _fetch_json(self, url: str) -> Optional[dict]:
        """Fetch URL as JSON."""
        text = self._fetch(url)
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from {url}: {e}")
            return None

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)


def _parse_bga_notifications(notifications: list) -> List[Dict]:
    """Parse BGA notification objects into standardized moves.

    BGA notification structure (typical):
    {
        "type": "notificationName",
        "log": "message template with ${vars}",
        "args": { ... move data ... },
        "uid": "...",
        "move_id": N
    }

    The exact notification types for YINSH need to be discovered from
    actual replay data. This parser handles the known/expected types
    and logs unrecognized ones for investigation.
    """
    moves = []

    for notif in notifications:
        if not isinstance(notif, dict):
            continue

        ntype = notif.get('type', '')
        args = notif.get('args', {})

        if not isinstance(args, dict):
            continue

        # Try to parse based on notification type
        # BGA game modules typically use types like:
        # placeRing, moveRing, removeMarkers, removeRing, etc.
        parsed = _parse_bga_notification(ntype, args)
        if parsed:
            moves.extend(parsed)

    return moves


def _parse_bga_notification(ntype: str, args: dict) -> List[Dict]:
    """Parse a single BGA notification into move dicts.

    This handles multiple naming conventions since the exact BGA
    YINSH notification types are discovered empirically.
    """
    ntype_lower = ntype.lower()

    # Determine player color from args
    player = _bga_player_color(args)

    # Ring placement
    if any(x in ntype_lower for x in ('placering', 'place_ring', 'ringplaced')):
        pos = _bga_extract_position(args)
        if pos and player:
            return [{'move_type': 'PLACE_RING', 'player': player, 'position': pos}]

    # Ring movement
    if any(x in ntype_lower for x in ('movering', 'move_ring', 'ringmoved')):
        src = _bga_extract_position(args, 'from') or _bga_extract_position(args, 'source')
        dst = _bga_extract_position(args, 'to') or _bga_extract_position(args, 'destination')
        if src and dst and player:
            return [{'move_type': 'MOVE_RING', 'player': player,
                     'source': src, 'destination': dst}]

    # Marker removal
    if any(x in ntype_lower for x in ('removemarker', 'remove_marker', 'markersremoved',
                                        'removerow', 'remove_row', 'rowremoved')):
        markers = _bga_extract_markers(args)
        if markers and player:
            return [{'move_type': 'REMOVE_MARKERS', 'player': player, 'markers': markers}]

    # Ring removal (after row capture)
    if any(x in ntype_lower for x in ('removering', 'remove_ring', 'ringremoved')):
        pos = _bga_extract_position(args)
        if pos and player:
            return [{'move_type': 'REMOVE_RING', 'player': player, 'position': pos}]

    # Score/end-game notifications — skip silently
    if any(x in ntype_lower for x in ('score', 'end', 'result', 'winner',
                                        'gameover', 'newround', 'updatescores',
                                        'simplenote', 'log')):
        return []

    # Unknown notification — log for investigation (only at debug level
    # to avoid spam during initial discovery)
    if ntype and args:
        logger.debug(f"Unhandled BGA notification: type={ntype}, args_keys={list(args.keys())}")

    return []


def _bga_player_color(args: dict) -> Optional[str]:
    """Extract player color from BGA notification args."""
    # BGA uses player_id or color fields
    color = args.get('color', args.get('player_color', ''))
    if isinstance(color, str):
        color_lower = color.lower()
        if 'white' in color_lower or color == '000000':
            return 'white'
        if 'black' in color_lower or color == 'ffffff':
            return 'black'

    # Try player_id mapping (player 1 = white, player 2 = black typically)
    pid = args.get('player_id', args.get('playerId', ''))
    if pid:
        # We can't determine color from player_id alone without the
        # game setup data. Return None and let the caller handle it.
        pass

    # Try direct player field
    player = args.get('player', '')
    if isinstance(player, str):
        if player.lower() in ('white', 'w', '1'):
            return 'white'
        if player.lower() in ('black', 'b', '2'):
            return 'black'

    return None


def _bga_extract_position(args: dict, prefix: str = '') -> Optional[str]:
    """Extract a board position from BGA notification args."""
    # Try various key naming conventions
    if prefix:
        keys = [f'{prefix}_col', f'{prefix}_row', f'{prefix}Col', f'{prefix}Row',
                f'{prefix}_x', f'{prefix}_y', f'{prefix}X', f'{prefix}Y']
    else:
        keys = ['col', 'row', 'x', 'y', 'position', 'pos', 'coord']

    # Direct position string (e.g., "E5")
    for key in ['position', 'pos', 'coord', f'{prefix}_position', f'{prefix}Position',
                f'{prefix}', f'{prefix}_pos']:
        val = args.get(key, '')
        if isinstance(val, str) and len(val) >= 2:
            try:
                pos = Position.from_string(val.upper())
                if is_valid_position(pos):
                    return str(pos)
            except (ValueError, IndexError):
                pass

    # Column + row as separate fields
    col_keys = [k for k in keys if 'col' in k.lower() or 'x' in k.lower()]
    row_keys = [k for k in keys if 'row' in k.lower() or 'y' in k.lower()]

    for ck in col_keys:
        for rk in row_keys:
            col_val = args.get(ck)
            row_val = args.get(rk)
            if col_val is not None and row_val is not None:
                try:
                    col = str(col_val).upper() if isinstance(col_val, str) else chr(ord('A') + int(col_val))
                    row = int(row_val)
                    pos = Position(col, row)
                    if is_valid_position(pos):
                        return str(pos)
                except (ValueError, TypeError):
                    pass

    return None


def _bga_extract_markers(args: dict) -> List[str]:
    """Extract marker positions from BGA notification args."""
    # Try list of positions
    for key in ['markers', 'positions', 'row', 'removed', 'marker_positions']:
        val = args.get(key, [])
        if isinstance(val, list) and val:
            positions = []
            for item in val:
                if isinstance(item, str):
                    try:
                        pos = Position.from_string(item.upper())
                        if is_valid_position(pos):
                            positions.append(str(pos))
                    except (ValueError, IndexError):
                        pass
                elif isinstance(item, dict):
                    pos = _bga_extract_position(item)
                    if pos:
                        positions.append(pos)
            if positions:
                return positions

    # Try start/end endpoints
    start = _bga_extract_position(args, 'from') or _bga_extract_position(args, 'start')
    end = _bga_extract_position(args, 'to') or _bga_extract_position(args, 'end')
    if start and end:
        line = positions_on_line(
            Position.from_string(start), Position.from_string(end)
        )
        if line:
            return [str(p) for p in line]

    return []


def _extract_bga_result(notifications: list) -> str:
    """Extract game result from BGA notifications."""
    for notif in reversed(notifications):
        if not isinstance(notif, dict):
            continue
        ntype = notif.get('type', '').lower()
        args = notif.get('args', {})

        if 'result' in ntype or 'end' in ntype or 'winner' in ntype:
            # Try to extract winner
            winner = args.get('winner', args.get('winner_color', ''))
            if isinstance(winner, str):
                if 'white' in winner.lower():
                    return 'white'
                if 'black' in winner.lower():
                    return 'black'

    return 'unknown'


def _extract_bga_players(notifications: list, data: dict) -> Dict:
    """Extract player information from BGA data."""
    players = {
        'white': {'name': 'unknown', 'rating': 0},
        'black': {'name': 'unknown', 'rating': 0},
    }

    # Try to find player info in the top-level data
    if isinstance(data, dict):
        for key in ('players', 'playerorder', 'gamestate'):
            val = data.get(key, {})
            if isinstance(val, dict):
                for pid, pdata in val.items():
                    if isinstance(pdata, dict):
                        color = pdata.get('color', '')
                        name = pdata.get('name', pdata.get('player_name', ''))
                        elo = pdata.get('elo', pdata.get('rank', 0))
                        if 'white' in str(color).lower() or color == '000000':
                            players['white'] = {'name': name, 'rating': int(elo or 0)}
                        elif 'black' in str(color).lower() or color == 'ffffff':
                            players['black'] = {'name': name, 'rating': int(elo or 0)}

    return players
