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
                Cookie names to include — export ALL of these or
                /archive endpoints reject the session:
                    PHPSESSID                  - session (rotates each request)
                    TournoiEnLigne_sso_id      - SSO session
                    TournoiEnLigne_sso_user    - SSO user hint
                    TournoiEnLigneid           - persistent "remember me" id
                    TournoiEnLigneidt          - rotating session id
                    TournoiEnLignetk           - persistent auth token
                    TournoiEnLignetkt          - rotating auth token
                The id/tk pair (without trailing 't') are persistent and
                let BGA refresh the rotating idt/tkt pair. Without them
                BGA's session-promotion strands the connection partway
                through and /archive returns "not logged in".
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

        Real BGA replay structure (discovered 2026-04-13):
            { "status": "OK",
              "data": {
                "logs": [ { "data": [ <notification>, ... ] }, ... ],
                "players": [ { "id", "color", "name", ... }, ... ]
              }
            }
        Each notification has type ∈ {move, restartUndo, gameStateChange,
        confirmMove, updateReflexionTime, gameEnd, simpleNode}; only
        move/restartUndo/gameEnd carry game-relevant info.
        """
        try:
            payload = data.get('data') if isinstance(data, dict) else None
            if not isinstance(payload, dict):
                logger.warning(f"Unexpected replay shape for table {table_id}")
                return None

            # Build player_id → color map from the players list. BGA encodes
            # color as '#ffffff' (white) / '#000000' (black). The previously
            # checked-in mapping had these inverted.
            color_by_pid: Dict[str, str] = {}
            players_raw = payload.get('players') or []
            for p in players_raw:
                if not isinstance(p, dict):
                    continue
                pid = str(p.get('id', ''))
                color = (p.get('color') or '').lower().lstrip('#')
                if color in ('ffffff', 'fff', 'white'):
                    color_by_pid[pid] = 'white'
                elif color in ('000000', '000', 'black'):
                    color_by_pid[pid] = 'black'

            # Flatten notifications across all log entries.
            notifications = []
            for entry in payload.get('logs') or []:
                if not isinstance(entry, dict):
                    continue
                inner = entry.get('data')
                if isinstance(inner, list):
                    notifications.extend(inner)

            moves = _parse_bga_notifications(notifications, color_by_pid)
            result = _extract_bga_result(notifications, color_by_pid)
            players = _extract_bga_players_from_payload(players_raw, color_by_pid)

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


# BGA emits a single notification type 'move' for all in-game actions and
# discriminates the action via the `log` template string. We dispatch on
# substrings (templates carry literal English even when 'translated=true').
# Word boundaries are load-bearing: 'removes a ring from' contains the
# substring 'moves a ring from' (within 're-MOVES'), and a substring match
# would route ring-removals into the MOVE_RING branch.
_PLACE_RING_PAT = re.compile(r'\bplaces a ring on\b')
_PLACE_MARKER_PAT = re.compile(r'\bplaces a marker on\b')
_MOVE_RING_PAT = re.compile(r'\bmoves a ring from\b')
_REMOVE_ROW_PAT = re.compile(r'\bremoves a row of markers\b')
_REMOVE_RING_PAT = re.compile(r'\bremoves a ring from\b')


def _parse_bga_notifications(notifications: list,
                              color_by_pid: Dict[str, str]) -> List[Dict]:
    """Parse BGA notifications into standardized moves.

    Three-phase pipeline:
      1. Filter to game-relevant types (move, restartUndo).
      2. Apply restartUndo by popping the most recent `move` notification
         from the same player_id (BGA undoes one user action at a time).
      3. Walk the cleaned move stream and emit standardized moves. Skip
         place-marker notifications — the marker square is implicit in
         the next move-ring's locationFrom.
    """
    # Phase 1+2: filter and resolve undos.
    cleaned: List[dict] = []
    for notif in notifications:
        if not isinstance(notif, dict):
            continue
        ntype = notif.get('type')
        if ntype == 'move':
            cleaned.append(notif)
        elif ntype == 'restartUndo':
            args = notif.get('args') or {}
            pid = str(args.get('player_id', ''))
            for j in range(len(cleaned) - 1, -1, -1):
                prev_args = cleaned[j].get('args') or {}
                if str(prev_args.get('player_id', '')) == pid:
                    del cleaned[j]
                    break
            else:
                logger.debug(
                    f"BGA restartUndo by player {pid} found no prior move to pop"
                )
        # Other types (gameStateChange, confirmMove, updateReflexionTime,
        # gameEnd, simpleNode) carry no move data — handled separately.

    # Phase 3: convert remaining `move` notifications to our schema.
    moves: List[Dict] = []
    for notif in cleaned:
        args = notif.get('args')
        if not isinstance(args, dict):
            continue
        log = notif.get('log', '')
        pid = str(args.get('player_id', ''))
        player = color_by_pid.get(pid)
        if player is None:
            logger.debug(f"BGA move with unknown player_id={pid}; skipping")
            continue

        loc_from = args.get('locationFrom')
        loc_to = args.get('locationTo')

        if _PLACE_RING_PAT.search(log):
            pos = _coerce_pos(loc_to)
            if pos:
                moves.append({'move_type': 'PLACE_RING',
                              'player': player, 'position': pos})
        elif _PLACE_MARKER_PAT.search(log):
            # Implicit in the next MOVE_RING (locationFrom). No emission.
            continue
        elif _MOVE_RING_PAT.search(log):
            src = _coerce_pos(loc_from)
            dst = _coerce_pos(loc_to)
            if src and dst:
                moves.append({'move_type': 'MOVE_RING', 'player': player,
                              'source': src, 'destination': dst})
        elif _REMOVE_ROW_PAT.search(log):
            src = _coerce_pos(loc_from)
            dst = _coerce_pos(loc_to)
            if src and dst:
                line = positions_on_line(
                    Position.from_string(src), Position.from_string(dst)
                )
                if line:
                    moves.append({'move_type': 'REMOVE_MARKERS',
                                  'player': player,
                                  'markers': [str(p) for p in line]})
        elif _REMOVE_RING_PAT.search(log):
            # locationTo is a sentinel like '@0'/'@1' (the reserve slot);
            # the actual board square the ring sat on is in locationFrom.
            pos = _coerce_pos(loc_from)
            if pos:
                moves.append({'move_type': 'REMOVE_RING',
                              'player': player, 'position': pos})
        else:
            logger.debug(f"BGA move with unhandled log template: {log!r}")

    return moves


def _coerce_pos(val) -> Optional[str]:
    """Validate a BGA position string (e.g. 'E5'). Returns canonical form."""
    if not isinstance(val, str) or len(val) < 2 or val.startswith('@'):
        return None
    try:
        pos = Position.from_string(val.upper())
    except (ValueError, IndexError):
        return None
    return str(pos) if is_valid_position(pos) else None


def _extract_bga_result(notifications: list,
                        color_by_pid: Dict[str, str]) -> str:
    """Extract game result from the gameEnd notification."""
    for notif in notifications:
        if not isinstance(notif, dict) or notif.get('type') != 'gameEnd':
            continue
        args = notif.get('args') or {}
        pid = str(args.get('player_id', ''))
        winner_color = color_by_pid.get(pid)
        if winner_color:
            return winner_color
    return 'unknown'


def _extract_bga_players_from_payload(players_raw: list,
                                       color_by_pid: Dict[str, str]) -> Dict:
    """Build the standardized players block from data.players list.

    Note: the replay endpoint does not return ELO; ratings come from a
    separate hall-of-fame fetch. We leave rating=0 here and let the
    caller enrich if needed.
    """
    players = {
        'white': {'name': 'unknown', 'rating': 0},
        'black': {'name': 'unknown', 'rating': 0},
    }
    for p in players_raw:
        if not isinstance(p, dict):
            continue
        pid = str(p.get('id', ''))
        color = color_by_pid.get(pid)
        if color is None:
            continue
        players[color] = {
            'name': p.get('name', 'unknown'),
            'rating': 0,
        }
    return players
