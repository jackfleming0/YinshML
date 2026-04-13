"""Scraper for Little Golem YINSH games.

NOTE: As of 2025, YINSH is NOT available on Little Golem. The platform
hosts DVONN, TZAAR, and LYNGK from the GIPF project, but not YINSH.

This module is included for completeness and can be adapted if YINSH
is added to Little Golem in the future, or used as a template for
scraping games from other sources that use the official GIPF notation.

The parser supports the official GIPF-project notation format
(from gipf.com/yinsh/notations/notation.html).
"""

import logging
import time
import re
import json
from pathlib import Path
from typing import List, Dict, Optional
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

from ..parsers.lg_notation import parse_game_record

logger = logging.getLogger(__name__)

# Default politeness delay between requests (seconds)
DEFAULT_DELAY = 2.0


class LittleGolemScraper:
    """Scraper for Little Golem game server.

    Note: YINSH is not currently available on Little Golem.
    This class is a template that can be adapted for future use.
    """

    BASE_URL = "https://littlegolem.net"
    GAME_URL = f"{BASE_URL}/jsp/game/game.jsp?gid={{gid}}"
    PLAYER_GAMES_URL = (
        f"{BASE_URL}/jsp/info/player_game_list_txt.jsp"
        "?plid={plid}&gtid=yinsh"
    )

    def __init__(self, delay: float = DEFAULT_DELAY):
        """
        Args:
            delay: Seconds to wait between HTTP requests.
        """
        self.delay = delay
        self._last_request_time = 0.0

    def scrape_game(self, game_id: int) -> Optional[Dict]:
        """Scrape a single game by ID.

        Args:
            game_id: Little Golem game ID.

        Returns:
            Standardized game dict, or None if scraping fails.
        """
        url = self.GAME_URL.format(gid=game_id)
        html = self._fetch(url)
        if html is None:
            return None

        return self._parse_game_page(html, game_id)

    def scrape_player_games(self, player_id: int,
                            min_rating: int = 0) -> List[Dict]:
        """Scrape all YINSH games for a player.

        Args:
            player_id: Little Golem player ID.
            min_rating: Minimum player rating to include.

        Returns:
            List of standardized game dicts.
        """
        url = self.PLAYER_GAMES_URL.format(plid=player_id)
        text = self._fetch(url)
        if text is None:
            return []

        game_ids = self._parse_game_list(text)
        logger.info(f"Found {len(game_ids)} games for player {player_id}")

        games = []
        for gid in game_ids:
            game = self.scrape_game(gid)
            if game is not None:
                # Apply rating filter
                if min_rating > 0:
                    players = game.get('players', {})
                    w_rating = players.get('white', {}).get('rating', 0)
                    b_rating = players.get('black', {}).get('rating', 0)
                    if w_rating < min_rating or b_rating < min_rating:
                        continue
                games.append(game)

        return games

    def _fetch(self, url: str) -> Optional[str]:
        """Fetch a URL with rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

        try:
            req = Request(url, headers={
                'User-Agent': 'YinshML-Research/1.0 (academic research)',
            })
            with urlopen(req, timeout=30) as resp:
                self._last_request_time = time.time()
                return resp.read().decode('utf-8', errors='replace')
        except (HTTPError, URLError) as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def _parse_game_page(self, html: str, game_id: int) -> Optional[Dict]:
        """Parse a game page HTML to extract game data.

        This is a template — the actual HTML structure needs to be
        determined from live Little Golem pages.
        """
        # Extract move notation from the page
        # LG typically embeds game data in JavaScript or a specific HTML element
        # This pattern would need to be updated for the actual page structure
        move_match = re.search(
            r'(?:moveList|game_moves|notation)\s*[=:]\s*["\']([^"\']+)["\']',
            html, re.IGNORECASE
        )

        if not move_match:
            logger.warning(f"Could not find move data in game {game_id}")
            return None

        move_text = move_match.group(1)

        # Extract player info
        white_name = self._extract_player(html, 'white') or 'unknown'
        black_name = self._extract_player(html, 'black') or 'unknown'

        # Parse result
        result = self._extract_result(html)

        # Parse moves using the GIPF notation parser
        try:
            moves = parse_game_record(move_text)
        except (ValueError, IndexError) as e:
            logger.error(f"Failed to parse moves for game {game_id}: {e}")
            return None

        return {
            'source': 'little_golem',
            'game_id': f'lg_{game_id}',
            'players': {
                'white': {'name': white_name, 'rating': 0},
                'black': {'name': black_name, 'rating': 0},
            },
            'result': result or 'unknown',
            'moves': moves,
            'quality_tier': 'unknown',
        }

    @staticmethod
    def _extract_player(html: str, color: str) -> Optional[str]:
        """Extract player name from HTML."""
        pattern = rf'{color}\s*(?:player|:)\s*(\w+)'
        m = re.search(pattern, html, re.IGNORECASE)
        return m.group(1) if m else None

    @staticmethod
    def _extract_result(html: str) -> Optional[str]:
        """Extract game result from HTML."""
        if re.search(r'white\s+wins?|result.*white', html, re.IGNORECASE):
            return 'white'
        if re.search(r'black\s+wins?|result.*black', html, re.IGNORECASE):
            return 'black'
        if re.search(r'draw', html, re.IGNORECASE):
            return 'draw'
        return None

    @staticmethod
    def _parse_game_list(text: str) -> List[int]:
        """Parse a player game list text to extract game IDs."""
        game_ids = []
        for line in text.strip().split('\n'):
            m = re.search(r'gid=(\d+)', line)
            if m:
                game_ids.append(int(m.group(1)))
        return game_ids


def load_gipf_notation_file(path: str) -> List[Dict]:
    """Load games from a file in official GIPF notation format.

    This is useful for importing games from any source that uses
    the standard notation (e.g., game databases, tournament records).

    The file should contain one game per line, or a single game
    with moves separated by whitespace.

    Args:
        path: Path to the notation file.

    Returns:
        List of standardized game dicts.
    """
    with open(path) as f:
        content = f.read().strip()

    # Try to parse as a single game
    try:
        moves = parse_game_record(content)
        return [{
            'source': 'gipf_notation',
            'game_id': Path(path).stem,
            'players': {
                'white': {'name': 'unknown', 'rating': 0},
                'black': {'name': 'unknown', 'rating': 0},
            },
            'result': 'unknown',
            'moves': moves,
            'quality_tier': 'unknown',
        }]
    except (ValueError, IndexError) as e:
        logger.error(f"Failed to parse {path}: {e}")
        return []
