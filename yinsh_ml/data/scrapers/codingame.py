"""Scraper for CodinGame YINSH bot-programming replays.

Fetches game replays from CodinGame's API and converts them to the
standardized game format.

CodinGame uses an undocumented API. Key endpoints:
- findLastBattlesByTestSessionHandle: list recent battles
- findByGameId: get full game replay

Only top-ranked bot games are useful for training data quality.
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional
import ssl
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

import certifi

from ..parsers.cg_notation import parse_cg_moves

_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

logger = logging.getLogger(__name__)

DEFAULT_DELAY = 1.5  # Seconds between API requests

# CodinGame API endpoints
_CG_API_BASE = "https://www.codingame.com/services"
_FIND_BY_GAME_ID = f"{_CG_API_BASE}/gameResultRemoteService/findByGameId"
_FIND_BATTLES = (
    f"{_CG_API_BASE}/gamesPlayersRankingRemoteService/"
    "findLastBattlesByTestSessionHandle"
)
_LEADERBOARD = (
    f"{_CG_API_BASE}/LeaderboardRemoteService/"
    "getFilteredPuzzleLeaderboard"
)


class CodinGameScraper:
    """Scraper for CodinGame YINSH bot programming replays.

    Usage:
        scraper = CodinGameScraper()
        game = scraper.scrape_game(game_id)
        games = scraper.scrape_leaderboard_games(top_n=50)
    """

    def __init__(self, delay: float = DEFAULT_DELAY):
        self.delay = delay
        self._last_request_time = 0.0

    def scrape_game(self, game_id: int,
                    user_id: Optional[int] = None) -> Optional[Dict]:
        """Scrape a single game replay by game ID.

        Args:
            game_id: CodinGame game ID.
            user_id: Optional user ID for the API call. Can be None.

        Returns:
            Standardized game dict, or None on failure.
        """
        payload = json.dumps([game_id, user_id]).encode('utf-8')
        data = self._post_json(_FIND_BY_GAME_ID, payload)
        if data is None:
            return None

        return self._parse_replay(data, game_id)

    def scrape_battles(self, test_session_handle: str,
                       max_games: int = 100) -> List[Dict]:
        """Scrape recent battles for a bot's test session.

        Args:
            test_session_handle: The test session handle from CG.
            max_games: Maximum number of games to scrape.

        Returns:
            List of standardized game dicts.
        """
        payload = json.dumps([test_session_handle, None]).encode('utf-8')
        battles = self._post_json(_FIND_BATTLES, payload)
        if not battles or not isinstance(battles, list):
            return []

        games = []
        for battle in battles[:max_games]:
            game_id = battle.get('gameId')
            if game_id is None:
                continue
            game = self.scrape_game(game_id)
            if game is not None:
                games.append(game)

        return games

    def _parse_replay(self, data: dict, game_id: int) -> Optional[Dict]:
        """Parse a CG game replay into standardized format.

        The replay contains:
        - agents: player metadata
        - frames: turn-by-turn game state
        - outputs: stdout from each agent (contains their move strings)
        - scores: final scores per player
        """
        try:
            agents = data.get('agents', [])
            outputs = data.get('outputs', {})
            scores = data.get('scores', {})

            # Extract player info
            players = {}
            player_map = {}  # CG index → 'white'/'black'
            for agent in agents:
                idx = agent.get('index', 0)
                color = 'white' if idx == 0 else 'black'
                player_map[str(idx)] = color

                codingamer = agent.get('codingamer', {})
                players[color] = {
                    'name': codingamer.get('pseudo', f'bot_{idx}'),
                    'rating': agent.get('score', 0),
                }

            # Determine result from scores
            result = self._determine_result(scores)

            # Extract moves from agent outputs
            moves = self._extract_moves(outputs, player_map)

            if not moves:
                logger.warning(f"No moves extracted from game {game_id}")
                return None

            return {
                'source': 'codingame',
                'game_id': f'cg_{game_id}',
                'players': players,
                'result': result,
                'moves': moves,
                'quality_tier': 'bot',
            }

        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Failed to parse CG replay {game_id}: {e}")
            return None

    @staticmethod
    def _determine_result(scores: dict) -> str:
        """Determine game result from CG scores dict."""
        if not scores:
            return 'unknown'

        # CG scores: player index → score. Higher = better.
        # In YINSH, the winner has the higher score.
        try:
            s0 = scores.get('0', scores.get(0, 0))
            s1 = scores.get('1', scores.get(1, 0))
            if s0 > s1:
                return 'white'
            elif s1 > s0:
                return 'black'
            return 'draw'
        except (TypeError, ValueError):
            return 'unknown'

    @staticmethod
    def _extract_moves(outputs: dict, player_map: dict) -> List[Dict]:
        """Extract move sequence from CG outputs.

        CG outputs is a dict: { player_index: { subframe: stdout_line } }
        or sometimes { player_index: [stdout_lines_per_frame] }.
        """
        all_moves = []

        # CG outputs can be structured in different ways depending on the game.
        # The most common format is: outputs[player_idx][frame_idx] = move_string
        # Moves alternate: frame 0 = init, then player 0, player 1, player 0, ...

        # Try to extract ordered moves
        if isinstance(outputs, dict):
            # Collect all (frame_number, player, move_string) tuples
            frame_moves = []

            for player_idx_str, frames in outputs.items():
                if player_idx_str == 'referee':
                    continue

                color = player_map.get(player_idx_str, 'white')

                if isinstance(frames, dict):
                    for frame_key, move_str in sorted(frames.items(),
                                                       key=lambda x: int(x[0])):
                        if move_str and move_str.strip():
                            frame_moves.append(
                                (int(frame_key), color, move_str.strip())
                            )
                elif isinstance(frames, list):
                    for frame_idx, move_str in enumerate(frames):
                        if move_str and move_str.strip():
                            frame_moves.append(
                                (frame_idx, color, move_str.strip())
                            )

            # Sort by frame number to get correct turn order
            frame_moves.sort(key=lambda x: x[0])

            for _frame, color, move_str in frame_moves:
                try:
                    parsed = parse_cg_moves(move_str, color)
                    all_moves.extend(parsed)
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse CG move '{move_str}': {e}")

        return all_moves

    def _post_json(self, url: str, payload: bytes) -> Optional[dict]:
        """POST JSON to CG API with rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

        try:
            req = Request(url, data=payload, headers={
                'Content-Type': 'application/json',
                'User-Agent': 'YinshML-Research/1.0',
            })
            with urlopen(req, timeout=30, context=_SSL_CONTEXT) as resp:
                self._last_request_time = time.time()
                return json.loads(resp.read().decode('utf-8'))
        except (HTTPError, URLError) as e:
            logger.error(f"CG API error for {url}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from {url}: {e}")
            return None


    def scrape_leaderboard(self, top_n: int = 50) -> List[Dict]:
        """Fetch the YINSH leaderboard to discover top bot agent IDs.

        Args:
            top_n: Number of top players to fetch.

        Returns:
            List of leaderboard entries with agentId, pseudo, rank, score.
        """
        # CG leaderboard API payload format
        payload = json.dumps([
            "yinsh", None,
            {"active": True, "column": "RANK", "filter": "ALL"},
            "global", None, 1, top_n
        ]).encode('utf-8')

        data = self._post_json(_LEADERBOARD, payload)
        if not data:
            return []

        # The response structure has a 'users' key with leaderboard entries
        users = data.get('users', data) if isinstance(data, dict) else data
        if not isinstance(users, list):
            logger.warning(f"Unexpected leaderboard response type: {type(users)}")
            return []

        entries = []
        for entry in users:
            if isinstance(entry, dict):
                entries.append({
                    'agent_id': entry.get('agentId'),
                    'codingamer_id': entry.get('codingamerId'),
                    'pseudo': entry.get('pseudo', 'unknown'),
                    'rank': entry.get('rank', 0),
                    'score': entry.get('score', 0),
                    'test_session_handle': entry.get('testSessionHandle'),
                })

        logger.info(f"Fetched {len(entries)} leaderboard entries")
        return entries

    def scrape_agent_battles(self, agent_id: int,
                             max_battles: int = 100) -> List[int]:
        """Get recent game IDs for a specific bot agent.

        Args:
            agent_id: CodinGame agent ID.
            max_battles: Maximum battles to fetch.

        Returns:
            List of game IDs.
        """
        payload = json.dumps([agent_id, None]).encode('utf-8')
        url = (f"{_CG_API_BASE}/gamesPlayersRankingRemoteService/"
               "findLastBattlesAndProgressByAgentId")
        data = self._post_json(url, payload)

        if not data:
            return []

        # Response may be a dict with 'battles' key, or a list directly
        battles = data.get('lastBattles', data) if isinstance(data, dict) else data
        if not isinstance(battles, list):
            return []

        game_ids = []
        for battle in battles[:max_battles]:
            if isinstance(battle, dict):
                gid = battle.get('gameId')
                if gid is not None:
                    game_ids.append(gid)

        return game_ids

    def scrape_leaderboard_games(self, top_n: int = 50,
                                  max_per_agent: int = 50,
                                  existing_game_ids: Optional[set] = None
                                  ) -> List[Dict]:
        """Scrape games from top-ranked bots on the leaderboard.

        Args:
            top_n: Number of top leaderboard entries to scrape from.
            max_per_agent: Maximum battles to fetch per agent.
            existing_game_ids: Set of game IDs to skip (already scraped).

        Returns:
            List of standardized game dicts.
        """
        seen = set(existing_game_ids or set())
        games = []

        # Get leaderboard
        entries = self.scrape_leaderboard(top_n)
        if not entries:
            logger.warning("Could not fetch leaderboard")
            return games

        logger.info(f"Scraping games from top {len(entries)} bots...")

        for entry in entries:
            agent_id = entry.get('agent_id')
            if not agent_id:
                continue

            # Get battle history for this agent
            game_ids = self.scrape_agent_battles(agent_id, max_per_agent)
            new_ids = [gid for gid in game_ids if gid not in seen]
            logger.info(f"Agent {entry['pseudo']} (rank {entry['rank']}): "
                        f"{len(new_ids)} new games")

            for gid in new_ids:
                seen.add(gid)
                game = self.scrape_game(gid)
                if game is not None:
                    games.append(game)

        logger.info(f"Scraped {len(games)} total CG games")
        return games
