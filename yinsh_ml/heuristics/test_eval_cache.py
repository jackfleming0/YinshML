"""Parity tests for the YinshHeuristics evaluate_position cache.

Cached results MUST be bit-identical to uncached. The cache key in
evaluator.py:_cache_key is supposed to include every input the
uncached path reads from game_state — these tests exercise that
contract across diverse positions (multiple phases, both players,
mid-game scoring states) so a missed key component fails noisily.

Run: pytest yinsh_ml/heuristics/test_eval_cache.py -v
"""

import random
import unittest
from typing import List

from yinsh_ml.game.constants import Player
from yinsh_ml.game.game_state import GameState
from yinsh_ml.heuristics import YinshHeuristics


def _generate_states(num_states: int = 50, seed: int = 1337) -> List[GameState]:
    """Random-walk a game and snapshot states across phases.

    Captures a deepcopy after each move so later mutations don't poison
    earlier snapshots. Stops early on terminal states; resumes a fresh
    game until num_states is reached.
    """
    import copy

    rng = random.Random(seed)
    states: List[GameState] = []
    state = GameState()
    while len(states) < num_states:
        valid = state.get_valid_moves()
        if not valid or state.is_terminal():
            state = GameState()
            continue
        move = rng.choice(valid)
        if state.make_move(move):
            states.append(copy.deepcopy(state))
    return states


class TestEvalCacheParity(unittest.TestCase):
    """Cached and uncached evaluate_position must agree exactly."""

    @classmethod
    def setUpClass(cls):
        # Disable forced sequence detection so test runtime is reasonable
        # — same flag MCTS hybrid mode uses, so this is the realistic path.
        cls.cached = YinshHeuristics(
            enable_forced_sequence_detection=False,
            eval_cache_size=10_000,
        )
        cls.uncached = YinshHeuristics(
            enable_forced_sequence_detection=False,
            eval_cache_size=0,
        )
        cls.states = _generate_states(num_states=80)

    def test_cached_matches_uncached(self):
        """Across diverse positions and both players, cache returns the
        same float as a direct uncached evaluation."""
        for i, state in enumerate(self.states):
            for player in (Player.WHITE, Player.BLACK):
                ref = self.uncached.evaluate_position(state, player)
                got = self.cached.evaluate_position(state, player)
                self.assertEqual(
                    ref, got,
                    f"state #{i} player={player.name}: cache={got!r} ref={ref!r}",
                )

    def test_cache_returns_consistent_value_on_repeated_calls(self):
        """Same (state, player) → same float, every time. Sanity for the
        cache itself in case of mutation hazards."""
        ev = YinshHeuristics(
            enable_forced_sequence_detection=False,
            eval_cache_size=10_000,
        )
        for state in self.states[:20]:
            v1 = ev.evaluate_position(state, Player.WHITE)
            v2 = ev.evaluate_position(state, Player.WHITE)
            v3 = ev.evaluate_position(state, Player.WHITE)
            self.assertEqual(v1, v2)
            self.assertEqual(v2, v3)


class TestEvalCacheStats(unittest.TestCase):
    """Hit/miss counters and clear_cache lifecycle."""

    def setUp(self):
        self.ev = YinshHeuristics(
            enable_forced_sequence_detection=False,
            eval_cache_size=10_000,
        )
        self.states = _generate_states(num_states=10, seed=42)

    def test_first_pass_all_misses_second_pass_all_hits(self):
        """A two-pass walk over the same states should yield N misses
        and then N hits — confirms the cache is actually doing work."""
        n = len(self.states)
        for state in self.states:
            self.ev.evaluate_position(state, Player.WHITE)
        first = self.ev.cache_stats()
        self.assertEqual(first["misses"], n)
        self.assertEqual(first["hits"], 0)

        for state in self.states:
            self.ev.evaluate_position(state, Player.WHITE)
        second = self.ev.cache_stats()
        self.assertEqual(second["misses"], n)        # unchanged
        self.assertEqual(second["hits"], n)          # one hit per state
        self.assertAlmostEqual(second["hit_rate"], 0.5)

    def test_clear_cache_resets_state_and_counters(self):
        for state in self.states:
            self.ev.evaluate_position(state, Player.WHITE)
        self.assertGreater(self.ev.cache_stats()["size"], 0)
        self.ev.clear_cache()
        cleared = self.ev.cache_stats()
        self.assertEqual(cleared["size"], 0)
        self.assertEqual(cleared["hits"], 0)
        self.assertEqual(cleared["misses"], 0)


class TestEvalCacheDisabled(unittest.TestCase):
    """eval_cache_size=0 must skip the cache entirely without errors."""

    def test_disabled_cache_does_not_record_any_entries(self):
        ev = YinshHeuristics(
            enable_forced_sequence_detection=False,
            eval_cache_size=0,
        )
        state = GameState()
        ev.evaluate_position(state, Player.WHITE)
        ev.evaluate_position(state, Player.WHITE)
        stats = ev.cache_stats()
        self.assertEqual(stats["size"], 0)
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 0)


class TestCacheKeyDiscriminatesScoreAndMoveCount(unittest.TestCase):
    """Two states with identical board/side/phase but different scores
    or move counts must NOT collide in the cache. This guards against
    silent staleness — the kind of bug that only shows up as ML
    training drift weeks later."""

    def test_different_scores_produce_different_cache_keys(self):
        # Two GameState instances; one with score (1, 0), the other (0, 0).
        # Without scores in the key these would collide on Zobrist alone
        # (board/side/phase identical) and one would shadow the other.
        a = GameState()
        b = GameState()
        b.white_score = 1

        ev = YinshHeuristics(
            enable_forced_sequence_detection=False,
            eval_cache_size=10_000,
        )
        key_a = ev._cache_key(a, Player.WHITE)
        key_b = ev._cache_key(b, Player.WHITE)
        self.assertNotEqual(key_a, key_b)

    def test_different_move_counts_produce_different_cache_keys(self):
        # Phase classification (EARLY/MID/LATE) keys off len(move_history),
        # not on phase-the-enum. Two states with identical Zobrist could
        # theoretically reach via different move counts (transposition);
        # they must not share a cache slot.
        a = GameState()
        b = GameState()
        # Fake out the move history without altering the board so the
        # fingerprint stays stable. (The eval function only reads len()
        # of move_history, not its contents.)
        b.move_history = [None] * 10  # type: ignore[list-item]

        ev = YinshHeuristics(
            enable_forced_sequence_detection=False,
            eval_cache_size=10_000,
        )
        key_a = ev._cache_key(a, Player.WHITE)
        key_b = ev._cache_key(b, Player.WHITE)
        self.assertNotEqual(key_a, key_b)


if __name__ == "__main__":
    unittest.main()
