"""Parity tests for the cached Move.__hash__.

Move.__hash__ is memoized in self.__dict__['_hash'] after the first
call (BITBOARD_FOLLOWUP_PLAN.md Candidate A'). The invariant that
matters is: equal Moves must hash equal, and the cache must not
survive pickling — string hashes depend on PYTHONHASHSEED, so a
worker-process hash carried back to the parent would silently break
dict invariants.

Run: pytest yinsh_ml/tests/test_move_hash_cache.py -v
"""

import copy
import pickle
import unittest

from yinsh_ml.game.constants import Player, Position
from yinsh_ml.game.types import Move, MoveType


def _pos(s: str) -> Position:
    return Position.from_string(s)


class TestMoveHashEquivalence(unittest.TestCase):
    """If a == b then hash(a) == hash(b). The hash invariant."""

    def test_equal_place_ring_moves_hash_equal(self):
        a = Move(type=MoveType.PLACE_RING, player=Player.WHITE, source=_pos("E5"))
        b = Move(type=MoveType.PLACE_RING, player=Player.WHITE, source=_pos("E5"))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_equal_move_ring_moves_hash_equal(self):
        a = Move(type=MoveType.MOVE_RING, player=Player.BLACK,
                 source=_pos("E5"), destination=_pos("F6"))
        b = Move(type=MoveType.MOVE_RING, player=Player.BLACK,
                 source=_pos("E5"), destination=_pos("F6"))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_equal_remove_markers_moves_hash_equal(self):
        markers = (_pos("A2"), _pos("B3"), _pos("C4"), _pos("D5"), _pos("E6"))
        a = Move(type=MoveType.REMOVE_MARKERS, player=Player.WHITE, markers=markers)
        b = Move(type=MoveType.REMOVE_MARKERS, player=Player.WHITE, markers=markers)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_different_moves_can_have_different_hashes(self):
        # Sanity: nothing's collapsing to a constant.
        a = Move(type=MoveType.PLACE_RING, player=Player.WHITE, source=_pos("E5"))
        b = Move(type=MoveType.PLACE_RING, player=Player.BLACK, source=_pos("E5"))
        c = Move(type=MoveType.PLACE_RING, player=Player.WHITE, source=_pos("F6"))
        self.assertNotEqual(hash(a), hash(b))
        self.assertNotEqual(hash(a), hash(c))


class TestHashIdempotency(unittest.TestCase):
    """Hashing twice returns the same value, even after intervening operations."""

    def test_repeated_hash_returns_same_value(self):
        m = Move(type=MoveType.MOVE_RING, player=Player.WHITE,
                 source=_pos("E5"), destination=_pos("F6"))
        h1 = hash(m)
        h2 = hash(m)
        h3 = hash(m)
        self.assertEqual(h1, h2)
        self.assertEqual(h2, h3)

    def test_cache_actually_used(self):
        """After first hash, _hash slot should be populated. This is an
        implementation-level guard so the cache doesn't silently regress
        to a no-op (e.g., if someone removes the object.__setattr__)."""
        m = Move(type=MoveType.PLACE_RING, player=Player.WHITE, source=_pos("E5"))
        self.assertNotIn("_hash", m.__dict__)
        hash(m)
        self.assertIn("_hash", m.__dict__)


class TestPickleStripsHashCache(unittest.TestCase):
    """Critical: PYTHONHASHSEED is per-process randomized, so a cached
    hash from one process is wrong in another. Pickle must strip the
    cache so the receiving process recomputes."""

    def test_pickle_strips_hash_cache(self):
        m = Move(type=MoveType.MOVE_RING, player=Player.WHITE,
                 source=_pos("E5"), destination=_pos("F6"))
        hash(m)  # populate cache
        self.assertIn("_hash", m.__dict__)
        restored = pickle.loads(pickle.dumps(m))
        self.assertNotIn(
            "_hash", restored.__dict__,
            "Pickle must strip _hash — otherwise a worker-process hash "
            "would silently corrupt dicts in the parent process."
        )

    def test_unpickled_move_recomputes_consistent_hash(self):
        m = Move(type=MoveType.MOVE_RING, player=Player.WHITE,
                 source=_pos("E5"), destination=_pos("F6"))
        hash(m)
        restored = pickle.loads(pickle.dumps(m))
        # Within the same process, the recomputed hash matches the
        # original. Across processes (different seed) it would diverge,
        # but that's exactly why we strip on pickle.
        self.assertEqual(hash(restored), hash(m))

    def test_deepcopy_carries_hash_cache(self):
        # Same-process copies are safe. Carrying the cache avoids a
        # redundant recompute. Verify that's what happens.
        m = Move(type=MoveType.MOVE_RING, player=Player.WHITE,
                 source=_pos("E5"), destination=_pos("F6"))
        hash(m)
        m_copy = copy.deepcopy(m)
        self.assertEqual(hash(m_copy), hash(m))


class TestDictAndSetInvariants(unittest.TestCase):
    """End-to-end: Move can be used as a dict key / set member without
    surprises — the actual reason MCTS depends on the hash."""

    def test_move_works_as_dict_key(self):
        a = Move(type=MoveType.PLACE_RING, player=Player.WHITE, source=_pos("E5"))
        b = Move(type=MoveType.PLACE_RING, player=Player.WHITE, source=_pos("E5"))
        d = {}
        d[a] = "value"
        # a and b are equal; b should retrieve a's slot.
        self.assertEqual(d[b], "value")

    def test_move_works_in_set(self):
        a = Move(type=MoveType.PLACE_RING, player=Player.WHITE, source=_pos("E5"))
        b = Move(type=MoveType.PLACE_RING, player=Player.WHITE, source=_pos("E5"))
        s = {a, b}
        self.assertEqual(len(s), 1)


if __name__ == "__main__":
    unittest.main()
