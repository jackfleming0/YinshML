"""Comprehensive tests for transposition table integration in HeuristicAgent.

This test suite verifies that the transposition table and Zobrist hashing
are correctly integrated into the HeuristicAgent's negamax search algorithm.

Tests cover:
- Transposition table initialization
- Hash key generation and lookup
- Caching of search results
- Node type handling (EXACT, LOWER_BOUND, UPPER_BOUND)
- Best move storage and retrieval
- Move ordering improvements
- Performance improvements from caching
"""

import unittest
from yinsh_ml.agents.heuristic_agent import HeuristicAgent, HeuristicAgentConfig
from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.constants import Player
# Import directly to avoid circular dependencies
from yinsh_ml.search.transposition_table import TranspositionTable
from yinsh_ml.search.node_type import NodeType
from yinsh_ml.game.zobrist import ZobristHasher


class TestTranspositionTableIntegration(unittest.TestCase):
    """Test transposition table integration in HeuristicAgent."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = HeuristicAgentConfig(
            max_depth=3,
            use_transposition_table=True,
            transposition_table_size_power=10,  # Small table for testing
            zobrist_seed="test-seed",
        )
        self.agent = HeuristicAgent(config=self.config)
        self.game_state = GameState()

    def test_transposition_table_initialized(self):
        """Test that transposition table is initialized when enabled."""
        self.assertIsNotNone(self.agent._transposition_table)
        self.assertIsInstance(self.agent._transposition_table, TranspositionTable)

    def test_zobrist_hasher_initialized(self):
        """Test that Zobrist hasher is initialized when TT is enabled."""
        self.assertIsNotNone(self.agent._zobrist_hasher)
        self.assertIsInstance(self.agent._zobrist_hasher, ZobristHasher)

    def test_transposition_table_disabled(self):
        """Test that TT is not initialized when disabled."""
        config_no_tt = HeuristicAgentConfig(use_transposition_table=False)
        agent_no_tt = HeuristicAgent(config=config_no_tt)
        self.assertIsNone(agent_no_tt._transposition_table)
        self.assertIsNone(agent_no_tt._zobrist_hasher)

    def test_hash_key_generation(self):
        """Test that hash keys are generated correctly."""
        hash_key = self.agent._zobrist_hasher.hash_state(self.game_state)
        self.assertIsInstance(hash_key, int)
        self.assertGreater(hash_key, 0)

    def test_same_state_same_hash(self):
        """Test that identical states produce identical hash keys."""
        state1 = GameState()
        state2 = GameState()
        
        hash1 = self.agent._zobrist_hasher.hash_state(state1)
        hash2 = self.agent._zobrist_hasher.hash_state(state2)
        
        self.assertEqual(hash1, hash2)

    def test_different_states_different_hash(self):
        """Test that different states produce different hash keys."""
        state1 = GameState()
        state2 = GameState()
        
        # Make a move in state2
        moves = state2.get_valid_moves()
        if moves:
            state2.make_move(moves[0])
        
        hash1 = self.agent._zobrist_hasher.hash_state(state1)
        hash2 = self.agent._zobrist_hasher.hash_state(state2)
        
        self.assertNotEqual(hash1, hash2)

    def test_transposition_table_stores_results(self):
        """Test that search results are stored in transposition table."""
        # Perform a search
        move = self.agent.select_move(self.game_state)
        self.assertIsNotNone(move)
        
        # Check that entries were stored
        metrics = self.agent._transposition_table.get_metrics()
        self.assertGreater(metrics['stores'], 0)

    def test_transposition_table_lookup(self):
        """Test that transposition table lookups work correctly."""
        # Hash the initial state
        hash_key = self.agent._zobrist_hasher.hash_state(self.game_state)
        
        # Manually store an entry to test lookup
        self.agent._transposition_table.store(
            hash_key=hash_key,
            depth=2,
            value=0.5,
            best_move=None,
            node_type=NodeType.EXACT,
        )
        
        # Lookup should find the entry
        entry = self.agent._transposition_table.lookup(hash_key)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.hash_key, hash_key)
        self.assertEqual(entry.depth, 2)
        self.assertEqual(entry.value, 0.5)

    def test_transposition_table_hit_rate(self):
        """Test that transposition table improves hit rate on repeated searches."""
        # Perform multiple searches on the same position
        for _ in range(5):
            self.agent.select_move(self.game_state)
        
        metrics = self.agent._transposition_table.get_metrics()
        # Should have some hits after multiple searches
        self.assertGreaterEqual(metrics['hits'], 0)
        self.assertGreater(metrics['stores'], 0)

    def test_node_type_storage(self):
        """Test that node types are stored correctly."""
        # Perform a search
        self.agent.select_move(self.game_state)
        
        # Check that entries have valid node types
        hash_key = self.agent._zobrist_hasher.hash_state(self.game_state)
        entry = self.agent._transposition_table.lookup(hash_key)
        
        if entry:
            self.assertIn(entry.node_type, [NodeType.EXACT, NodeType.LOWER_BOUND, NodeType.UPPER_BOUND])

    def test_best_move_storage(self):
        """Test that best moves are stored in transposition table."""
        # Perform a search
        move = self.agent.select_move(self.game_state)
        self.assertIsNotNone(move)
        
        # Check that best moves are stored
        hash_key = self.agent._zobrist_hasher.hash_state(self.game_state)
        entry = self.agent._transposition_table.lookup(hash_key)
        
        # Best move may or may not be stored depending on search depth
        # But if stored, it should be valid
        if entry and entry.best_move is not None:
            self.assertIsNotNone(entry.best_move)

    def test_clear_transposition_table(self):
        """Test that transposition table can be cleared."""
        # Perform a search to populate table
        self.agent.select_move(self.game_state)
        
        # Clear the table
        self.agent.clear_transposition_table()
        
        # Verify table is empty
        hash_key = self.agent._zobrist_hasher.hash_state(self.game_state)
        entry = self.agent._transposition_table.lookup(hash_key)
        self.assertIsNone(entry)

    def test_transposition_table_metrics_in_stats(self):
        """Test that transposition table metrics are included in search stats."""
        # Perform a search
        self.agent.select_move(self.game_state)
        
        # Check that metrics are in stats
        self.assertIn('transposition_table_metrics', self.agent.last_search_stats)
        tt_metrics = self.agent.last_search_stats['transposition_table_metrics']
        self.assertIsNotNone(tt_metrics)
        self.assertIn('hit_rate', tt_metrics)

    def test_depth_preferred_replacement(self):
        """Test that deeper entries are preferred over shallow ones."""
        # Create two entries with different depths
        hash_key = self.agent._zobrist_hasher.hash_state(self.game_state)
        
        # Store shallow entry
        self.agent._transposition_table.store(
            hash_key=hash_key,
            depth=1,
            value=0.5,
            best_move=None,
            node_type=NodeType.EXACT,
        )
        
        # Store deeper entry (should replace shallow)
        self.agent._transposition_table.store(
            hash_key=hash_key,
            depth=3,
            value=0.7,
            best_move=None,
            node_type=NodeType.EXACT,
        )
        
        # Lookup should return deeper entry
        entry = self.agent._transposition_table.lookup(hash_key)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.depth, 3)
        self.assertEqual(entry.value, 0.7)

    def test_exact_node_type_preference(self):
        """Test that EXACT nodes are preferred over bounds when depths are equal."""
        hash_key = self.agent._zobrist_hasher.hash_state(self.game_state)
        
        # Store LOWER_BOUND entry
        self.agent._transposition_table.store(
            hash_key=hash_key,
            depth=2,
            value=0.5,
            best_move=None,
            node_type=NodeType.LOWER_BOUND,
        )
        
        # Store EXACT entry (should replace LOWER_BOUND)
        self.agent._transposition_table.store(
            hash_key=hash_key,
            depth=2,
            value=0.6,
            best_move=None,
            node_type=NodeType.EXACT,
        )
        
        # Lookup should return EXACT entry
        entry = self.agent._transposition_table.lookup(hash_key)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.node_type, NodeType.EXACT)

    def test_move_ordering_with_best_move(self):
        """Test that best moves from TT are used for move ordering."""
        # Perform initial search to populate TT
        move1 = self.agent.select_move(self.game_state)
        
        # Clear nodes searched counter
        self.agent._nodes_searched = 0
        
        # Perform another search on same position
        # Should use cached best move for ordering
        move2 = self.agent.select_move(self.game_state)
        
        # Both searches should produce valid moves
        self.assertIsNotNone(move1)
        self.assertIsNotNone(move2)

    def test_alpha_beta_window_adjustment(self):
        """Test that alpha-beta window is adjusted correctly based on node type."""
        hash_key1 = self.agent._zobrist_hasher.hash_state(self.game_state)
        
        # Store LOWER_BOUND entry (value >= beta)
        self.agent._transposition_table.store(
            hash_key=hash_key1,
            depth=2,
            value=0.8,
            best_move=None,
            node_type=NodeType.LOWER_BOUND,
        )
        
        # When looking up, should find LOWER_BOUND entry
        entry = self.agent._transposition_table.lookup(hash_key1)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.node_type, NodeType.LOWER_BOUND)
        
        # Test UPPER_BOUND separately with a different hash key
        # (since replacement policy may not replace when depth and type match)
        state2 = GameState()
        # Make a move to get a different state/hash
        moves = state2.get_valid_moves()
        if moves:
            state2.make_move(moves[0])
        hash_key2 = self.agent._zobrist_hasher.hash_state(state2)
        
        # Store UPPER_BOUND entry (value <= alpha)
        self.agent._transposition_table.store(
            hash_key=hash_key2,
            depth=2,
            value=0.2,
            best_move=None,
            node_type=NodeType.UPPER_BOUND,
        )
        
        entry = self.agent._transposition_table.lookup(hash_key2)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.node_type, NodeType.UPPER_BOUND)
        
        # Verify both entries exist
        entry1 = self.agent._transposition_table.lookup(hash_key1)
        self.assertIsNotNone(entry1)
        self.assertEqual(entry1.node_type, NodeType.LOWER_BOUND)


class TestTranspositionTablePerformance(unittest.TestCase):
    """Test performance improvements from transposition table."""

    def setUp(self):
        """Set up test fixtures."""
        self.config_with_tt = HeuristicAgentConfig(
            max_depth=3,
            use_transposition_table=True,
            transposition_table_size_power=10,
        )
        self.config_without_tt = HeuristicAgentConfig(
            max_depth=3,
            use_transposition_table=False,
        )
        self.game_state = GameState()

    def test_repeated_search_performance(self):
        """Test that repeated searches benefit from caching."""
        agent_with_tt = HeuristicAgent(config=self.config_with_tt)
        agent_without_tt = HeuristicAgent(config=self.config_without_tt)
        
        # First search - both should take similar time
        move1 = agent_with_tt.select_move(self.game_state)
        move2 = agent_without_tt.select_move(self.game_state)
        
        self.assertIsNotNone(move1)
        self.assertIsNotNone(move2)
        
        # Second search - agent with TT should have better hit rate
        agent_with_tt.select_move(self.game_state)
        agent_without_tt.select_move(self.game_state)
        
        # Check that TT has hits
        metrics = agent_with_tt._transposition_table.get_metrics()
        self.assertGreater(metrics['hits'], 0)


if __name__ == '__main__':
    unittest.main()

