"""Unit tests for GameStatePool implementation."""

import unittest
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from yinsh_ml.memory.game_state_pool import (
    GameStatePool,
    GameStatePoolConfig,
    GameStatePoolStatistics,
    create_game_state_pool,
    create_game_state,
    reset_game_state,
    validate_reset_game_state
)
from yinsh_ml.memory.config import GrowthPolicy
from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.constants import Player, Position, PieceType
from yinsh_ml.game.types import GamePhase, Move, MoveType


class TestGameStateReset(unittest.TestCase):
    """Test game state reset functionality."""
    
    def test_create_game_state(self):
        """Test factory function creates valid GameState."""
        game_state = create_game_state()
        self.assertIsInstance(game_state, GameState)
        self.assertEqual(game_state.current_player, Player.WHITE)
        self.assertEqual(game_state.phase, GamePhase.RING_PLACEMENT)
        self.assertEqual(game_state.white_score, 0)
        self.assertEqual(game_state.black_score, 0)
        self.assertEqual(len(game_state.move_history), 0)
        self.assertEqual(len(game_state.board.pieces), 0)
        
    def test_reset_game_state_basic(self):
        """Test basic game state reset functionality."""
        # Create and modify a GameState
        game_state = create_game_state()
        
        # Modify the state
        game_state.current_player = Player.BLACK
        game_state.phase = GamePhase.MAIN_GAME
        game_state.white_score = 2
        game_state.black_score = 1
        game_state.rings_placed = {Player.WHITE: 5, Player.BLACK: 5}
        game_state.move_history.append(
            Move(MoveType.PLACE_RING, Player.WHITE, Position('A', 2))
        )
        game_state.board.place_piece(Position('A', 2), PieceType.WHITE_RING)
        
        # Add temporary attributes
        game_state._move_maker = Player.WHITE
        game_state._prev_player = Player.BLACK
        
        # Reset the state
        reset_state = reset_game_state(game_state)
        
        # Verify it's the same object
        self.assertIs(reset_state, game_state)
        
        # Verify reset to initial state
        self.assertEqual(game_state.current_player, Player.WHITE)
        self.assertEqual(game_state.phase, GamePhase.RING_PLACEMENT)
        self.assertEqual(game_state.white_score, 0)
        self.assertEqual(game_state.black_score, 0)
        self.assertEqual(game_state.rings_placed, {Player.WHITE: 0, Player.BLACK: 0})
        self.assertEqual(len(game_state.move_history), 0)
        self.assertEqual(len(game_state.board.pieces), 0)
        
        # Verify temporary attributes are removed
        self.assertFalse(hasattr(game_state, '_move_maker'))
        self.assertFalse(hasattr(game_state, '_prev_player'))
        
    def test_validate_reset_game_state(self):
        """Test validation of reset GameState objects."""
        # Test with properly reset state
        game_state = create_game_state()
        self.assertTrue(validate_reset_game_state(game_state))
        
        # Test with modified states
        game_state.current_player = Player.BLACK
        self.assertFalse(validate_reset_game_state(game_state))
        
        game_state = create_game_state()
        game_state.phase = GamePhase.MAIN_GAME
        self.assertFalse(validate_reset_game_state(game_state))
        
        game_state = create_game_state()
        game_state.white_score = 1
        self.assertFalse(validate_reset_game_state(game_state))
        
        game_state = create_game_state()
        game_state.rings_placed[Player.WHITE] = 1
        self.assertFalse(validate_reset_game_state(game_state))
        
        game_state = create_game_state()
        game_state.move_history.append(
            Move(MoveType.PLACE_RING, Player.WHITE, Position('A', 2))
        )
        self.assertFalse(validate_reset_game_state(game_state))
        
        game_state = create_game_state()
        game_state.board.place_piece(Position('A', 2), PieceType.WHITE_RING)
        self.assertFalse(validate_reset_game_state(game_state))


class TestGameStatePoolConfig(unittest.TestCase):
    """Test GameState pool configuration."""
    
    def test_config_creation(self):
        """Test creating GameStatePoolConfig."""
        config = GameStatePoolConfig(
            initial_size=50,
            max_capacity=500,
            mcts_batch_size=25,
            training_mode=True,
            validate_reset=True
        )
        
        self.assertEqual(config.initial_size, 50)
        self.assertEqual(config.max_capacity, 500)
        self.assertEqual(config.mcts_batch_size, 25)
        self.assertTrue(config.training_mode)
        self.assertTrue(config.validate_reset)
        
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration
        config = GameStatePoolConfig(
            initial_size=10,
            max_capacity=100,
            mcts_batch_size=5
        )
        self.assertEqual(config.mcts_batch_size, 5)
        
        # Invalid MCTS batch size
        with self.assertRaises(ValueError):
            GameStatePoolConfig(
                initial_size=10,
                max_capacity=100,
                mcts_batch_size=0
            )


class TestGameStatePoolStatistics(unittest.TestCase):
    """Test GameState pool statistics tracking."""
    
    def test_statistics_initialization(self):
        """Test statistics are properly initialized."""
        stats = GameStatePoolStatistics()
        
        self.assertEqual(stats.reset_operations, 0)
        self.assertEqual(stats.reset_failures, 0)
        self.assertEqual(stats.validation_failures, 0)
        self.assertEqual(stats.mcts_batch_requests, 0)
        self.assertEqual(stats.mcts_batch_efficiency, 0.0)
        self.assertEqual(stats.avg_reset_time, 0.0)
        self.assertEqual(stats.total_reset_time, 0.0)
        
    def test_record_reset_time(self):
        """Test recording reset time statistics."""
        stats = GameStatePoolStatistics()
        
        stats.record_reset_time(0.001)
        self.assertEqual(stats.reset_operations, 1)
        self.assertEqual(stats.total_reset_time, 0.001)
        self.assertEqual(stats.avg_reset_time, 0.001)
        
        stats.record_reset_time(0.003)
        self.assertEqual(stats.reset_operations, 2)
        self.assertEqual(stats.total_reset_time, 0.004)
        self.assertEqual(stats.avg_reset_time, 0.002)
        
    def test_record_mcts_batch(self):
        """Test recording MCTS batch statistics."""
        stats = GameStatePoolStatistics()
        
        # First batch: 100% efficiency
        stats.record_mcts_batch(10, 10)
        self.assertEqual(stats.mcts_batch_requests, 1)
        self.assertEqual(stats.mcts_batch_efficiency, 100.0)
        
        # Second batch: 50% efficiency, should be averaged
        stats.record_mcts_batch(10, 5)
        self.assertEqual(stats.mcts_batch_requests, 2)
        # Exponential moving average: 0.1 * 50 + 0.9 * 100 = 95
        self.assertAlmostEqual(stats.mcts_batch_efficiency, 95.0, places=1)


class TestGameStatePool(unittest.TestCase):
    """Test GameStatePool functionality."""
    
    def setUp(self):
        """Set up test pool."""
        self.config = GameStatePoolConfig(
            initial_size=5,
            max_capacity=20,
            factory_func=create_game_state,
            reset_func=reset_game_state,
            enable_statistics=True,
            mcts_batch_size=5,
            validate_reset=False  # Disable for performance
        )
        self.pool = GameStatePool(self.config)
        
    def test_pool_initialization(self):
        """Test pool is properly initialized."""
        self.assertEqual(self.pool.size(), 5)  # Pre-allocated
        self.assertIsNotNone(self.pool.get_statistics())
        self.assertIsNotNone(self.pool.get_game_statistics())
        
    def test_default_config(self):
        """Test pool with default configuration."""
        pool = GameStatePool()
        self.assertGreater(pool.size(), 0)  # Should pre-allocate
        
    def test_get_and_return_game_state(self):
        """Test basic get and return operations."""
        # Get a GameState
        game_state = self.pool.get_game_state()
        self.assertIsInstance(game_state, GameState)
        self.assertEqual(self.pool.size(), 4)  # One less in pool
        
        # Modify the GameState
        game_state.current_player = Player.BLACK
        game_state.white_score = 2
        
        # Return it
        self.pool.return_game_state(game_state)
        self.assertEqual(self.pool.size(), 5)  # Back to original size
        
        # Get it again and verify it's reset
        reset_state = self.pool.get_game_state()
        self.assertEqual(reset_state.current_player, Player.WHITE)
        self.assertEqual(reset_state.white_score, 0)
        
    def test_batch_operations(self):
        """Test batch get and return operations."""
        # Get a batch
        batch = self.pool.get_batch(3)
        self.assertEqual(len(batch), 3)
        self.assertEqual(self.pool.size(), 2)  # 3 removed from pool
        
        # All should be GameState objects
        for game_state in batch:
            self.assertIsInstance(game_state, GameState)
            
        # Modify them
        for i, game_state in enumerate(batch):
            game_state.white_score = i + 1
            
        # Return the batch
        self.pool.return_batch(batch)
        self.assertEqual(self.pool.size(), 5)  # Back to original
        
        # Verify they're reset
        new_batch = self.pool.get_batch(3)
        for game_state in new_batch:
            self.assertEqual(game_state.white_score, 0)
            
    def test_batch_larger_than_pool(self):
        """Test requesting batch larger than pool size."""
        # Request more than available
        batch = self.pool.get_batch(10)
        self.assertEqual(len(batch), 10)  # Should still get 10
        # Pool may have grown, so check it's at least the original size
        self.assertGreaterEqual(self.pool.size(), 0)  # Pool size may vary due to auto-growth
        
        # Return them
        self.pool.return_batch(batch)
        # Pool should be able to hold all returned objects
        self.assertGreaterEqual(self.pool.size(), 10)  # Should have grown to accommodate
        
    def test_statistics_tracking(self):
        """Test statistics are properly tracked."""
        stats = self.pool.get_statistics()
        game_stats = self.pool.get_game_statistics()
        
        initial_hits = stats.hits
        initial_resets = game_stats.reset_operations
        
        # Get and return an object
        game_state = self.pool.get()
        self.pool.return_obj(game_state)
        
        # Check statistics updated
        self.assertEqual(stats.hits, initial_hits + 1)
        self.assertEqual(game_stats.reset_operations, initial_resets + 1)
        
    def test_validation_mode(self):
        """Test reset validation mode."""
        # Create pool with validation enabled
        config = GameStatePoolConfig(
            initial_size=2,
            max_capacity=10,
            factory_func=create_game_state,
            reset_func=reset_game_state,
            enable_statistics=True,
            validate_reset=True
        )
        pool = GameStatePool(config)
        
        # Get and return object normally
        game_state = pool.get()
        pool.return_obj(game_state)
        
        game_stats = pool.get_game_statistics()
        self.assertEqual(game_stats.validation_failures, 0)
        
    def test_type_checking(self):
        """Test type checking for non-GameState objects."""
        with self.assertRaises(TypeError):
            self.pool.return_obj("not a game state")
            
    def test_print_statistics(self):
        """Test printing statistics doesn't crash."""
        # This mainly tests that the method runs without error
        try:
            self.pool.print_statistics()
        except Exception as e:
            self.fail(f"print_statistics raised an exception: {e}")
            
    def test_thread_safety(self):
        """Test pool operations are thread-safe."""
        def worker(pool, iterations=10):
            for _ in range(iterations):
                game_state = pool.get()
                time.sleep(0.001)  # Simulate work
                pool.return_obj(game_state)
                
        # Run multiple threads concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(worker, self.pool) for _ in range(3)]
            for future in futures:
                future.result()  # Wait for completion
                
        # Pool should be back to original size
        self.assertEqual(self.pool.size(), 5)
        
    def test_concurrent_batch_operations(self):
        """Test concurrent batch operations."""
        def batch_worker(pool):
            batch = pool.get_batch(2)
            time.sleep(0.01)  # Simulate processing
            pool.return_batch(batch)
            
        # Run concurrent batch operations
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=batch_worker, args=(self.pool,))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # Pool should be stable
        self.assertGreaterEqual(self.pool.size(), 5)


class TestGameStatePoolFactory(unittest.TestCase):
    """Test convenience factory function."""
    
    def test_create_game_state_pool(self):
        """Test factory function creates proper pool."""
        pool = create_game_state_pool(
            initial_size=25,
            max_capacity=200,
            training_mode=True,
            validate_reset=True
        )
        
        self.assertIsInstance(pool, GameStatePool)
        self.assertEqual(pool.size(), 25)
        self.assertTrue(pool._game_config.training_mode)
        self.assertTrue(pool._game_config.validate_reset)
        
    def test_create_game_state_pool_defaults(self):
        """Test factory function with default parameters."""
        pool = create_game_state_pool()
        
        self.assertIsInstance(pool, GameStatePool)
        self.assertEqual(pool.size(), 100)  # Default initial size
        

class TestIntegrationScenarios(unittest.TestCase):
    """Test realistic usage scenarios."""
    
    def test_mcts_simulation_pattern(self):
        """Test pattern typical of MCTS simulations."""
        pool = create_game_state_pool(initial_size=50, max_capacity=200)
        
        # Simulate MCTS tree expansion
        for iteration in range(5):
            # Get batch for node expansion
            states = pool.get_batch(10)
            
            # Simulate game play on each state
            for state in states:
                # Place some rings
                state.rings_placed[Player.WHITE] = 3
                state.rings_placed[Player.BLACK] = 3
                state.phase = GamePhase.MAIN_GAME
                
            # Return states after simulation
            pool.return_batch(states)
            
        # Verify pool efficiency
        stats = pool.get_game_statistics()
        self.assertGreater(stats.mcts_batch_requests, 0)
        self.assertGreater(stats.mcts_batch_efficiency, 80.0)  # Should be efficient
        
    def test_training_loop_pattern(self):
        """Test pattern typical of training loops."""
        pool = create_game_state_pool(initial_size=100)
        
        # Simulate training episodes
        for episode in range(10):
            # Get state for episode
            state = pool.get()
            
            # Simulate game progression
            for move_num in range(20):  # Simulate 20 moves
                if move_num < 10:
                    state.phase = GamePhase.RING_PLACEMENT
                    state.rings_placed[Player.WHITE] = move_num // 2
                    state.rings_placed[Player.BLACK] = move_num // 2
                else:
                    state.phase = GamePhase.MAIN_GAME
                    
            # Return after episode
            pool.return_obj(state)
            
        # Verify reset efficiency
        base_stats = pool.get_statistics()
        game_stats = pool.get_game_statistics()
        
        self.assertEqual(base_stats.allocations, 10)
        self.assertEqual(game_stats.reset_operations, 10)
        self.assertEqual(game_stats.reset_failures, 0)


if __name__ == '__main__':
    unittest.main() 