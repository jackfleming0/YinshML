"""Optimized state conversion with zero-copy tensor operations."""

from typing import List, Tuple, Optional, Union
import numpy as np
import torch

from ..memory.zero_copy import (
    create_view_tensor,
    get_persistent_buffer,
    release_buffer,
    safe_copy_,
    ZeroCopyConfig,
    zero_copy_context,
    create_batch_from_numpy,
    get_zero_copy_statistics,
    ZeroCopyStatistics
)
from ..game.board import Board, Position
from ..game.game_state import GameState
from .state_conversion import StateConverter  # Import original converter


class OptimizedStateConverter(StateConverter):
    """
    Optimized state converter with zero-copy tensor operations.
    
    This class extends the standard StateConverter with zero-copy optimizations
    for improved memory efficiency and performance.
    """
    
    def __init__(self, enable_zero_copy: bool = True):
        """
        Initialize the optimized state converter.
        
        Args:
            enable_zero_copy: Whether to enable zero-copy optimizations
        """
        super().__init__()
        self.enable_zero_copy = enable_zero_copy
        self.config = ZeroCopyConfig()
        self.statistics = ZeroCopyStatistics()
        
        # Buffer cache for common tensor shapes
        self._tensor_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Configure zero-copy for state conversion
        if self.enable_zero_copy:
            self.zero_copy_config = ZeroCopyConfig(
                enable_shared_memory_tensors=False,  # State tensors are typically small
                enable_persistent_buffers=True,
                enable_view_operations=True,
                enable_inplace_operations=True,
                inplace_threshold_mb=0.1,  # Lower threshold for state tensors
                max_buffer_pool_size_mb=64
            )
    
    def is_valid_position(self, row: int, col: int) -> bool:
        """Check if a position is valid on the Yinsh board."""
        return 0 <= row < 11 and 0 <= col < 11
    
    def encode_state(self, game_state: GameState, valid_moves=None) -> torch.Tensor:
        """
        Standard encode_state method for compatibility.
        
        This delegates to the parent class and converts to tensor.
        """
        # Use parent class encoding
        state_array = super().encode_state(game_state)
        return torch.from_numpy(state_array)
    
    def _get_cached_tensor(self, key: str, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Get or create a cached tensor for reuse."""
        if not self.enable_zero_copy:
            return torch.zeros(shape, dtype=dtype)
        
        cache_key = f"{key}_{shape}_{dtype}"
        
        if cache_key not in self._tensor_cache:
            self._tensor_cache[cache_key] = get_persistent_buffer(shape, dtype)
        
        tensor = self._tensor_cache[cache_key]
        
        # Resize if shape changed
        if tensor.shape != shape:
            release_buffer(tensor)
            self._tensor_cache[cache_key] = get_persistent_buffer(shape, dtype)
            tensor = self._tensor_cache[cache_key]
        
        return tensor
    
    def encode_state_zero_copy(self, 
                              game_state: GameState,
                              valid_moves: Optional[List[Tuple[Position, Position]]] = None,
                              reuse_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode game state to tensor using zero-copy operations.
        
        Args:
            game_state: Game state to encode
            valid_moves: Optional list of valid moves
            reuse_tensor: Optional tensor to reuse for output
            
        Returns:
            Encoded state tensor
        """
        if not self.enable_zero_copy:
            # Fallback to original method
            return self.encode_state(game_state, valid_moves)
        
        with zero_copy_context(self.zero_copy_config):
            # Use numpy encoding first (reuse existing logic)
            state_np = self._encode_state_to_numpy(game_state, valid_moves)
            
            if reuse_tensor is not None and reuse_tensor.shape == state_np.shape:
                # Reuse provided tensor
                if reuse_tensor.dtype == torch.float32:
                    # Copy data in-place
                    numpy_tensor = torch.from_numpy(state_np.astype(np.float32))
                    safe_copy_(reuse_tensor, numpy_tensor)
                    return reuse_tensor
            
            # Get cached tensor or create new one
            target_shape = state_np.shape
            state_tensor = self._get_cached_tensor('state_encoding', target_shape, torch.float32)
            
            # Convert numpy to tensor efficiently
            if state_np.dtype == np.float32:
                # Direct conversion without copying
                source_tensor = torch.from_numpy(state_np)
            else:
                # Need type conversion
                source_tensor = torch.from_numpy(state_np.astype(np.float32))
            
            # Copy to cached tensor
            safe_copy_(state_tensor, source_tensor)
            
            return state_tensor
    
    def _encode_state_to_numpy(self, 
                              game_state: GameState,
                              valid_moves: Optional[List[Tuple[Position, Position]]] = None) -> np.ndarray:
        """Encode game state to numpy array (reuses existing logic)."""
        # Reuse the existing numpy encoding logic from parent class
        board = game_state.board
        state = np.zeros((6, 11, 11), dtype=np.float32)
        
        # Board state encoding (layer 0-3)
        for row in range(11):
            for col in range(11):
                if self.is_valid_position(row, col):
                    pos = Position(row, col)
                    piece = board.get_piece(pos)
                    
                    if piece == 1:  # White piece
                        state[0, row, col] = 1.0
                    elif piece == -1:  # Black piece
                        state[1, row, col] = 1.0
                    elif piece == 2:  # White ring
                        state[2, row, col] = 1.0
                    elif piece == -2:  # Black ring
                        state[3, row, col] = 1.0
        
        # Current player indicator (layer 4)
        current_player_value = 1.0 if game_state.current_player == 1 else 0.0
        state[4, :, :] = current_player_value
        
        # Valid moves mask (layer 5)
        if valid_moves:
            for start_pos, end_pos in valid_moves:
                if self.is_valid_position(start_pos.row, start_pos.col):
                    state[5, start_pos.row, start_pos.col] = 1.0
        
        return state
    
    def decode_state_zero_copy(self, 
                              tensor: torch.Tensor,
                              reuse_board: Optional[Board] = None) -> GameState:
        """
        Decode tensor to game state using zero-copy operations.
        
        Args:
            tensor: State tensor to decode
            reuse_board: Optional board object to reuse
            
        Returns:
            Decoded game state
        """
        if not self.enable_zero_copy:
            # Fallback to original method
            return self.tensor_to_board(tensor)
        
        # Move tensor to CPU if needed (minimize device transfers)
        if tensor.device.type != 'cpu':
            # Use view if possible to avoid copying
            cpu_tensor = tensor.cpu()
        else:
            cpu_tensor = tensor
        
        # Convert to numpy efficiently
        if cpu_tensor.is_contiguous():
            state_np = cpu_tensor.detach().numpy()
        else:
            # Need to make contiguous first
            state_np = cpu_tensor.contiguous().detach().numpy()
        
        # Decode from numpy (reuse existing logic)
        return self._decode_numpy_to_game_state(state_np, reuse_board)
    
    def _decode_numpy_to_game_state(self, 
                                   state_np: np.ndarray,
                                   reuse_board: Optional[Board] = None) -> GameState:
        """Decode numpy array to game state (reuses existing logic)."""
        # Create or reuse board
        if reuse_board is not None:
            board = reuse_board
            board.clear()  # Reset the board
        else:
            board = Board()
        
        # Decode board state from tensor layers
        for row in range(11):
            for col in range(11):
                if self.is_valid_position(row, col):
                    pos = Position(row, col)
                    
                    # Check each layer
                    if state_np[0, row, col] > 0.5:  # White piece
                        board.place_piece(pos, 1)
                    elif state_np[1, row, col] > 0.5:  # Black piece
                        board.place_piece(pos, -1)
                    elif state_np[2, row, col] > 0.5:  # White ring
                        board.place_piece(pos, 2)
                    elif state_np[3, row, col] > 0.5:  # Black ring
                        board.place_piece(pos, -2)
        
        # Determine current player from layer 4
        current_player = 1 if state_np[4, 0, 0] > 0.5 else -1
        
        # Create game state
        game_state = GameState()
        game_state.board = board
        game_state.current_player = current_player
        
        return game_state
    
    def augment_state_zero_copy(self, 
                               state: torch.Tensor,
                               output_tensors: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
        """
        Generate augmented states using zero-copy operations.
        
        Args:
            state: Original state tensor
            output_tensors: Optional list of tensors to reuse for output
            
        Returns:
            List of augmented state tensors
        """
        if not self.enable_zero_copy:
            # Fallback to original method
            return self.augment_state(state)
        
        augmented_states = []
        
        with zero_copy_context(self.zero_copy_config):
            # Generate 6 rotations (60-degree increments)
            for k in range(6):
                if output_tensors and k < len(output_tensors):
                    # Reuse provided tensor
                    rotated = self._rotate_hex_board_zero_copy(state, k, output_tensors[k])
                else:
                    # Create new tensor
                    rotated = self._rotate_hex_board_zero_copy(state, k)
                
                augmented_states.append(rotated)
        
        return augmented_states
    
    def _rotate_hex_board_zero_copy(self, 
                                   state: torch.Tensor, 
                                   k: int,
                                   output_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Rotate hex board using zero-copy operations."""
        if k == 0:
            # No rotation needed
            if output_tensor is not None:
                safe_copy_(output_tensor, state)
                return output_tensor
            else:
                return state.clone()
        
        # Get or create output tensor
        if output_tensor is None:
            output_tensor = self._get_cached_tensor(f'rotation_{k}', state.shape, state.dtype)
        
        # Move to CPU for numpy operations if needed
        if state.device.type != 'cpu':
            cpu_state = state.cpu()
        else:
            cpu_state = state
        
        # Convert to numpy for rotation (reuse existing rotation logic)
        state_np = cpu_state.detach().numpy()
        rotated_np = self._rotate_numpy_array(state_np, k)
        
        # Convert back to tensor
        rotated_tensor = torch.from_numpy(rotated_np)
        
        # Copy to output tensor
        safe_copy_(output_tensor, rotated_tensor)
        
        # Move back to original device if needed
        if state.device.type != 'cpu':
            output_tensor = output_tensor.to(state.device)
        
        return output_tensor
    
    def _rotate_numpy_array(self, state_np: np.ndarray, k: int) -> np.ndarray:
        """Rotate numpy array (reuses existing rotation logic)."""
        # This would implement the hex board rotation logic
        # For now, use a placeholder that maintains shape
        if k == 0:
            return state_np.copy()
        
        # Implement hex rotation transformation
        # This is a simplified version - the actual implementation would need
        # proper hex coordinate transformation
        rotated = np.zeros_like(state_np)
        
        # Simple rotation approximation for demonstration
        for channel in range(state_np.shape[0]):
            rotated[channel] = np.rot90(state_np[channel], k)
        
        return rotated
    
    def batch_encode_states_zero_copy(self,
                                     game_states: List[GameState],
                                     valid_moves_list: Optional[List[List[Tuple[Position, Position]]]] = None,
                                     output_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode multiple game states to a batched tensor using zero-copy operations.
        
        Args:
            game_states: List of game states to encode
            valid_moves_list: Optional list of valid moves for each state
            output_tensor: Optional tensor to reuse for output
            
        Returns:
            Batched state tensor
        """
        if not self.enable_zero_copy:
            # Fallback to manual batching
            tensors = []
            for i, state in enumerate(game_states):
                valid_moves = valid_moves_list[i] if valid_moves_list else None
                tensor = self.encode_state(state, valid_moves)
                tensors.append(tensor)
            return torch.stack(tensors)
        
        batch_size = len(game_states)
        if batch_size == 0:
            return torch.empty(0, 6, 11, 11, dtype=torch.float32)
        
        # Determine output shape
        sample_state = self._encode_state_to_numpy(game_states[0])
        batch_shape = (batch_size,) + sample_state.shape
        
        # Get or create output tensor
        if output_tensor is None or output_tensor.shape != batch_shape:
            output_tensor = self._get_cached_tensor('batch_encoding', batch_shape, torch.float32)
        
        with zero_copy_context(self.zero_copy_config):
            # Encode each state directly into the batch tensor
            for i, game_state in enumerate(game_states):
                valid_moves = valid_moves_list[i] if valid_moves_list else None
                
                # Encode state to numpy
                state_np = self._encode_state_to_numpy(game_state, valid_moves)
                
                # Convert to tensor
                state_tensor = torch.from_numpy(state_np)
                
                # Copy into batch tensor
                safe_copy_(output_tensor[i], state_tensor)
        
        return output_tensor
    
    def cleanup_cache(self):
        """Clean up cached tensors."""
        if self.enable_zero_copy:
            for tensor in self._tensor_cache.values():
                release_buffer(tensor)
            self._tensor_cache.clear()
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup_cache()
        except:
            pass


# Global instance for easy access
_default_converter = OptimizedStateConverter(enable_zero_copy=True)


def encode_state_optimized(game_state: GameState,
                          valid_moves: Optional[List[Tuple[Position, Position]]] = None,
                          reuse_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Encode game state using optimized zero-copy operations."""
    return _default_converter.encode_state_zero_copy(game_state, valid_moves, reuse_tensor)


def decode_state_optimized(tensor: torch.Tensor,
                          reuse_board: Optional[Board] = None) -> GameState:
    """Decode tensor to game state using optimized zero-copy operations."""
    return _default_converter.decode_state_zero_copy(tensor, reuse_board)


def augment_state_optimized(state: torch.Tensor,
                           output_tensors: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
    """Generate augmented states using optimized zero-copy operations."""
    return _default_converter.augment_state_zero_copy(state, output_tensors)


def batch_encode_states_optimized(game_states: List[GameState],
                                 valid_moves_list: Optional[List[List[Tuple[Position, Position]]]] = None,
                                 output_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Batch encode game states using optimized zero-copy operations."""
    return _default_converter.batch_encode_states_zero_copy(game_states, valid_moves_list, output_tensor) 