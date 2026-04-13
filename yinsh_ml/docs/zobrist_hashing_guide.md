# Zobrist Hashing Guide

## Overview

Zobrist hashing is a technique for efficiently computing hash values of board game positions. This implementation provides fast, deterministic hashing for Yinsh game states, designed to support transposition tables in search algorithms like minimax, MCTS, or AlphaZero.

## Key Concepts

### What is Zobrist Hashing?

Zobrist hashing works by:
1. Precomputing random 64-bit values for each (position, piece_type) combination
2. Computing board hashes by XORing together the values for all pieces on the board
3. Updating hashes incrementally by XORing in/out values as pieces move

### Why Use Zobrist Hashing?

- **Fast incremental updates**: O(1) per position change vs O(n) for full recomputation
- **Deterministic**: Same board state always produces the same hash
- **Low collision rate**: Cryptographic randomness minimizes hash collisions
- **Efficient**: Perfect for transposition table lookups in game tree search

## Quick Start

### Basic Usage

```python
from yinsh_ml.game.zobrist import ZobristHasher
from yinsh_ml.game.board import Board
from yinsh_ml.game.constants import Position, PieceType, Player

# Create a hasher
hasher = ZobristHasher(seed="my-game-seed")

# Hash a board
board = Board()
board.place_piece(Position.from_string("E5"), PieceType.WHITE_RING)
hash_value = hasher.hash_board(board)
```

### Incremental Updates

```python
# Instead of recomputing the full hash, update incrementally
new_hash = hasher.place_marker(
    Position.from_string("F6"), 
    Player.WHITE, 
    hash_value
)

# Verify it matches full recomputation
board.place_piece(Position.from_string("F6"), PieceType.WHITE_MARKER)
assert new_hash == hasher.hash_board(board)
```

## API Reference

### ZobristHasher

The main class for computing and updating hashes.

#### Initialization

```python
# Random initialization (different each run)
hasher = ZobristHasher()

# Deterministic initialization (same hash table each run)
hasher = ZobristHasher(seed="my-seed")

# Using a precomputed table (for sharing across instances)
from yinsh_ml.game.zobrist import ZobristInitializer
initializer = ZobristInitializer(seed="shared")
hasher1 = ZobristHasher(table=initializer.table)
hasher2 = ZobristHasher(table=initializer.table)  # Same table
```

#### Methods

##### `hash_board(board: Board) -> int`
Compute the hash of a complete board state.

```python
board = Board()
board.place_piece(Position.from_string("E5"), PieceType.WHITE_RING)
hash_value = hasher.hash_board(board)
```

##### `hash_state(game_state: GameState) -> int`
Compute the hash of a complete game state (currently just the board).

```python
from yinsh_ml.game.game_state import GameState
state = GameState()
state.board.place_piece(Position.from_string("E5"), PieceType.WHITE_RING)
hash_value = hasher.hash_state(state)
```

##### `place_ring(position: Position, player: Player, current_hash: int) -> int`
Incrementally update hash when placing a ring.

```python
hash_after = hasher.place_ring(
    Position.from_string("E5"),
    Player.WHITE,
    current_hash
)
```

##### `place_marker(position: Position, player: Player, current_hash: int) -> int`
Incrementally update hash when placing a marker.

```python
hash_after = hasher.place_marker(
    Position.from_string("F6"),
    Player.BLACK,
    current_hash
)
```

##### `remove_piece(position: Position, piece_type: PieceType, current_hash: int) -> int`
Incrementally update hash when removing a piece.

```python
hash_after = hasher.remove_piece(
    Position.from_string("E5"),
    PieceType.WHITE_RING,
    current_hash
)
```

##### `move_ring(source: Position, destination: Position, player: Player, current_hash: int, board: Board = None) -> int`
Incrementally update hash for a ring move (handles marker flips along path).

```python
hash_after = hasher.move_ring(
    Position.from_string("E5"),
    Position.from_string("E8"),
    Player.WHITE,
    current_hash,
    board  # Optional: needed to detect marker flips
)
```

##### `batch_update(updates: List[Tuple[Position, PieceType, PieceType]], current_hash: int) -> int`
Apply multiple position changes atomically.

```python
updates = [
    (Position.from_string("E5"), PieceType.EMPTY, PieceType.WHITE_RING),
    (Position.from_string("F6"), PieceType.EMPTY, PieceType.BLACK_MARKER),
]
hash_after = hasher.batch_update(updates, current_hash)
```

## Integration with Transposition Tables

### Basic Transposition Table

```python
from typing import Dict, Optional

class TranspositionTable:
    def __init__(self):
        self.hasher = ZobristHasher(seed="transposition-table")
        self.table: Dict[int, dict] = {}
    
    def store(self, board: Board, value: float, depth: int, move_type: str):
        """Store a position in the transposition table."""
        hash_key = self.hasher.hash_board(board)
        self.table[hash_key] = {
            'value': value,
            'depth': depth,
            'move_type': move_type,
        }
    
    def lookup(self, board: Board) -> Optional[dict]:
        """Look up a position in the transposition table."""
        hash_key = self.hasher.hash_board(board)
        return self.table.get(hash_key)
```

### Incremental Hash Tracking

For best performance in search algorithms, track the hash incrementally:

```python
class SearchNode:
    def __init__(self, board: Board, hasher: ZobristHasher):
        self.board = board
        self.hasher = hasher
        self.hash_value = hasher.hash_board(board)
    
    def make_move(self, move: Move) -> 'SearchNode':
        """Create a new node with updated hash."""
        new_board = self.board.copy()
        new_board.make_move(move)
        
        # Update hash incrementally
        if move.type == MoveType.PLACE_RING:
            new_hash = self.hasher.place_ring(
                move.source, move.player, self.hash_value
            )
        elif move.type == MoveType.MOVE_RING:
            new_hash = self.hasher.move_ring(
                move.source, move.destination, move.player,
                self.hash_value, self.board
            )
        # ... handle other move types
        
        new_node = SearchNode(new_board, self.hasher)
        new_node.hash_value = new_hash
        return new_node
```

## Performance Characteristics

### Hash Computation Speed

- **Full board hashing**: ~1-5 microseconds per board (depends on number of pieces)
- **Incremental update**: ~0.1-0.5 microseconds per position change
- **Speedup**: Incremental updates are typically 2-10x faster than full recomputation

### Memory Usage

- **Zobrist table**: ~425 KB (85 positions × 5 piece types × 8 bytes)
- **Per hash value**: 8 bytes (64-bit integer)
- **Overhead**: Minimal - just the hash table and hash values

### Collision Rate

- **Expected collisions**: < 0.1% for 100,000 random board states
- **Hash space**: 2^64 possible values
- **Quality**: Passes statistical tests for uniformity and avalanche effect

## Best Practices

### 1. Use Deterministic Seeds for Reproducibility

```python
# Good: Deterministic for testing/reproducibility
hasher = ZobristHasher(seed="production-seed")

# Also good: Random for production (different each run)
hasher = ZobristHasher()  # No seed = random
```

### 2. Prefer Incremental Updates

```python
# Good: Incremental update (fast)
new_hash = hasher.place_ring(position, player, current_hash)

# Avoid: Full recomputation (slower)
board.place_piece(position, ring_type)
new_hash = hasher.hash_board(board)  # Slower!
```

### 3. Share ZobristTable Across Instances

```python
# Good: Share table to save memory
initializer = ZobristInitializer(seed="shared")
hasher1 = ZobristHasher(table=initializer.table)
hasher2 = ZobristHasher(table=initializer.table)

# Avoid: Creating multiple tables unnecessarily
hasher1 = ZobristHasher(seed="shared")  # Creates new table
hasher2 = ZobristHasher(seed="shared")  # Creates duplicate table
```

### 4. Handle move_ring with Board Context

```python
# Good: Provide board for accurate marker flip detection
new_hash = hasher.move_ring(
    source, destination, player, current_hash, board
)

# Works but less accurate: No board (assumes no markers in path)
new_hash = hasher.move_ring(
    source, destination, player, current_hash
)
```

## Troubleshooting

### Issue: Different Hashes for Same Board

**Symptom**: Same board state produces different hash values.

**Causes**:
- Using different seeds for ZobristHasher instances
- Board state actually differs (check piece positions carefully)

**Solution**:
```python
# Ensure consistent seed
hasher1 = ZobristHasher(seed="consistent")
hasher2 = ZobristHasher(seed="consistent")
assert hasher1.hash_board(board) == hasher2.hash_board(board)
```

### Issue: Incremental Update Doesn't Match Full Recompute

**Symptom**: Incremental update produces different hash than full recomputation.

**Causes**:
- Incorrect piece type in incremental update
- Board state changed between incremental update and recomputation
- Missing intermediate steps in complex moves

**Solution**:
```python
# Always verify incremental matches full recomputation
incremental_hash = hasher.place_ring(position, player, current_hash)
board.place_piece(position, ring_type)
full_hash = hasher.hash_board(board)
assert incremental_hash == full_hash, "Mismatch detected!"
```

### Issue: High Collision Rate

**Symptom**: Many hash collisions in transposition table.

**Causes**:
- Extremely large transposition table (> 1M entries)
- Bug in hash computation
- Using same seed across different game instances incorrectly

**Solution**:
- Verify hash quality with validation tests
- Use different seeds for different game instances if needed
- Consider using 128-bit hashes for very large tables (future enhancement)

## Advanced Usage

### Custom Piece Types

```python
from yinsh_ml.game.zobrist import ZobristInitializer, DEFAULT_PIECE_ORDER

# Use default piece types
initializer = ZobristInitializer(seed="test")

# Or specify custom piece types
custom_pieces = (
    PieceType.EMPTY,
    PieceType.WHITE_RING,
    PieceType.BLACK_RING,
)
initializer = ZobristInitializer(
    seed="test",
    piece_types=custom_pieces
)
```

### Performance Benchmarking

```python
import time

hasher = ZobristHasher(seed="benchmark")
board = generate_test_board()

# Benchmark full hashing
start = time.perf_counter()
for _ in range(1000):
    hasher.hash_board(board)
full_time = time.perf_counter() - start

# Benchmark incremental
hash_val = hasher.hash_board(board)
start = time.perf_counter()
for _ in range(1000):
    hasher.place_marker(Position.from_string("E5"), Player.WHITE, hash_val)
incr_time = time.perf_counter() - start

print(f"Full: {full_time:.4f}s, Incremental: {incr_time:.4f}s")
print(f"Speedup: {full_time / incr_time:.2f}x")
```

## Testing

The implementation includes comprehensive tests:

- **Unit tests**: `test_zobrist_initializer.py`, `test_zobrist_hasher.py`
- **Incremental tests**: `test_zobrist_incremental.py`
- **Validation tests**: `test_zobrist_validation.py`

Run all tests:
```bash
pytest yinsh_ml/tests/test_zobrist*.py -v
```

Run fast tests only (excludes slow statistical tests):
```bash
pytest yinsh_ml/tests/test_zobrist*.py -v -m "not slow"
```

## References

- Zobrist, A. L. (1970). "A New Hashing Method with Application for Game Playing"
- Wikipedia: [Zobrist Hashing](https://en.wikipedia.org/wiki/Zobrist_hashing)
- Chess Programming Wiki: [Zobrist Hashing](https://www.chessprogramming.org/Zobrist_Hashing)

## See Also

- `yinsh_ml.game.board.Board`: Board representation
- `yinsh_ml.game.game_state.GameState`: Complete game state
- `yinsh_ml.game.constants`: Game constants and types

