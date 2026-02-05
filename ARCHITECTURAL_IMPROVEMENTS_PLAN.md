# Architectural Improvements Plan

**Date:** February 4, 2026
**Branch:** `architectural-improvements`
**Status:** Planning Phase
**Authors:** Claude + Peer Reviewers

---

## Executive Summary

Based on peer feedback and analysis of our hyperparameter tuning results, we've identified that **architectural changes will yield larger gains than continued hyperparameter optimization**. Our current system plateaus around ELO 1520-1580, which represents basic tactical understanding but not strategic play.

This document outlines a comprehensive plan to implement architectural improvements that could enable "phase transition" jumps in performance (1550 → 1700+).

### Key Insight

> "If you improve encoding, constrain actions, guide self-play, and trade depth for throughput—you don't get incremental gains, you get phase transitions." — Peer Reviewer 1

---

## Current System Analysis

### What We Have

| Component | Current Implementation | Location |
|-----------|----------------------|----------|
| State Encoding | 6 channels (rings, markers, valid moves, phase) | `yinsh_ml/utils/encoding.py` |
| Data Augmentation | None | - |
| Action Masking | Hard legal move mask only | `yinsh_ml/game/moves.py` |
| Self-Play | Pure self-play, single opponent | `yinsh_ml/training/self_play.py` |
| MCTS Budget | Configurable but static per run | `yinsh_ml/search/mcts.py` |
| Metrics | ELO, loss curves, basic stats | `yinsh_ml/experiments/` |

### Current Performance Ceiling

From our experiments (24+ runs):
- **Best ELO achieved:** 1575 (temp_schedule_003)
- **Typical range:** 1500-1540
- **Held peak rate:** ~17% (2/12 in replication study)
- **Training stability:** Volatile, frequent regressions

### Why We're Plateauing

1. **State encoding hides structure** - Network must rediscover adjacency, threats, connectivity
2. **No data augmentation** - Missing 12x data multiplier from Yinsh's symmetries
3. **No action guidance** - Policy spreads probability across dominated moves
4. **Pure self-play** - Both players equally weak early, no learning signal
5. **Limited instrumentation** - Can't diagnose what's failing

---

## Proposed Improvements

### Phase 1: Enhanced State Encoding

**Priority:** HIGHEST
**Expected Impact:** +100 ELO
**Effort:** Medium (2-3 days)

#### Current Encoding (6 channels)

```python
# From yinsh_ml/utils/encoding.py
Channel 0: White rings (binary)
Channel 1: Black rings (binary)
Channel 2: White markers (binary)
Channel 3: Black markers (binary)
Channel 4: Valid moves (binary)
Channel 5: Game phase (scalar)
```

#### Proposed Encoding (14+ channels)

```python
# Core piece planes (4 channels) - KEEP
Channel 0: Current player rings
Channel 1: Opponent rings
Channel 2: Current player markers
Channel 3: Opponent markers

# Threat/Tactical planes (4 channels) - NEW
Channel 4: Row threats (cells that complete a row of 5 for current player)
Channel 5: Opponent row threats (defensive awareness)
Channel 6: Partial rows (3-4 markers in a line, current player)
Channel 7: Opponent partial rows

# Positional planes (3 channels) - NEW
Channel 8: Ring mobility (number of valid moves per ring, normalized)
Channel 9: Center distance (proximity to board center)
Channel 10: Ring influence (cells reachable by current player's rings)

# Game state planes (3 channels) - ENHANCED
Channel 11: Valid move destinations (current)
Channel 12: Game phase (one-hot or scalar)
Channel 13: Turn number (normalized)
Channel 14: Score differential (rings removed)
```

#### Implementation Details

**File:** `yinsh_ml/utils/encoding.py`

```python
class EnhancedStateEncoder:
    """
    Enhanced state encoder exposing tactical and positional features.

    The network's ceiling is determined by what information we expose,
    not how deep the network is.
    """

    NUM_CHANNELS = 15  # Up from 6

    def encode(self, game_state: GameState) -> np.ndarray:
        planes = []

        # Core piece planes (relative to current player)
        planes.append(self._encode_pieces(game_state, current_player=True, piece_type='ring'))
        planes.append(self._encode_pieces(game_state, current_player=False, piece_type='ring'))
        planes.append(self._encode_pieces(game_state, current_player=True, piece_type='marker'))
        planes.append(self._encode_pieces(game_state, current_player=False, piece_type='marker'))

        # Threat planes
        planes.append(self._encode_row_threats(game_state, current_player=True))
        planes.append(self._encode_row_threats(game_state, current_player=False))
        planes.append(self._encode_partial_rows(game_state, current_player=True))
        planes.append(self._encode_partial_rows(game_state, current_player=False))

        # Positional planes
        planes.append(self._encode_ring_mobility(game_state))
        planes.append(self._encode_center_distance())
        planes.append(self._encode_ring_influence(game_state))

        # Game state planes
        planes.append(self._encode_valid_moves(game_state))
        planes.append(self._encode_phase(game_state))
        planes.append(self._encode_turn_number(game_state))
        planes.append(self._encode_score_differential(game_state))

        return np.stack(planes, axis=0)

    def _encode_row_threats(self, game_state, current_player: bool) -> np.ndarray:
        """
        Encode cells that would complete a row of 5 markers.
        This is the most critical tactical information in Yinsh.
        """
        # Find all lines of 4 markers with an empty cell that could complete
        pass

    def _encode_partial_rows(self, game_state, current_player: bool) -> np.ndarray:
        """
        Encode cells participating in partial rows (3-4 markers).
        Helps network learn to build toward rows.
        """
        pass

    def _encode_ring_mobility(self, game_state) -> np.ndarray:
        """
        For each cell with a ring, encode how many moves that ring has.
        High mobility rings are more valuable.
        """
        pass

    def _encode_ring_influence(self, game_state) -> np.ndarray:
        """
        Encode cells reachable by current player's rings.
        Shows board control.
        """
        pass
```

#### Migration Strategy

1. Create `EnhancedStateEncoder` alongside existing `StateEncoder`
2. Add config flag: `use_enhanced_encoding: bool = True`
3. Update `NetworkWrapper` input channels: 6 → 15
4. Retrain from scratch (encoding change invalidates old weights)

---

### Phase 2: Symmetry-Aware Data Augmentation

**Priority:** HIGH
**Expected Impact:** 12x effective training data
**Effort:** Medium (2 days)

#### Yinsh Symmetry Group

Yinsh's hexagonal board has **dihedral group D6** symmetry:
- 6 rotations (0°, 60°, 120°, 180°, 240°, 300°)
- 6 reflections (across 6 axes)
- Total: **12 equivalent positions** per game state

#### Implementation

**File:** `yinsh_ml/training/augmentation.py` (NEW)

```python
class YinshSymmetryAugmenter:
    """
    Augment training data using Yinsh's D6 symmetry group.

    IMPORTANT: Not all symmetries are valid in all game phases.
    - Ring placement: Full D6 symmetry
    - Main game: Full D6 symmetry
    - First move: May break symmetry (first-player advantage)

    We use SELECTIVE augmentation - only transforms the game truly respects.
    """

    # Hexagonal coordinate transformations
    ROTATIONS = [0, 60, 120, 180, 240, 300]  # degrees
    REFLECTIONS = [0, 1, 2, 3, 4, 5]  # axes

    def __init__(self, include_reflections: bool = True):
        self.transforms = self._build_transform_group(include_reflections)

    def augment(self, state: np.ndarray, policy: np.ndarray, value: float) -> List[Tuple]:
        """
        Generate all valid symmetric versions of a training sample.

        Args:
            state: Encoded game state (C, H, W)
            policy: Policy distribution (action_space_size,)
            value: Value target (scalar)

        Returns:
            List of (state, policy, value) tuples
        """
        augmented = [(state, policy, value)]  # Original

        for transform in self.transforms[1:]:  # Skip identity
            aug_state = self._apply_state_transform(state, transform)
            aug_policy = self._apply_policy_transform(policy, transform)
            augmented.append((aug_state, aug_policy, value))

        return augmented

    def _apply_state_transform(self, state: np.ndarray, transform) -> np.ndarray:
        """Apply geometric transform to state encoding."""
        # Rotate/reflect each channel
        pass

    def _apply_policy_transform(self, policy: np.ndarray, transform) -> np.ndarray:
        """
        Apply transform to policy distribution.

        Policy is over (from_pos, to_pos) pairs, so we must transform
        both the source and destination coordinates.
        """
        pass
```

#### Coordinate System

Yinsh uses an 11x11 grid with hexagonal topology. Key considerations:

```python
# Hexagonal rotation (60 degrees clockwise)
def rotate_hex_60(x, y, center=(5, 5)):
    # Convert to cube coordinates, rotate, convert back
    pass

# Reflection across axis
def reflect_hex(x, y, axis, center=(5, 5)):
    pass
```

#### Integration

```python
# In training loop (self_play.py or trainer.py)
for game in completed_games:
    for state, policy, value in game.trajectory:
        # Original sample
        buffer.add(state, policy, value)

        # Augmented samples (if enabled)
        if config.use_augmentation:
            for aug_state, aug_policy, aug_value in augmenter.augment(state, policy, value):
                buffer.add(aug_state, aug_policy, aug_value)
```

---

### Phase 3: Enhanced Instrumentation

**Priority:** HIGH
**Expected Impact:** Debugging capability
**Effort:** Low-Medium (1-2 days)

#### New Metrics to Track

**File:** `yinsh_ml/utils/training_metrics.py` (NEW)

```python
@dataclass
class EnhancedTrainingMetrics:
    """Comprehensive metrics for diagnosing training issues."""

    # Core metrics (existing)
    iteration: int
    elo: float
    policy_loss: float
    value_loss: float

    # MCTS Agreement (NEW)
    mcts_agreement_rate: float  # % time policy top move == MCTS choice

    # Policy Health (NEW)
    policy_entropy: float  # Should decrease over training
    policy_top1_confidence: float  # Confidence in best move
    policy_top5_coverage: float  # Probability mass in top 5 moves

    # Value Calibration (NEW)
    value_prediction_accuracy: float  # On held-out positions
    value_confidence: float  # Average |v| (should increase)

    # Game Statistics (NEW)
    avg_game_length: float
    game_length_std: float
    win_margin_distribution: Dict[str, int]  # {"3-0": n, "3-1": n, "3-2": n}

    # Training Dynamics (NEW)
    checkpoint_vs_previous_winrate: float  # iter-N vs iter-N-10
    opening_diversity: float  # Entropy of first 5 moves

    # Tactical Benchmarks (NEW)
    tactical_solve_rate: float  # % of test positions solved correctly
```

#### MCTS Agreement Rate

```python
def compute_mcts_agreement(policy_output, mcts_policy):
    """
    Measure how often the raw policy network agrees with MCTS.

    Early training: Should be low (20-30%)
    Late training: Should be high (50-70%)

    If this stays below 30% forever, network isn't learning from search.
    """
    policy_top = np.argmax(policy_output)
    mcts_top = np.argmax(mcts_policy)
    return policy_top == mcts_top
```

#### Tactical Benchmark Suite

**File:** `yinsh_ml/evaluation/tactical_benchmarks.py` (NEW)

```python
class TacticalBenchmarks:
    """
    Fixed test positions with known correct answers.

    Categories:
    1. Complete row in 1 move (win)
    2. Block opponent's row threat (defense)
    3. Dominant ring placement (positional)
    4. Ring removal choice (endgame)
    """

    BENCHMARKS = [
        {
            "name": "complete_row_1",
            "description": "White can complete a row and win",
            "fen": "...",  # Position encoding
            "correct_moves": ["e5-e9"],  # Winning move(s)
            "category": "tactics"
        },
        # ... more positions
    ]

    def evaluate(self, network, mcts=None) -> Dict:
        """
        Test network (with optional MCTS) on benchmark positions.

        Returns solve rate per category.
        """
        pass
```

#### Value Calibration Test

```python
def evaluate_value_calibration(network, test_positions):
    """
    Test value head accuracy on positions with known outcomes.

    Save positions from iteration 10, then every 10 iterations:
    - Run inference on these positions
    - Compare predictions to actual game outcomes
    - Track calibration improvement over time
    """
    predictions = []
    actuals = []

    for pos in test_positions:
        pred = network.predict_value(pos.state)
        actual = pos.outcome  # 1.0, 0.0, or -1.0
        predictions.append(pred)
        actuals.append(actual)

    # Compute calibration metrics
    mse = mean_squared_error(actuals, predictions)
    correlation = np.corrcoef(actuals, predictions)[0, 1]

    return {"mse": mse, "correlation": correlation}
```

---

### Phase 4: Action Space Soft Masking

**Priority:** MEDIUM-HIGH
**Expected Impact:** 2x learning speed
**Effort:** Low (1 day)

#### Concept

Don't delete dominated moves—discourage them with soft penalties.

```python
# Hard mask (current)
policy = softmax(logits) * legal_mask  # Illegal moves = 0

# Soft mask (proposed)
soft_mask = compute_soft_mask(state)  # 1.0 for good, 0.1 for dominated
policy = softmax(logits + log(soft_mask))  # Dominated moves suppressed
```

#### Dominated Move Categories for Yinsh

```python
def compute_soft_mask(game_state) -> np.ndarray:
    """
    Compute soft penalties for dominated moves.

    Dominated move categories:
    1. Moves that give opponent immediate row completion
    2. Ring moves that trap the ring (no future mobility)
    3. Redundant placements in opening (placing rings too close)
    4. Moves that break own partial rows unnecessarily
    """
    mask = np.ones(ACTION_SPACE_SIZE)

    # Category 1: Giving opponent a row
    for move in get_opponent_winning_responses(game_state):
        enabling_moves = get_moves_enabling(move)
        for m in enabling_moves:
            mask[m] = 0.1

    # Category 2: Self-trapping moves
    for move in get_self_trapping_moves(game_state):
        mask[move] = 0.3

    # Category 3: Redundant opening placements
    if game_state.phase == Phase.RING_PLACEMENT:
        for move in get_redundant_placements(game_state):
            mask[move] = 0.5

    return mask
```

#### Implementation Location

**File:** `yinsh_ml/search/mcts.py`

```python
def get_policy_with_masking(self, state, network_policy):
    """Apply soft masking to raw network policy."""
    if self.config.use_soft_masking:
        soft_mask = compute_soft_mask(state)
        # Apply in log space for numerical stability
        masked_logits = np.log(network_policy + 1e-8) + np.log(soft_mask)
        return softmax(masked_logits)
    else:
        return network_policy
```

---

### Phase 5: Curriculum Self-Play

**Priority:** MEDIUM
**Expected Impact:** Stabilized early learning
**Effort:** Medium (1-2 days)

#### Problem

Pure self-play with two equally weak players:
- Neither punishes mistakes
- Value targets collapse toward 0
- No learning signal

#### Solution: Mixed Opponents

```python
def select_opponent(iteration: int, checkpoints: List[Path]) -> Policy:
    """
    Select opponent based on training phase.

    Early: Include random/weak opponents to provide learning signal
    Mid: Mix current with recent checkpoints
    Late: Pure self-play for refinement
    """
    if iteration < 5:
        # Early: 50% random, 50% current
        if random.random() < 0.5:
            return RandomPolicy()
        return CurrentNetworkPolicy()

    elif iteration < 20:
        # Mid: Mix current with historical checkpoints
        weights = [0.6, 0.25, 0.15]  # current, prev, prev-prev
        opponent_idx = random.choices([0, 1, 2], weights=weights)[0]

        if opponent_idx == 0:
            return CurrentNetworkPolicy()
        elif opponent_idx == 1 and len(checkpoints) >= 1:
            return load_checkpoint(checkpoints[-1])
        elif opponent_idx == 2 and len(checkpoints) >= 2:
            return load_checkpoint(checkpoints[-2])
        return CurrentNetworkPolicy()

    else:
        # Late: Pure self-play
        return CurrentNetworkPolicy()
```

#### Integration

**File:** `yinsh_ml/training/self_play.py`

```python
class CurriculumSelfPlayRunner:
    """Self-play with curriculum opponent selection."""

    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.checkpoint_history = []

    def generate_games(self, current_net, iteration: int) -> List[Game]:
        opponent = self.select_opponent(iteration)

        games = []
        for _ in range(self.config.games_per_iteration):
            # Alternate who plays white
            if random.random() < 0.5:
                game = play_game(white=current_net, black=opponent)
            else:
                game = play_game(white=opponent, black=current_net)
            games.append(game)

        return games
```

---

### Phase 6: Adaptive MCTS Budget

**Priority:** MEDIUM
**Expected Impact:** Better throughput/quality tradeoff
**Effort:** Low (already partially implemented)

#### Current State

We already have `early_simulations` and `late_simulations` config. Extend this to be iteration-aware:

```python
def get_mcts_budget(iteration: int, game_ply: int) -> int:
    """
    Adaptive MCTS budget based on training phase and game state.

    Principle: Early training wants diversity and throughput.
    Later training wants quality and depth.
    """
    # Base budget by iteration
    if iteration < 10:
        base = 64  # Fast, diverse games
    elif iteration < 30:
        base = 128  # Moderate depth
    else:
        base = 256  # Quality games

    # Adjust for game phase
    if game_ply < 10:  # Opening
        return base
    elif game_ply > 40:  # Endgame - more critical
        return int(base * 1.5)
    else:
        return base
```

---

## Implementation Order

```
┌─────────────────────────────────────────────────────────────────┐
│ Week 1: Foundation                                               │
├─────────────────────────────────────────────────────────────────┤
│ Day 1-2: Enhanced State Encoding                                 │
│   - Implement EnhancedStateEncoder                               │
│   - Update network input channels                                │
│   - Add threat/influence plane computations                      │
│                                                                  │
│ Day 3-4: Symmetry Augmentation                                   │
│   - Implement D6 transforms for hex board                        │
│   - Policy transform mapping                                     │
│   - Integration with training loop                               │
│                                                                  │
│ Day 5: Testing & Validation                                      │
│   - Unit tests for encoding                                      │
│   - Verify augmentation correctness                              │
│   - Baseline run with new encoding                               │
├─────────────────────────────────────────────────────────────────┤
│ Week 2: Instrumentation & Masking                                │
├─────────────────────────────────────────────────────────────────┤
│ Day 1-2: Enhanced Metrics                                        │
│   - MCTS agreement tracking                                      │
│   - Policy entropy logging                                       │
│   - Value calibration tests                                      │
│                                                                  │
│ Day 3: Tactical Benchmarks                                       │
│   - Create 10-20 test positions                                  │
│   - Implement benchmark runner                                   │
│                                                                  │
│ Day 4-5: Action Soft Masking                                     │
│   - Dominated move detection                                     │
│   - Soft mask integration                                        │
├─────────────────────────────────────────────────────────────────┤
│ Week 3: Training Dynamics                                        │
├─────────────────────────────────────────────────────────────────┤
│ Day 1-2: Curriculum Self-Play                                    │
│   - Opponent selection logic                                     │
│   - Checkpoint management                                        │
│                                                                  │
│ Day 3-4: Adaptive MCTS                                           │
│   - Iteration-aware budgets                                      │
│   - Game-phase adjustments                                       │
│                                                                  │
│ Day 5: Integration Testing                                       │
│   - Full pipeline test                                           │
│   - Performance profiling                                        │
├─────────────────────────────────────────────────────────────────┤
│ Week 4: Validation                                               │
├─────────────────────────────────────────────────────────────────┤
│ Day 1-3: Extended Training Run                                   │
│   - 50+ iterations with new architecture                         │
│   - Full metric collection                                       │
│                                                                  │
│ Day 4-5: Analysis & Documentation                                │
│   - Compare to baseline                                          │
│   - Document findings                                            │
│   - Plan next phase                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Success Criteria

### Phase 1 Complete (Week 1)
- [ ] Enhanced encoding implemented and tested
- [ ] Augmentation producing 12x samples correctly
- [ ] Network training with new encoding

### Phase 2 Complete (Week 2)
- [ ] All new metrics being logged
- [ ] Tactical benchmark suite with 10+ positions
- [ ] Soft masking reducing policy entropy on dominated moves

### Phase 3 Complete (Week 3)
- [ ] Curriculum self-play producing varied games
- [ ] Adaptive MCTS adjusting budget correctly

### Overall Success (Week 4)
- [ ] ELO > 1650 (vs baseline 1520-1580)
- [ ] Tactical benchmark solve rate > 80%
- [ ] MCTS agreement rate > 50% by iteration 30
- [ ] Policy entropy decreasing over training
- [ ] Value calibration improving over training

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Enhanced encoding breaks existing code | Feature flag, parallel implementation |
| Augmentation introduces bugs | Extensive unit tests, visual verification |
| New metrics slow down training | Async logging, sampling |
| Changes don't improve performance | A/B testing, incremental rollout |

---

## Files to Create/Modify

### New Files
- `yinsh_ml/utils/enhanced_encoding.py`
- `yinsh_ml/training/augmentation.py`
- `yinsh_ml/training/curriculum.py`
- `yinsh_ml/utils/training_metrics.py`
- `yinsh_ml/evaluation/tactical_benchmarks.py`
- `yinsh_ml/search/action_masking.py`

### Modified Files
- `yinsh_ml/network/model.py` (input channels)
- `yinsh_ml/network/wrapper.py` (encoding selection)
- `yinsh_ml/training/self_play.py` (augmentation, curriculum)
- `yinsh_ml/training/supervisor.py` (metrics integration)
- `yinsh_ml/experiments/experiment_config.py` (new config options)

---

## Configuration Changes

```yaml
# New config options
encoding:
  type: enhanced  # or "basic" for backward compatibility
  include_threats: true
  include_influence: true
  include_mobility: true

augmentation:
  enabled: true
  include_reflections: true
  max_augmentations: 12  # Full D6 group

action_masking:
  enabled: true
  dominated_move_penalty: 0.1
  self_trap_penalty: 0.3

curriculum:
  enabled: true
  random_opponent_until: 5
  mixed_opponents_until: 20
  checkpoint_retention: 5

metrics:
  track_mcts_agreement: true
  track_policy_entropy: true
  track_value_calibration: true
  tactical_benchmark_interval: 10
```

---

## References

- AlphaZero paper: Silver et al., 2017
- Peer feedback: January 2026 review
- Previous experiments: `experiments/` directory
- Yinsh rules: `yinsh_ml/game/constants.py`

---

## Appendix: Yinsh-Specific Considerations

### Board Topology

Yinsh uses an 11x11 grid with hexagonal connectivity. Not all cells are valid (corners cut off). The hex coordinate system requires careful handling for:
- Rotations (60° increments around center)
- Reflections (6 axes of symmetry)
- Line detection (6 directions instead of 4)

### Game Phases

1. **Ring Placement** (moves 1-10): Place 5 rings each
2. **Main Game**: Move rings, flip markers, form rows
3. **Ring Removal**: After forming row of 5, remove a ring

Each phase may need different encoding emphasis:
- Placement: Ring positioning, center control
- Main game: Threats, mobility, marker chains
- Removal: Score tracking, endgame tactics

### Action Space

- Ring placement: 85 valid positions (during placement phase)
- Ring moves: Up to ~20 destinations per ring × 5 rings = ~100 moves
- Total action space: ~85 × 85 = 7,225 (from, to) pairs

This is large but manageable. Soft masking can significantly reduce effective branching factor.
