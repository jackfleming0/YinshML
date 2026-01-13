# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YinshML is an AlphaZero-inspired machine learning framework for the YINSH board game. It combines classical AI techniques (MCTS, negamax, alpha-beta pruning) with modern deep learning to train game-playing agents through self-play.

**Current Branch**: `heuristic_seeding` - Integrating learned heuristics into MCTS for improved training efficiency.

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate    # On Windows

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### Testing
```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/test_game_logic.py

# Run tests matching pattern
pytest -k "test_heuristic"

# Skip slow tests
pytest -m "not slow"

# Run tests in specific directory
pytest yinsh_ml/tests/

# Note: macOS users - NumPy Accelerate backend is automatically disabled
# via conftest.py to prevent floating point exceptions
```

### Training and Self-Play
```bash
# Generate training data via large-scale self-play
python run_large_scale_selfplay.py

# Run training supervisor (orchestrates self-play + training)
python scripts/run_training.py

# Monitor training progress (run in separate terminal)
./monitor_training.sh

# Evaluate heuristic quality
python run_heuristic_evaluator.py

# Analyze game dataset
python run_complete_analysis.py
```

### Demos and Visualization
```bash
# Interactive gameplay demo
python real_time_demo.py

# Automated demo
python final_demo.py

# Launch training dashboard
python run_dashboard.py
```

### MCTS Hyperparameter Tuning
```bash
# Tune MCTS parameters
python scripts/tune_mcts_hyperparameters.py

# Quick smoke test
python scripts/tune_mcts_hyperparameters.py --smoke-test
```

### CLI Interface
```bash
# Main CLI entry point
yinsh-track --help

# Available commands in yinsh_ml/cli/commands/
```

## High-Level Architecture

### System Overview
```
┌─────────────────────────────────────────────────┐
│           Training Supervisor                   │
│  (Orchestrates self-play + network training)    │
└──────────────┬─────────────────┬────────────────┘
               │                 │
       ┌───────▼──────┐   ┌──────▼─────────┐
       │  Self-Play   │   │ Neural Network │
       │  Game Runner │   │   Training     │
       └──────┬───────┘   └────────┬───────┘
              │                    │
       ┌──────▼────────────────────▼────┐
       │     Search Algorithms          │
       │  (MCTS, Negamax + Transposition)│
       └──────┬─────────────────────────┘
              │
       ┌──────▼──────────────────────┐
       │  Heuristic Evaluation       │
       │  (7 learned features)       │
       └──────┬──────────────────────┘
              │
       ┌──────▼──────────────────────┐
       │  Core Game Logic            │
       │  (Board, Moves, Rules)      │
       └─────────────────────────────┘
```

### Key Module Responsibilities

**`yinsh_ml/game/`** - Core YINSH game implementation
- `constants.py` - Game rules, board geometry (11×11 hex grid)
- `board.py` - Board state management
- `game_state.py` - Game state with move history and phase tracking
- `moves.py` - Move generation logic
- `types.py` - Enums (MoveType, GamePhase, Player)
- `zobrist.py` - Zobrist hashing for position encoding (fast position caching)

**`yinsh_ml/search/`** - Search algorithms
- `mcts.py` - Monte Carlo Tree Search with configurable evaluation modes
- `transposition_table.py` - Position cache using Zobrist hashing (depth-preferred replacement)
- `node_type.py` - Node type enums for alpha-beta pruning (EXACT, LOWER_BOUND, UPPER_BOUND)
- See `yinsh_ml/search/README.md` for detailed search system documentation

**`yinsh_ml/agents/`** - Game-playing agents
- `heuristic_agent.py` - Pure heuristic agent using negamax with alpha-beta pruning
- Includes built-in transposition table support (enabled by default)
- `HeuristicAgentConfig` - Configure search depth, transposition table size, etc.

**`yinsh_ml/heuristics/`** - Heuristic evaluation system
- `evaluator.py` - Main evaluator with 7 learned features
- `features.py` - Feature extraction (runs, centrality, mobility, etc.)
- `phase_detection.py` - Game phase classification
- `weight_manager.py` - Phase-specific weight management
- Weights learned from 100K+ games analysis

**`yinsh_ml/network/`** - Neural network
- `model.py` - YinshNetwork (ResNet-style with attention)
- Input: 6×11×11 channels (white/black rings/markers, valid moves, phase)
- Outputs: Policy head (121×121 move probabilities) + Value head (position evaluation)
- `wrapper.py` - NetworkWrapper for inference

**`yinsh_ml/training/`** - Training pipeline
- `trainer.py` - YinshTrainer (network training loop)
- `self_play.py` - Self-play data generation
- `supervisor.py` - TrainingSupervisor (orchestrates everything)
- `enhanced_mcts.py` - MCTS with additional features

**`yinsh_ml/self_play/`** - Self-play infrastructure
- `game_runner.py` - SelfPlayRunner (manages game execution)
- `policies.py` - Move selection policies (Random, Heuristic, MCTS, Adaptive)
- `data_storage.py` - Parquet-based data persistence
- `quality_metrics.py` - Game quality analysis

**`yinsh_ml/memory/`** - Memory optimization
- `game_state_pool.py` - GameState object reuse (reduces GC pressure)
- `tensor_pool.py` - Tensor reuse for efficient batching
- `adaptive.py` - Adaptive pool sizing
- `zero_copy.py` - GPU zero-copy transfers

**`yinsh_ml/utils/`** - Utilities
- `encoding.py` - StateEncoder (state ↔ tensor conversion)
- `metrics_logger.py` - Training metrics tracking
- `tournament.py` - Tournament infrastructure
- `elo_manager.py` - ELO rating system

## Critical Implementation Details

### YINSH Game Mechanics
- **Board**: 11×11 hexagonal grid (99 valid positions)
- **Pieces**: Rings (movable) + Markers (flip when ring passes)
- **Phases**: RING_PLACEMENT (place 5 rings) → MAIN_GAME (normal play) → RING_REMOVAL (after 3+ rows captured)
- **Win Condition**: First to capture 3 complete rows (5+ consecutive markers)
- **Position Notation**: Algebraic (A-K columns, 1-11 rows)

### State Encoding (6 Channels)
1. White Rings (11×11 binary)
2. Black Rings (11×11 binary)
3. White Markers (11×11 binary)
4. Black Markers (11×11 binary)
5. Valid Moves (11×11 binary)
6. Game Phase (11×11 scalar)

### Heuristic Feature Set (7 Features)
Learned from 100K+ games analysis (see `QUICK_START_GUIDE.md` for details):
1. **Completed Runs Differential** (weight: 0.239) - Most important
2. **Ring Centrality Score** (weight: 0.211)
3. **Ring Spread** (weight: 0.187)
4. **Potential Runs Count** (weight: 0.171)
5. **Connected Marker Chains** (weight: 0.086)
6. **Ring Mobility** (weight: 0.071)
7. **Edge Proximity Score** (weight: 0.036)

Weights are **phase-specific** - different priorities for early/mid/late game.

### Transposition Table Integration
- **Enabled by default** in `HeuristicAgent`
- Uses Zobrist hashing for position encoding
- Depth-preferred replacement policy (keeps deeper searches)
- Typical hit rates: 60-80% in game tree search
- Default size: 2^20 entries (1M positions, ~40-50MB)
- Clear between games: `agent.clear_transposition_table()`

### MCTS Evaluation Modes
Configured via `MCTSConfig.evaluation_mode`:
- `PURE_HEURISTIC` - Only use heuristic evaluation
- `PURE_NEURAL` - Only use neural network
- `HYBRID` - Weighted combination (neural + heuristic)

### Training Data Flow
1. **Self-Play**: Generate games using MCTS with heuristic evaluation
2. **Experience Storage**: Store (state, policy, value) tuples
3. **Subsampling**: Games >100 moves are subsampled to reduce redundancy
4. **Training**: Policy head (cross-entropy) + Value head (MSE)
5. **Evaluation**: Tournament vs previous versions + ELO rating

## Common Workflows

### Adding a New Heuristic Feature
1. Add feature extraction in `yinsh_ml/heuristics/features.py`
2. Update `YinshHeuristics.calculate_features()` in `evaluator.py`
3. Add phase-specific weights in `weight_manager.py`
4. Run analysis to determine optimal weights: `python run_complete_analysis.py`
5. Test with HeuristicAgent: `python scripts/test_game.py`

### Modifying MCTS Search
1. Edit `yinsh_ml/search/mcts.py`
2. Update `MCTSConfig` if adding parameters
3. Test with smoke test: `python scripts/tune_mcts_hyperparameters.py --smoke-test`
4. Run full hyperparameter tuning if needed

### Adjusting Neural Network Architecture
1. Modify `YinshNetwork` in `yinsh_ml/network/model.py`
2. Update `NetworkWrapper` if interface changes
3. Test with: `pytest yinsh_ml/network/tests/`
4. Retrain from scratch or fine-tune existing model

### Running Experiments
1. Configure experiment in `configs/`
2. Run training: `python scripts/run_training.py --config configs/my_config.yaml`
3. Monitor via dashboard: `python run_dashboard.py`
4. Analyze results: `python run_complete_analysis.py`

## Important Files Reference

**Documentation**:
- `TRAINING_REFACTOR_PLAN.md` - **Comprehensive refactor plan for superhuman play** (start here for training improvements)
- `QUICK_START_GUIDE.md` - Quick start with 100K game analysis results
- `ANALYSIS_SUMMARY_100K_GAMES.md` - Detailed findings from 100K games
- `ROADMAP_TO_ALPHAZERO.md` - Roadmap for AlphaZero implementation
- `yinsh_ml/search/README.md` - Search system documentation (Zobrist + Transposition Table)
- `START_SELF_PLAY.md` - Guide for starting self-play

**Configuration**:
- `requirements.txt` - Python dependencies
- `setup.py` - Package setup and CLI entry points
- `conftest.py` - Pytest configuration (macOS NumPy workaround)
- `configs/` - Training configuration files

**Key Scripts**:
- `run_large_scale_selfplay.py` - Generate training data (100K+ games)
- `scripts/run_training.py` - Main training entry point
- `scripts/tune_mcts_hyperparameters.py` - MCTS parameter optimization
- `run_heuristic_evaluator.py` - Evaluate heuristic quality
- `run_complete_analysis.py` - Comprehensive game analysis

**Analysis Data**:
- `analysis_data/` - 100K games in JSON format (~9M positions)
- `analysis_output/` - Trained models and analysis results
- `large_scale_selfplay_data/parquet_data/` - 100K games in Parquet format (344MB)

## Data Storage

**Parquet Format**: All training data stored in Parquet for efficiency
- `GameRecord` schema: state, policy, value, metadata
- ~3KB per game (vs ~1.5MB JSON)
- Fast parallel loading with PyArrow

**Analysis Results**:
- `analysis_output/heuristic_evaluator_model.pkl` - Trained heuristic model (ready to use)
- `*.png` - Visualizations
- `*_report.txt` - Statistical reports

## Testing Considerations

### macOS NumPy Issue
- **Problem**: NumPy's Accelerate backend can cause floating point exceptions
- **Solution**: Automatically disabled via `conftest.py` (`NPY_DISABLE_MACOS_ACCELERATE=1`)
- No action needed - handled automatically by pytest

### Test Organization
- `tests/` - Root-level integration tests
- `yinsh_ml/tests/` - Component-specific unit tests
- Mark slow tests with `@pytest.mark.slow`
- Skip with: `pytest -m "not slow"`

### Key Test Categories
- Game logic: Move generation, state transitions, win conditions
- Search: MCTS correctness, transposition table hit rates
- Heuristics: Feature extraction, evaluation scoring
- Training: Self-play generation, network training loop
- Memory: Pool efficiency, leak detection

## Performance Optimization

### Memory Pools
- `GameStatePool`: Reuses GameState objects (reduces allocation)
- `TensorPool`: Reuses tensors for batching
- Benefits: Reduced GC pressure, improved training throughput

### Transposition Table
- Caches evaluated positions during search
- 60-80% typical hit rate (avoids redundant evaluations)
- Increase size for deeper searches: `transposition_table_size_power=22` (4M entries)

### Parquet Storage
- Use Parquet instead of JSON for training data
- 500× size reduction (1.5MB → 3KB per game)
- Fast parallel loading

### GPU Acceleration
- Zero-copy transfers in `yinsh_ml/memory/zero_copy.py`
- Batch processing for neural network inference
- Multi-GPU support in training pipeline

## Key Architectural Patterns

### Phase-Aware Evaluation
Heuristic weights adjust based on game phase:
- **Early Game** (turns 1-15): Prioritize mobility
- **Mid Game** (turns 16-35): Balanced approach
- **Late Game** (turns 36+): Heavily weight completed runs + center control

### Hybrid Search
MCTS combines neural network + heuristic evaluation:
- Neural network provides policy prior (move probabilities)
- Heuristic provides value estimate at leaf nodes
- Adaptive weighting based on training progress

### Experience Replay
- Circular buffer stores game trajectories
- Subsampling for long games (keeps critical positions)
- Shuffle batches during training

### Self-Play Policies
- **Random**: Baseline (uniform move selection)
- **Heuristic**: Pure heuristic negamax search
- **MCTS**: MCTS with neural network
- **Adaptive**: Switches policies based on training progress

## Recent Development (heuristic_seeding branch)

**Latest Changes**:
- Added transposition table + Zobrist hashing to HeuristicAgent (aef3d08)
- Refactored MCTS hyperparameter tuning with heuristic integration (e446d24)
- Post-heuristic incorporation work (21743c1)

**Current Focus**:
- Integrating learned heuristics into MCTS for better training efficiency
- Optimizing transposition table performance
- Tuning MCTS parameters for hybrid evaluation mode

## Troubleshooting

### Low MCTS Performance
- Check hit rate: `agent.last_search_stats['transposition_table_metrics']['hit_rate']`
- If <50%, increase table size or clear between games
- Verify Zobrist hashing is deterministic (use `zobrist_seed`)

### Memory Issues
- Enable memory pools: `GameStatePool` + `TensorPool`
- Reduce batch size in training
- Clear transposition table periodically
- Use Parquet instead of JSON for storage

### Slow Training
- Profile with `yinsh_ml/monitoring/performance_profiler.py`
- Check GPU utilization (should be >80%)
- Increase batch size if memory allows
- Enable zero-copy transfers for GPU

### Test Failures on macOS
- NumPy Accelerate issue: Already handled via `conftest.py`
- If still failing, manually set: `export NPY_DISABLE_MACOS_ACCELERATE=1`

## Additional Resources

- **YINSH Rules**: See `yinsh_ml/game/constants.py` for official rules
- **AlphaZero Paper**: Basis for training methodology
- **100K Game Analysis**: `QUICK_START_GUIDE.md` for key findings
- **Search System Docs**: `yinsh_ml/search/README.md` for Zobrist + Transposition Table details
