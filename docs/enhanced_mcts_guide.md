# Enhanced MCTS Implementation Guide

## Overview

The Enhanced MCTS implementation incorporates findings from comprehensive analysis of Yinsh self-play data, including linear analysis, phase-based analysis, and heuristic evaluation. This implementation provides a drop-in replacement for the standard MCTS with improved performance through analysis-driven optimizations.

## Key Features

### 1. Phase-Aware Simulation Budget Allocation

Based on phase analysis findings, the enhanced MCTS adjusts simulation budgets according to game phase:

- **Early Game (moves 1-15)**: Standard budget (1.0x)
- **Mid Game (moves 16-35)**: Increased budget (1.2x) for complexity
- **Late Game (moves 36+)**: Reduced budget (0.8x) for tactical focus

```python
config = EnhancedMCTSConfig(
    use_phase_aware_budget=True,
    phase_budget_multipliers={
        GamePhase.EARLY: 1.0,
        GamePhase.MID: 1.2,
        GamePhase.LATE: 0.8
    }
)
```

### 2. Enhanced UCB Selection

The UCB formula is enhanced with phase-aware exploration adjustments:

- **Early Game**: More exploration (1.2x multiplier)
- **Mid Game**: Balanced exploration (1.0x multiplier)
- **Late Game**: Less exploration, more exploitation (0.8x multiplier)

### 3. Heuristic Evaluation Integration

Incorporates the heuristic evaluation function for faster simulations:

```python
config = EnhancedMCTSConfig(
    use_heuristic_evaluation=True,
    heuristic_weight=0.3  # 30% heuristic, 70% neural network
)
```

The heuristic evaluation provides:
- Fast position evaluation (~2000 evaluations/second)
- Phase-aware weighting (more heuristic in early game)
- Combined with neural network predictions

### 4. Heuristic Guidance Integration (Task 10)

Advanced heuristic guidance features for improved MCTS performance:

```python
config = EnhancedMCTSConfig(
    use_heuristic_guidance=True,
    heuristic_alpha=0.3,      # Weight for heuristic in UCB1 combination
    epsilon_greedy=0.4,       # Epsilon for greedy rollouts
    use_heuristic_rollouts=True  # Use heuristic for leaf node evaluation
)
```

Key features:
- **Heuristic-guided UCB1**: `combined_score = (1-alpha) * UCB1 + alpha * heuristic_eval`
- **Epsilon-greedy rollouts**: Biased move selection during simulation
- **Heuristic leaf evaluation**: Replace random rollouts with heuristic assessment

### 5. Analysis-Driven Optimizations

- **Linear Analysis**: Improved UCB exploration constants
- **Phase Analysis**: Game phase awareness throughout search
- **Heuristic Analysis**: Fast evaluation for simulation rollouts

## Usage

### Basic Usage

```python
from yinsh_ml.training.enhanced_mcts import EnhancedMCTS, EnhancedMCTSConfig
from yinsh_ml.network.wrapper import NetworkWrapper

# Create configuration
config = EnhancedMCTSConfig(
    num_simulations=100,
    use_heuristic_evaluation=True,
    use_phase_aware_budget=True,
    use_enhanced_ucb=True
)

# Create enhanced MCTS
mcts = EnhancedMCTS(network, config)

# Perform search
policy = mcts.search(game_state, move_number)
```

### Self-Play Integration

```python
from yinsh_ml.training.enhanced_self_play import EnhancedSelfPlay

# Create enhanced self-play
self_play = EnhancedSelfPlay(
    network=network,
    num_workers=4,
    enhanced_config=config
)

# Generate games
games = self_play.generate_games(100)
```

### Configuration File Usage

```python
import yaml
from yinsh_ml.training.enhanced_self_play import create_enhanced_self_play_from_config

# Load configuration
with open('configs/enhanced_mcts_example.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)

# Create self-play from config
self_play = create_enhanced_self_play_from_config(network, config_dict)
```

## Configuration Options

### Standard MCTS Parameters

- `num_simulations`: Base simulation budget (default: 100)
- `late_simulations`: Late game simulation budget (default: None, uses num_simulations)
- `simulation_switch_ply`: Move number to switch budgets (default: 20)
- `c_puct`: Exploration constant (default: 1.0)
- `dirichlet_alpha`: Dirichlet noise parameter (default: 0.3)
- `value_weight`: Value weighting in UCB (default: 1.0)
- `max_depth`: Maximum search depth (default: 50)

### Temperature Parameters

- `initial_temp`: Starting temperature (default: 1.0)
- `final_temp`: Final temperature (default: 0.1)
- `annealing_steps`: Annealing duration (default: 30)
- `temp_clamp_fraction`: Temperature clamp fraction (default: 0.8)

### Analysis Integration Parameters

- `use_heuristic_evaluation`: Enable heuristic evaluation (default: True)
- `use_phase_aware_budget`: Enable phase-aware budgeting (default: True)
- `use_enhanced_ucb`: Enable enhanced UCB (default: True)
- `heuristic_weight`: Heuristic vs neural network weight (default: 0.3)
- `phase_budget_multipliers`: Phase-specific budget multipliers

### Heuristic Guidance Parameters (Task 10)

- `use_heuristic_guidance`: Enable heuristic-guided UCB1 (default: True)
- `heuristic_alpha`: Weight for heuristic in UCB1 combination (default: 0.3)
- `epsilon_greedy`: Epsilon for greedy rollouts (default: 0.4)
- `use_heuristic_rollouts`: Use heuristic for leaf node evaluation (default: True)

## Performance Benefits

### Simulation Efficiency

- **Phase-aware budgeting**: 20% more simulations in mid-game, 20% fewer in late-game
- **Heuristic evaluation**: ~2000 evaluations/second vs ~100 neural network evaluations/second
- **Enhanced UCB**: Better exploration-exploitation balance

### Game Quality

- **Phase awareness**: Better strategic decisions in different game phases
- **Heuristic integration**: More accurate position evaluation
- **Analysis insights**: Incorporates discovered patterns and correlations

## Migration from Standard MCTS

The enhanced MCTS is designed as a drop-in replacement:

1. **Import changes**:
   ```python
   # Old
   from yinsh_ml.training.self_play import MCTS
   
   # New
   from yinsh_ml.training.enhanced_mcts import EnhancedMCTS, EnhancedMCTSConfig
   ```

2. **Configuration changes**:
   ```python
   # Old
   mcts = MCTS(network, num_simulations=100, c_puct=1.0, ...)
   
   # New
   config = EnhancedMCTSConfig(num_simulations=100, c_puct=1.0, ...)
   mcts = EnhancedMCTS(network, config)
   ```

3. **Usage remains the same**:
   ```python
   policy = mcts.search(game_state, move_number)
   ```

## Testing

Run the example usage script to test the implementation:

```bash
python examples/enhanced_mcts_usage.py
```

This will demonstrate:
- Basic enhanced MCTS functionality
- Phase-aware features
- Heuristic integration
- Configuration file usage

## Analysis Findings Integration

### Linear Analysis Insights

- **Correlation patterns**: Feature importance varies by game phase
- **Value head performance**: Different accuracy across phases
- **Move selection patterns**: Phase-specific strategic preferences

### Phase-Based Analysis

- **Early game**: Focus on ring placement and basic positioning
- **Mid game**: Complex tactical sequences and ring management
- **Late game**: Tactical precision and endgame optimization

### Heuristic Analysis

- **Fast evaluation**: Weighted feature combination for position assessment
- **Phase weighting**: Different feature importance by game phase
- **Speed optimization**: Designed for high-frequency MCTS simulations

## Future Enhancements

Potential areas for further improvement:

1. **Dynamic phase detection**: More sophisticated phase classification
2. **Adaptive budgeting**: Machine learning-based budget allocation
3. **Multi-objective optimization**: Balancing multiple strategic goals
4. **Real-time analysis**: Continuous learning from game outcomes

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
2. **Configuration errors**: Check phase budget multipliers use GamePhase enum
3. **Performance issues**: Adjust heuristic_weight for speed vs accuracy trade-off

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## References

- [Phase Analysis Documentation](phase_analysis.md)
- [Heuristic Evaluation Guide](heuristic_evaluation.md)
- [Linear Analysis Results](linear_analysis.md)
- [Standard MCTS Implementation](standard_mcts.md)
