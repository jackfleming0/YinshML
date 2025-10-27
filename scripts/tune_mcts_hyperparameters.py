#!/usr/bin/env python3
"""Script for tuning MCTS hyperparameters."""

import sys
import os
import logging
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from yinsh_ml.training.mcts_hyperparameter_tuning import (
    MCTSHyperparameterTuner, 
    TuningConfig, 
    HyperparameterSpace
)
from yinsh_ml.training.enhanced_mcts import EnhancedMCTSConfig
from yinsh_ml.network.wrapper import NetworkWrapper


def create_mock_network():
    """Create a mock network for testing."""
    class MockNetwork:
        def predict(self, state_tensor):
            import numpy as np
            policy = np.random.random(1000)
            policy /= policy.sum()
            value = np.random.uniform(-1, 1)
            return policy, value
    
    return MockNetwork()


def create_baseline_config():
    """Create baseline MCTS configuration."""
    return EnhancedMCTSConfig(
        num_simulations=1000,
        c_puct=1.414,  # √2
        dirichlet_alpha=0.3,
        value_weight=1.0,
        max_depth=50,
        initial_temp=1.0,
        final_temp=0.1,
        annealing_steps=30,
        temp_clamp_fraction=0.8,
        use_heuristic_evaluation=True,
        use_phase_aware_budget=True,
        use_enhanced_ucb=True,
        heuristic_weight=0.3,
        use_heuristic_guidance=True,
        heuristic_alpha=0.3,
        epsilon_greedy=0.4,
        use_heuristic_rollouts=True
    )


def main():
    """Main hyperparameter tuning script."""
    parser = argparse.ArgumentParser(description="Tune MCTS hyperparameters")
    parser.add_argument("--method", choices=["grid", "bayesian"], default="grid",
                       help="Optimization method")
    parser.add_argument("--games", type=int, default=10,
                       help="Number of evaluation games per parameter set")
    parser.add_argument("--workers", type=int, default=2,
                       help="Number of parallel workers")
    parser.add_argument("--no-move-cap", action="store_true",
                       help="Disable per-move time cap during evaluation")
    parser.add_argument("--max-moves", type=int, default=300,
                       help="Maximum moves per game (prevents infinite games)")
    parser.add_argument("--run-post-eval", action="store_true",
                       help="After tuning, evaluate best config vs baselines")
    parser.add_argument("--output", type=str, default="mcts_tuning_results.json",
                       help="Output file for results")
    parser.add_argument("--report", type=str, default="mcts_tuning_report.txt",
                       help="Output file for report")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("MCTSTuning")
    logger.info("Starting MCTS hyperparameter tuning")
    
    try:
        # Create components
        network = create_mock_network()
        baseline_config = create_baseline_config()
        
        # Create tuning configuration
        tuning_config = TuningConfig(
            evaluation_games=args.games,
            max_move_time=3.0,
            enforce_move_time_cap=(not args.no_move_cap),
            max_moves_per_game=args.max_moves,
            min_win_rate_improvement=0.05,
            use_bayesian_optimization=(args.method == "bayesian"),
            bayesian_iterations=20,
            max_workers=args.workers,
            timeout_per_game=300
        )
        
        # Create hyperparameter space
        hyperparameter_space = HyperparameterSpace(
            c_puct_min=0.5,
            c_puct_max=2.0,
            c_puct_step=0.5,
            heuristic_alpha_min=0.1,
            heuristic_alpha_max=0.7,
            heuristic_alpha_step=0.2,
            epsilon_greedy_min=0.2,
            epsilon_greedy_max=0.6,
            epsilon_greedy_step=0.2,
            num_simulations_min=1000,
            num_simulations_max=3000,
            num_simulations_step=1000,
            max_depth_min=30,
            max_depth_max=70,
            max_depth_step=20
        )
        
        # Create tuner
        tuner = MCTSHyperparameterTuner(
            network=network,
            baseline_config=baseline_config,
            tuning_config=tuning_config
        )
        tuner.hyperparameter_space = hyperparameter_space
        
        # Run optimization
        logger.info(f"Starting {args.method} search with {args.games} games per parameter set")
        
        if args.method == "grid":
            results = tuner.grid_search()
        else:
            results = tuner.bayesian_optimization()
        
        # Save results
        tuner.save_results(args.output)
        logger.info(f"Results saved to {args.output}")
        
        # Generate and save report
        report = tuner.generate_report()
        with open(args.report, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {args.report}")

        # Optional post-tuning evaluation vs baselines
        if args.run_post_eval and tuner.best_result:
            logger.info("Running post-tuning evaluation vs baselines...")
            baselines = [
                {"name": "random", "params": {"c_puct": 1.0, "heuristic_alpha": 0.0, "epsilon_greedy": 1.0, "num_simulations": 1, "max_depth": 10}},
                {"name": "weak_mcts_low_sims", "params": {"c_puct": 1.5, "heuristic_alpha": 0.2, "epsilon_greedy": 0.5, "num_simulations": 100, "max_depth": 30}},
                {"name": "weak_mcts_high_c", "params": {"c_puct": 2.0, "heuristic_alpha": 0.2, "epsilon_greedy": 0.4, "num_simulations": 200, "max_depth": 30}},
                {"name": "phase_unaware_like", "params": {"c_puct": 1.0, "heuristic_alpha": 0.1, "epsilon_greedy": 0.3, "num_simulations": 200, "max_depth": 20}},
            ]

            tuned_params = tuner.best_result.parameters

            def eval_head_to_head(params_a, params_b, games=20):
                from yinsh_ml.training.enhanced_mcts import EnhancedMCTS
                from yinsh_ml.training.enhanced_mcts import EnhancedMCTSConfig
                from yinsh_ml.game.game_state import GameState
                from yinsh_ml.game.constants import Player
                import numpy as np

                # Reuse mock network from earlier creator
                network_local = create_mock_network()
                cfg_a = create_baseline_config()
                for k, v in params_a.items():
                    setattr(cfg_a, k, v)
                cfg_b = create_baseline_config()
                for k, v in params_b.items():
                    setattr(cfg_b, k, v)

                mcts_a = EnhancedMCTS(network_local, cfg_a)
                mcts_b = EnhancedMCTS(network_local, cfg_b)

                wins_a = 0
                draws = 0
                for g in range(games):
                    state = GameState()
                    move_count = 0
                    while not state.is_terminal() and move_count < tuning_config.max_moves_per_game:
                        current = mcts_a if (state.current_player == Player.WHITE) else mcts_b
                        policy = current.search(state, move_count + 1)
                        valid_moves = state.get_valid_moves()
                        if not valid_moves:
                            break
                        # Greedy move
                        import numpy as np
                        move_probs = []
                        for move in valid_moves:
                            idx = current.state_encoder.move_to_index(move)
                            move_probs.append(policy[idx] if 0 <= idx < len(policy) else 0.0)
                        state.make_move(valid_moves[int(np.argmax(move_probs))])
                        move_count += 1

                    if state.is_terminal():
                        w = state.get_winner()
                        if w == Player.WHITE:
                            wins_a += 1
                        elif w is None:
                            draws += 1
                    else:
                        draws += 1

                return wins_a, draws, games

            # Evaluate tuned vs each baseline
            for b in baselines:
                name = b["name"]
                wins, draws, games = eval_head_to_head(tuned_params, b["params"], games=max(10, args.games))
                logger.info(f"Post-eval vs {name}: tuned wins={wins}/{games}, draws={draws}")
        
        # Print summary
        print("\n" + "="*60)
        print("MCTS HYPERPARAMETER TUNING COMPLETE")
        print("="*60)
        print(f"Method: {args.method}")
        print(f"Total parameter combinations tested: {len(results)}")
        print(f"Evaluation games per combination: {args.games}")
        
        if tuner.best_result:
            print(f"\nBest Parameters:")
            for param, value in tuner.best_result.parameters.items():
                print(f"  {param}: {value}")
            print(f"Best Win Rate: {tuner.best_result.win_rate:.3f}")
            print(f"Best Move Time: {tuner.best_result.move_time_avg:.3f}s")
        
        print(f"\nResults saved to: {args.output}")
        print(f"Report saved to: {args.report}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during hyperparameter tuning: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
