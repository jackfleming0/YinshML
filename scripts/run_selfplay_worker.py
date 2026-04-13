import argparse
import sys
from pathlib import Path

from yinsh_ml.training.self_play import SelfPlay
from yinsh_ml.network.wrapper import NetworkWrapper


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a self-play batch and exit')
    parser.add_argument('--checkpoint', type=str, required=False, help='Path to model checkpoint to load')
    parser.add_argument('--games', type=int, default=10, help='Number of games to generate')
    parser.add_argument('--sims', type=int, default=96, help='MCTS simulations (early)')
    parser.add_argument('--late-sims', type=int, default=None, help='MCTS simulations (late)')
    parser.add_argument('--switch-ply', type=int, default=20, help='Simulation switch ply')
    parser.add_argument('--c-puct', type=float, default=1.0, help='cPUCT')
    parser.add_argument('--alpha', type=float, default=0.3, help='Dirichlet alpha')
    parser.add_argument('--value-weight', type=float, default=1.0, help='Value term weight')
    parser.add_argument('--max-depth', type=int, default=300, help='Max depth')
    parser.add_argument('--temp0', type=float, default=1.0, help='Initial temperature')
    parser.add_argument('--temp1', type=float, default=0.1, help='Final temperature')
    parser.add_argument('--anneal', type=int, default=30, help='Annealing steps')
    parser.add_argument('--clamp-frac', type=float, default=0.6, help='Clamp fraction')
    args = parser.parse_args()

    network = NetworkWrapper()
    if args.checkpoint:
        network.load_model(args.checkpoint)

    sp = SelfPlay(
        network=network,
        num_workers=1,
        num_simulations=args.sims,
        late_simulations=args.late_sims,
        simulation_switch_ply=args.switch_ply,
        c_puct=args.c_puct,
        dirichlet_alpha=args.alpha,
        value_weight=args.value_weight,
        max_depth=args.max_depth,
        initial_temp=args.temp0,
        final_temp=args.temp1,
        annealing_steps=args.anneal,
        temp_clamp_fraction=args.clamp_frac,
    )

    results = sp.generate_games(args.games)
    print(f"Generated {len(results)} games")


if __name__ == '__main__':
    main()




