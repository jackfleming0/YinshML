#!/usr/bin/env python3
"""
Run YinshML Training Experiment

A simplified entry point for running hyperparameter tuning experiments.

Usage:
    # Run a specific experiment config
    python scripts/run_experiment.py experiments/configs/baseline_001.yaml

    # Run with live monitoring in another terminal
    python scripts/run_experiment.py experiments/configs/baseline_001.yaml &
    python scripts/monitor_training_live.py --latest

    # Resume a stopped experiment
    python scripts/run_experiment.py --resume abc123

    # Run all Phase A experiments sequentially
    python scripts/run_experiment.py --batch experiments/configs/baseline_*.yaml

Examples:
    # Quick test (2 iterations instead of 10)
    python scripts/run_experiment.py experiments/configs/baseline_001.yaml --iterations 2

    # Verbose output
    python scripts/run_experiment.py experiments/configs/baseline_001.yaml -v
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_logging(verbose: bool = False, log_file: str = None):
    """Configure logging for experiment run."""
    level = logging.DEBUG if verbose else logging.INFO

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    # Reduce noise from some modules
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def run_single_experiment(config_path: str, output_dir: str, iterations: int = None, resume: str = None):
    """Run a single experiment."""
    from yinsh_ml.experiments import run_experiment, load_config, ExperimentConfig

    config = load_config(config_path)

    # Override iterations if specified
    if iterations is not None:
        config.iterations = iterations
        logging.info(f"Overriding iterations: {iterations}")

    # Run the experiment
    from yinsh_ml.experiments.experiment_runner import ExperimentRunner
    runner = ExperimentRunner(config, output_dir, resume)
    return runner.run()


def run_batch(config_patterns: list, output_dir: str, iterations: int = None):
    """Run multiple experiments sequentially."""
    import glob

    # Expand patterns
    config_files = []
    for pattern in config_patterns:
        matches = glob.glob(pattern)
        config_files.extend(sorted(matches))

    if not config_files:
        logging.error(f"No config files found matching patterns: {config_patterns}")
        return []

    logging.info(f"Running batch of {len(config_files)} experiments:")
    for cf in config_files:
        logging.info(f"  - {cf}")

    results = []
    for i, config_path in enumerate(config_files):
        logging.info(f"\n{'='*60}")
        logging.info(f"BATCH EXPERIMENT {i+1}/{len(config_files)}: {Path(config_path).stem}")
        logging.info(f"{'='*60}\n")

        try:
            result = run_single_experiment(config_path, output_dir, iterations)
            results.append({
                'config': config_path,
                'status': result['final_status'],
                'experiment_id': result['experiment_id']
            })
        except Exception as e:
            logging.error(f"Experiment failed: {e}")
            results.append({
                'config': config_path,
                'status': 'failed',
                'error': str(e)
            })

    # Print summary
    logging.info(f"\n{'='*60}")
    logging.info("BATCH SUMMARY")
    logging.info(f"{'='*60}")
    for r in results:
        status_emoji = "✅" if r['status'] == 'completed' else "❌"
        logging.info(f"{status_emoji} {Path(r['config']).stem}: {r['status']}")

    return results


def list_experiments(output_dir: str):
    """List all experiments in the database."""
    from yinsh_ml.experiments import ExperimentDB

    db = ExperimentDB(str(Path(output_dir) / "experiments.db"))
    experiments = db.list_experiments(limit=50)

    if not experiments:
        print("No experiments found.")
        return

    print(f"\n{'ID':<10} {'Name':<25} {'Status':<12} {'ELO':<8} {'Created':<20}")
    print("-" * 80)

    for exp in experiments:
        created = exp.created_at[:19] if exp.created_at else ''
        print(f"{exp.experiment_id:<10} {exp.name[:24]:<25} {exp.status:<12} {exp.final_elo:<8.0f} {created:<20}")


def compare_experiments(experiment_ids: list, output_dir: str):
    """Compare multiple experiments."""
    from yinsh_ml.experiments import ExperimentDB

    db = ExperimentDB(str(Path(output_dir) / "experiments.db"))
    comparison = db.compare_experiments(experiment_ids)

    if not comparison:
        print("No experiments found to compare.")
        return

    print("\n" + "="*60)
    print("EXPERIMENT COMPARISON")
    print("="*60)

    for exp in comparison['summary']['final_comparison']:
        print(f"\n{exp['name']} ({exp['id']}):")
        print(f"  Final ELO: {exp['final_elo']:.0f} (best: {exp['best_elo']:.0f})")
        print(f"  Policy Loss: {exp['final_policy_loss']:.4f}")
        print(f"  Value Loss: {exp['final_value_loss']:.4f}")
        print(f"  Promoted: {exp['promoted']} | Rejected: {exp['rejected']}")
        print(f"  Runtime: {exp['runtime_hours']:.1f} hours")


def main():
    parser = argparse.ArgumentParser(
        description='Run YinshML training experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run an experiment')
    run_parser.add_argument('config', nargs='?', help='Path to experiment config YAML')
    run_parser.add_argument('--output-dir', '-o', default='experiments', help='Output directory')
    run_parser.add_argument('--resume', '-r', help='Experiment ID to resume')
    run_parser.add_argument('--iterations', '-i', type=int, help='Override number of iterations')
    run_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    run_parser.add_argument('--batch', '-b', nargs='+', help='Run multiple configs (glob patterns supported)')

    # List command
    list_parser = subparsers.add_parser('list', help='List experiments')
    list_parser.add_argument('--output-dir', '-o', default='experiments', help='Output directory')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare experiments')
    compare_parser.add_argument('experiment_ids', nargs='+', help='Experiment IDs to compare')
    compare_parser.add_argument('--output-dir', '-o', default='experiments', help='Output directory')

    args = parser.parse_args()

    # Default to run if no command specified but config given
    if args.command is None:
        # Check if first arg looks like a config file
        if len(sys.argv) > 1 and sys.argv[1].endswith('.yaml'):
            args.command = 'run'
            args.config = sys.argv[1]
            args.output_dir = 'experiments'
            args.resume = None
            args.iterations = None
            args.verbose = False
            args.batch = None
        else:
            parser.print_help()
            sys.exit(1)

    if args.command == 'run':
        # Setup logging
        log_file = Path(args.output_dir) / 'logs' / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        setup_logging(args.verbose, str(log_file))

        if args.batch:
            results = run_batch(args.batch, args.output_dir, args.iterations)
        elif args.config:
            result = run_single_experiment(args.config, args.output_dir, args.iterations, args.resume)
            print(f"\nExperiment {result['experiment_id']} finished: {result['final_status']}")
        elif args.resume:
            # Resume without config
            from yinsh_ml.experiments.experiment_runner import ExperimentRunner
            from yinsh_ml.experiments import ExperimentDB, ExperimentConfig

            db = ExperimentDB(str(Path(args.output_dir) / "experiments.db"))
            record = db.get_experiment(args.resume)
            if not record:
                print(f"Experiment {args.resume} not found")
                sys.exit(1)

            import json
            config = ExperimentConfig.from_dict(json.loads(record.config_json))
            runner = ExperimentRunner(config, args.output_dir, args.resume)
            result = runner.run()
            print(f"\nExperiment {result['experiment_id']} finished: {result['final_status']}")
        else:
            print("Please specify a config file or --resume")
            run_parser.print_help()
            sys.exit(1)

    elif args.command == 'list':
        list_experiments(args.output_dir)

    elif args.command == 'compare':
        compare_experiments(args.experiment_ids, args.output_dir)


if __name__ == '__main__':
    main()
