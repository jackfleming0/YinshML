"""Main training script for YINSH ML model."""

import os
import sys
import time
import argparse
from pathlib import Path
import logging
import torch
from typing import Literal

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.network.model import YinshNetwork
from yinsh_ml.training.trainer import YinshTrainer
from yinsh_ml.training.supervisor import TrainingSupervisor
from yinsh_ml.utils.visualization import TrainingVisualizer

logging.getLogger('coremltools').setLevel(logging.ERROR)


def setup_logging(log_dir: str, mode: str, debug: bool = False):
    """Setup logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)

    # Create a timestamp-based log file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f'training_{mode}_{timestamp}.log'

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train YINSH ML model')

    # Basic parameters
    parser.add_argument('--output-dir', type=str, default='models',
                      help='Directory to save models and logs')
    parser.add_argument('--mode', type=str, choices=['tiny','quick', 'dev', 'dev2', 'full'],
                      default='dev',
                      help='Training mode (quick/dev/full)')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume training from')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else (
            'mps' if torch.backends.mps.is_available() else 'cpu'
        ),
        help='Device to train on (cuda/mps/cpu)'
    )
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')

    # Advanced parameters (usually don't need to change these)
    parser.add_argument('--batch-size', type=int, default=256,
                      help='Training batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of worker processes for self-play')
    parser.add_argument('--export-every', type=int, default=5,
                      help='Export CoreML model every N iterations')

    return parser.parse_args()

def get_mode_settings(mode: Literal['tiny','quick', 'dev', 'full']) -> dict:
    """Get training settings based on mode."""
    settings = {
        'tiny': {  # For rapid debugging/testing
            'num_iterations': 2, #number of "study sessions"
            'games_per_iteration': 3,  # games played per study session
            'epochs_per_iteration': 1,  # number of teams each game is reviewed
            'mcts_simulations': 25,  # number of branches that the student considers in depth
            'initial_temp': 1.0,
            'final_temp': 0.2,
            'c_puct': 1.0,
            'max_depth': 20,
            'l2_reg': 0.0,
            'export_every': 1 #how frequently we save and export
        },
        'quick': {  # For quick testing
            'num_iterations': 2,
            'games_per_iteration': 10,
            'epochs_per_iteration': 2,
            'mcts_simulations': 25,  # Increased from 10
            'initial_temp': 1.0,
            'final_temp': 0.2,
            'c_puct': 1.0,
            'max_depth': 20,
            'l2_reg': 0.0,
            'export_every': 1
        },
        'dev': {   # For development
            'num_iterations': 10,
            'games_per_iteration': 50,
            'epochs_per_iteration': 5,
            'mcts_simulations': 100,  # Increased from 50
            'initial_temp': 1.0,
            'final_temp': 0.2,
            'c_puct': 1.0,
            'max_depth': 20,
            'l2_reg': 0.0,
            'export_every': 5
        },
        'dev2': {  # For further dev testing, emphasis on MCTS.
            'num_iterations': 10,
            'games_per_iteration': 50,
            'epochs_per_iteration': 8,
            'mcts_simulations': 300,  # Big bump
            'initial_temp': 2.0,        # Start with more exploration
            'final_temp': 0.5,         # Keep some exploration even late
            'c_puct': 2.0,            # Increase exploration in MCTS
            'max_depth': 15,          # Add depth limit
            'l2_reg': 0.0001,         # Add regularization
            'export_every': 5
        },
        'full': {  # For full training
            'num_iterations': 100,
            'games_per_iteration': 200,
            'epochs_per_iteration': 10,
            'mcts_simulations': 400,  # Increased from 200
            'initial_temp': 1.0,
            'final_temp': 0.2,
            'c_puct': 1.0,
            'max_depth': 20,
            'l2_reg': 0.0,
            'export_every': 10
        }
    }
    return settings[mode]

def main():
    """
    The main function orchestrates the training process of a network model.
    It performs argument parsing, sets up directories and logging, initializes network and training components, and runs the main training loop.
    It handles checkpoints, model exporting, and interrupt scenarios.

    Args:
        None

    Raises:
        Exception: If there is any error during the training process.
        KeyboardInterrupt: If the training is manually interrupted by the user.
    """
    args = parse_args()
    settings = get_mode_settings(args.mode)

    # Remove mcts_simulations from settings since it's passed explicitly
    mcts_simulations = settings.pop('mcts_simulations')  # Remove and store the value


    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    model_dir = output_dir / f'training_{args.mode}'
    model_dir.mkdir(exist_ok=True)
    log_dir = model_dir / 'logs'
    log_dir.mkdir(exist_ok=True)

    # Setup logging
    setup_logging(log_dir, args.mode, args.debug)
    logger = logging.getLogger('train')
    logger.info(f'Starting training with args: {args}')
    logger.info(f'Using settings for {args.mode} mode: {settings}')
    logger.debug(f'Using device: {args.device}')
    logger.debug(f'Output directory structure created at {output_dir}')

    try:
        # Initialize network and training components
        network = NetworkWrapper(device=args.device)
        if args.resume:
            logger.info(f'Resuming from checkpoint: {args.resume}')
            logger.debug('Loading model weights and optimizer state')
            network.load_model(args.resume)

        supervisor = TrainingSupervisor(
            network=network,
            save_dir=str(model_dir),
            num_workers=args.num_workers,
            mcts_simulations=mcts_simulations,
            mode=args.mode,  # Pass the mode
            device=args.device,  # Pass the device
            **settings
        )
        logger.debug(f'Training supervisor initialized with {args.num_workers} workers')

        # Main training loop
        logger.info('Starting training loop')
        start_time = time.time()

        for iteration in range(settings['num_iterations']):
            iteration_start = time.time()
            logger.info(f'Starting iteration {iteration + 1}/{settings["num_iterations"]}')
            if torch.cuda.is_available():
                memory_usage = torch.cuda.memory_allocated()
            elif torch.backends.mps.is_available():
                memory_usage = torch.mps.current_allocated_memory()
            else:
                memory_usage = "N/A"
            logger.debug(f'Memory usage before iteration: {memory_usage}')
            # Training iteration
            supervisor.train_iteration(
                num_games=settings['games_per_iteration'],
                epochs=settings['epochs_per_iteration']
            )

            # Export CoreML model periodically
            if (iteration + 1) % settings['export_every'] == 0:
                coreml_path = model_dir / f'yinsh_model_iteration_{iteration+1}.mlpackage'
                try:
                    logger.debug('Starting CoreML model export')
                    network.export_to_coreml(str(coreml_path))
                    logger.info(f'Exported CoreML model to {coreml_path}')
                except Exception as e:
                    logger.error(f'Failed to export CoreML model: {e}')
                    logger.debug(f'Export error details:', exc_info=True)

            # Save checkpoint
            checkpoint_path = model_dir / f'checkpoint_iteration_{iteration+1}.pt'
            logger.debug('Saving model checkpoint')
            network.save_model(str(checkpoint_path))

            # Log timing information
            iteration_time = time.time() - iteration_start
            elapsed_time = time.time() - start_time
            logger.info(f'Iteration {iteration + 1} completed in {iteration_time:.1f}s')
            logger.info(f'Total training time so far: {elapsed_time/3600:.1f}h')
            logger.info(f'Saved checkpoint to {checkpoint_path}')
            if torch.cuda.is_available():
                memory_usage = torch.cuda.memory_allocated()
            elif torch.backends.mps.is_available():
                memory_usage = torch.mps.current_allocated_memory()
            else:
                memory_usage = "N/A"
            logger.debug(f'Memory usage after iteration: {memory_usage}')

    except KeyboardInterrupt:
        logger.info('Training interrupted by user')
        elapsed_time = time.time() - start_time
        logger.info(f'Training ran for {elapsed_time/3600:.1f}h before interruption')

        # Save interrupted model
        interrupt_path = model_dir / f'yinsh_model_interrupted_{args.mode}.pt'
        logger.debug('Saving interrupted model checkpoint')
        network.save_model(str(interrupt_path))
        logger.info(f'Saved interrupted model to {interrupt_path}')
    except Exception as e:
        logger.exception(f'Training failed with error: {e}')
    finally:
        logger.info('Training completed')

if __name__ == '__main__':
    main()