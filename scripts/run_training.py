import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import yaml

# Ensure project root on path when executed directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.training.supervisor import TrainingSupervisor
from yinsh_ml.analysis.auto_tuner import AutoTuner


def load_config(cfg_path: Path) -> dict:
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f) or {}


def find_latest_checkpoint(run_dir: Path) -> Tuple[Optional[Path], int]:
    """Find the latest checkpoint in a run directory.

    Returns:
        Tuple of (checkpoint_path, iteration_number) or (None, 0) if not found.
    """
    iteration_dirs = list(run_dir.glob('iteration_*'))
    if not iteration_dirs:
        return None, 0

    # Extract iteration numbers and sort
    iterations = []
    for d in iteration_dirs:
        match = re.search(r'iteration_(\d+)', d.name)
        if match:
            iter_num = int(match.group(1))
            checkpoint = d / f'checkpoint_iteration_{iter_num}.pt'
            if checkpoint.exists():
                iterations.append((iter_num, checkpoint))

    if not iterations:
        return None, 0

    # Return the highest iteration
    iterations.sort(key=lambda x: x[0], reverse=True)
    return iterations[0][1], iterations[0][0]


def select_device(device_cfg: str) -> str:
    if device_cfg and device_cfg.lower() != 'auto':
        return device_cfg
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if torch.backends.mps.is_available():
            return 'mps'
    except Exception:
        pass
    return 'cpu'


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_model_state_compatible(model, state_dict: dict, logger: logging.Logger) -> None:
    """Load only shape-compatible keys so resume survives minor architecture drift."""
    model_state = model.state_dict()
    compatible_state = {}
    for key, param in state_dict.items():
        if key in model_state and hasattr(param, "shape") and param.shape == model_state[key].shape:
            compatible_state[key] = param

    if not compatible_state:
        raise RuntimeError("Checkpoint does not contain any compatible model weights")

    missing = len(model_state) - len(compatible_state)
    logger.info(
        f"Loaded {len(compatible_state)}/{len(model_state)} compatible tensor(s) from checkpoint "
        f"(missing or shape-mismatched: {missing})"
    )
    model.load_state_dict(compatible_state, strict=False)


def _restore_optimizer_state(supervisor, checkpoint: dict, logger: logging.Logger) -> None:
    trainer = getattr(supervisor, "trainer", None)
    if trainer is None:
        return

    try:
        if "optimizer_state_dict" in checkpoint and hasattr(trainer, "policy_optimizer"):
            trainer.policy_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Restored policy optimizer state (optimizer_state_dict)")
    except Exception as exc:
        logger.warning(f"Could not restore optimizer_state_dict: {exc}")

    try:
        if "policy_optimizer_state_dict" in checkpoint and hasattr(trainer, "policy_optimizer"):
            trainer.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
            logger.info("Restored policy optimizer state (policy_optimizer_state_dict)")
    except Exception as exc:
        logger.warning(f"Could not restore policy_optimizer_state_dict: {exc}")

    try:
        if "value_optimizer_state_dict" in checkpoint and hasattr(trainer, "value_optimizer"):
            trainer.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
            logger.info("Restored value optimizer state")
    except Exception as exc:
        logger.warning(f"Could not restore value_optimizer_state_dict: {exc}")

    try:
        if "policy_scheduler_state_dict" in checkpoint and hasattr(trainer, "policy_scheduler"):
            trainer.policy_scheduler.load_state_dict(checkpoint["policy_scheduler_state_dict"])
            logger.info("Restored policy scheduler state")
    except Exception as exc:
        logger.warning(f"Could not restore policy_scheduler_state_dict: {exc}")

    try:
        if "value_scheduler_state_dict" in checkpoint and hasattr(trainer, "value_scheduler"):
            trainer.value_scheduler.load_state_dict(checkpoint["value_scheduler_state_dict"])
            logger.info("Restored value scheduler state")
    except Exception as exc:
        logger.warning(f"Could not restore value_scheduler_state_dict: {exc}")


def _load_resume_checkpoint(network: NetworkWrapper, supervisor: TrainingSupervisor, checkpoint_path: Path, device: str, logger: logging.Logger) -> None:
    import torch

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint_obj = torch.load(checkpoint_path, map_location=device)

    checkpoint_dict = checkpoint_obj if isinstance(checkpoint_obj, dict) else {}
    if isinstance(checkpoint_obj, dict) and "model_state_dict" in checkpoint_obj and isinstance(
        checkpoint_obj["model_state_dict"], dict
    ):
        model_state = checkpoint_obj["model_state_dict"]
    elif isinstance(checkpoint_obj, dict):
        # Supervisor checkpoints are raw state_dict objects.
        model_state = checkpoint_obj
    else:
        raise RuntimeError(f"Unsupported checkpoint object type: {type(checkpoint_obj)}")

    _load_model_state_compatible(network.network, model_state, logger)
    _restore_optimizer_state(supervisor, checkpoint_dict, logger)
    logger.info("Checkpoint loaded successfully")


def main() -> None:
    parser = argparse.ArgumentParser(description='Run AlphaZero-style training for YinshML')
    parser.add_argument('-c', '--config', type=str, default=str(ROOT / 'configs' / 'training.yaml'), help='Path to YAML config')
    parser.add_argument('-n', '--iterations', type=int, default=None, help='Override num_iterations')
    parser.add_argument('--save-dir', type=str, default=None, help='Override save_dir')
    parser.add_argument('--export-every', type=int, default=None, help='Override export cadence')
    parser.add_argument('--resume', type=str, default=None, help='Resume from existing run directory (e.g., runs/20260216_094801)')
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    # Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('run_training')

    # Resolve settings
    device = select_device(cfg.get('device', 'auto'))
    num_iterations = args.iterations or int(cfg.get('num_iterations', 50))

    # Handle resume vs new run
    start_iteration = 0
    checkpoint_to_load = None

    if args.resume:
        run_dir = Path(args.resume)
        if not run_dir.exists():
            logger.error(f"Resume directory does not exist: {run_dir}")
            sys.exit(1)
        checkpoint_to_load, last_iteration = find_latest_checkpoint(run_dir)
        if checkpoint_to_load:
            start_iteration = last_iteration + 1
            logger.info(f"Resuming from {run_dir}")
            logger.info(f"  Found checkpoint: {checkpoint_to_load}")
            logger.info(f"  Last completed iteration: {last_iteration}")
            logger.info(f"  Will start from iteration: {start_iteration}")
        else:
            logger.warning(f"No checkpoints found in {run_dir}, starting from scratch")
    else:
        base_save_dir = Path(args.save_dir or cfg.get('save_dir', 'runs'))
        run_dir = ensure_dir(base_save_dir / time.strftime('%Y%m%d_%H%M%S'))

    # Export settings
    export_cfg = cfg.get('export', {})
    export_every = args.export_every or int(export_cfg.get('export_every', 5))
    coreml_dir = ensure_dir(run_dir / Path(export_cfg.get('coreml_dir', 'models/coreml')))

    # Self-play / trainer settings
    sp = cfg.get('self_play', {})
    trainer_cfg = cfg.get('trainer', {})
    arena_cfg = cfg.get('arena', {})
    phase_weights_cfg = cfg.get('phase_weights', {})

    # Encoding and augmentation settings (architectural improvements)
    encoding_cfg = cfg.get('encoding', {})
    augmentation_cfg = cfg.get('augmentation', {})

    use_enhanced_encoding = encoding_cfg.get('type', 'basic') == 'enhanced'
    enable_augmentation = augmentation_cfg.get('enabled', False) or trainer_cfg.get('enable_augmentation', False)
    max_augmentations = augmentation_cfg.get('max_augmentations', trainer_cfg.get('max_augmentations', 12))

    if use_enhanced_encoding:
        logger.info(f"Using ENHANCED encoding (15 channels)")
    else:
        logger.info(f"Using BASIC encoding (6 channels)")

    if enable_augmentation:
        logger.info(f"Augmentation ENABLED (max {max_augmentations}x expansion)")
    else:
        logger.info(f"Augmentation DISABLED")

    # Build mode_settings for supervisor
    mode_settings = {
        # Self-play / MCTS
        'evaluation_mode': sp.get('evaluation_mode', 'hybrid'),  # NEW: Default to hybrid mode
        'heuristic_weight': float(sp.get('heuristic_weight', 0.7)),  # NEW: Default 70% heuristic weight
        'heuristic_weight_start': float(sp.get('heuristic_weight_start', sp.get('heuristic_weight', 0.7))),
        'heuristic_weight_end': float(sp.get('heuristic_weight_end', sp.get('heuristic_weight', 0.7))),
        'heuristic_weight_anneal_iterations': int(sp.get('heuristic_weight_anneal_iterations', 0)),
        'num_workers': sp.get('num_workers', 'auto'),
        'late_simulations': sp.get('late_simulations'),
        'simulation_switch_ply': sp.get('simulation_switch_ply', 20),
        'c_puct': float(sp.get('c_puct', 1.0)),
        'dirichlet_alpha': float(sp.get('dirichlet_alpha', 0.3)),
        'value_weight': float(sp.get('value_weight', 1.0)),
        'max_depth': int(sp.get('max_depth', 300)),
        'use_batched_mcts': bool(sp.get('use_batched_mcts', True)),
        'mcts_batch_size': int(sp.get('mcts_batch_size', 32)),
        'initial_temp': float(sp.get('initial_temp', 1.0)),
        'final_temp': float(sp.get('final_temp', 0.1)),
        'annealing_steps': int(sp.get('annealing_steps', 30)),
        'temp_clamp_fraction': float(sp.get('temp_clamp_fraction', 0.6)),
        # Trainer
        'batch_size': int(trainer_cfg.get('batch_size', 256)),
        'l2_reg': float(trainer_cfg.get('l2_reg', 0.0)),
        'lr': float(trainer_cfg.get('lr', 1e-3)),
        'value_head_lr_factor': float(trainer_cfg.get('value_head_lr_factor', 5.0)),
        'value_loss_weights': tuple(trainer_cfg.get('value_loss_weights', [0.5, 0.5])),
        'batches_per_epoch': trainer_cfg.get('batches_per_epoch', 'auto'),
        'max_buffer_size': int(trainer_cfg.get('max_buffer_size', 10000)),
        'discrimination_weight': float(trainer_cfg.get('discrimination_weight', 0.5)),
        # Augmentation settings (Phase 2 architectural improvements)
        'enable_augmentation': enable_augmentation,
        'max_augmentations': int(max_augmentations),
        # Phase-weighted sampling (emphasize critical game phases)
        'phase_weights': {
            'RING_PLACEMENT': float(phase_weights_cfg.get('RING_PLACEMENT', 1.0)),
            'MAIN_GAME': float(phase_weights_cfg.get('MAIN_GAME', 1.0)),
            'RING_REMOVAL': float(phase_weights_cfg.get('RING_REMOVAL', 1.0)),
        },
        # Arena / gating
        'promotion_threshold': float(arena_cfg.get('promotion_threshold', 0.55)),
        'tournament_sliding_window': int(arena_cfg.get('tournament_sliding_window', 5)),
    }

    # Instantiate network and supervisor
    network = NetworkWrapper(device=device, use_enhanced_encoding=use_enhanced_encoding)
    # Attach difficulty presets for export metadata
    network.difficulty_presets = cfg.get('difficulty_presets', {})
    supervisor = TrainingSupervisor(
        network=network,
        save_dir=str(run_dir),
        device=device,
        tournament_games=int(arena_cfg.get('games_per_match', 200)),
        mcts_simulations=int(sp.get('num_simulations', 96)),
        mode_settings=mode_settings,
    )

    games_per_iteration = int(sp.get('games_per_iteration', 50))
    epochs_per_iteration = int(trainer_cfg.get('epochs_per_iteration', 40))  # INCREASED: from 4 to 40 for better training

    # Load checkpoint if resuming
    if checkpoint_to_load:
        _load_resume_checkpoint(network, supervisor, checkpoint_to_load, device, logger)

    start_time = time.time()
    for it in range(start_iteration, num_iterations):
        logger.info(f'Starting iteration {it + 1}/{num_iterations}')
        summary = supervisor.train_iteration(num_games=games_per_iteration, epochs=epochs_per_iteration)

        # Periodic CoreML export
        if export_every > 0 and ((it + 1) % export_every == 0):
            coreml_path = coreml_dir / f'yinsh_model_iteration_{it + 1}.mlpackage'
            try:
                logger.info('Exporting CoreML model...')
                network.export_to_coreml(str(coreml_path))
                logger.info(f'Exported CoreML model to {coreml_path}')
            except Exception as e:
                logger.error(f'CoreML export failed: {e}', exc_info=True)

        # Always save checkpoint for the iteration in supervisor; nothing else here

        elapsed_h = (time.time() - start_time) / 3600.0
        logger.info(f'Iteration {it + 1} complete. Elapsed: {elapsed_h:.2f}h')

        # After each iteration: snapshot current config and write tuner suggestions
        try:
            snapshot_dir = run_dir / 'config_snapshots'
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            # Save the current config snapshot
            with open(snapshot_dir / f'config_iter_{it + 1}.yaml', 'w') as f:
                yaml.safe_dump(cfg, f)
            # Run auto tuner
            tuner = AutoTuner(run_dir)
            suggestions = tuner.write_suggestions(snapshot_dir / f'config_iter_{it + 1}.yaml', run_dir / f'suggestions_iter_{it + 1}.yaml')
            logger.info(f'Wrote next-step suggestions to {run_dir / f"suggestions_iter_{it + 1}.yaml"}')

            # Append a simple feedback entry for the iteration
            feedback_path = run_dir / 'feedback.md'
            try:
                training_summary = summary.get('training', {}) if isinstance(summary, dict) else {}
                training_games = summary.get('training_games', {}) if isinstance(summary, dict) else {}
                eval_summary = summary.get('evaluation', {}) if isinstance(summary, dict) else {}
                with open(feedback_path, 'a') as fb:
                    fb.write(f"\n## Iteration {it + 1}\n")
                    fb.write(f"- policy_loss: {training_summary.get('policy_loss', 'n/a')}\n")
                    fb.write(f"- value_loss: {training_summary.get('value_loss', 'n/a')}\n")
                    fb.write(f"- value_accuracy: {training_summary.get('value_accuracy', 'n/a')}\n")
                    fb.write(f"- avg_game_length: {training_games.get('avg_game_length', 'n/a')}\n")
                    # Record key suggestion knobs
                    sp_sug = suggestions.get('self_play', {}) if isinstance(suggestions, dict) else {}
                    tr_sug = suggestions.get('trainer', {}) if isinstance(suggestions, dict) else {}
                    if sp_sug or tr_sug:
                        fb.write(f"- suggestions:self_play: { {k: sp_sug[k] for k in ['num_simulations','late_simulations','final_temp','games_per_iteration'] if k in sp_sug} }\n")
                        fb.write(f"- suggestions:trainer: { {k: tr_sug[k] for k in ['value_head_lr_factor'] if k in tr_sug} }\n")
            except Exception as e:
                logger.warning(f'Failed to append feedback entry: {e}')
        except Exception as e:
            logger.warning(f'Auto-tuning suggestions failed: {e}')


if __name__ == '__main__':
    main()
