import argparse
import logging
import os
import sys
import time
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description='Run AlphaZero-style training for YinshML')
    parser.add_argument('-c', '--config', type=str, default=str(ROOT / 'configs' / 'training.yaml'), help='Path to YAML config')
    parser.add_argument('-n', '--iterations', type=int, default=None, help='Override num_iterations')
    parser.add_argument('--save-dir', type=str, default=None, help='Override save_dir')
    parser.add_argument('--export-every', type=int, default=None, help='Override export cadence')
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    # Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('run_training')

    # Resolve settings
    device = select_device(cfg.get('device', 'auto'))
    num_iterations = args.iterations or int(cfg.get('num_iterations', 50))

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

    # Build mode_settings for supervisor
    mode_settings = {
        # Self-play / MCTS
        'evaluation_mode': sp.get('evaluation_mode', 'hybrid'),  # NEW: Default to hybrid mode
        'heuristic_weight': float(sp.get('heuristic_weight', 0.7)),  # NEW: Default 70% heuristic weight
        'late_simulations': sp.get('late_simulations'),
        'simulation_switch_ply': sp.get('simulation_switch_ply', 20),
        'c_puct': float(sp.get('c_puct', 1.0)),
        'dirichlet_alpha': float(sp.get('dirichlet_alpha', 0.3)),
        'value_weight': float(sp.get('value_weight', 1.0)),
        'max_depth': int(sp.get('max_depth', 300)),
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
        # Arena / gating
        'promotion_threshold': float(arena_cfg.get('promotion_threshold', 0.55)),
    }

    # Instantiate network and supervisor
    network = NetworkWrapper(device=device)
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

    start_time = time.time()
    for it in range(num_iterations):
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


