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


def _extract_model_state(checkpoint_obj, checkpoint_path: Path):
    """Return the raw model state_dict from a checkpoint payload.

    Supports both supervisor checkpoints (raw state_dict) and trainer-style
    payloads with a ``model_state_dict`` key. Used by both the resume path
    (which also restores optimizer/scheduler state) and the init-checkpoint
    path (weights only).
    """
    if isinstance(checkpoint_obj, dict) and "model_state_dict" in checkpoint_obj and isinstance(
        checkpoint_obj["model_state_dict"], dict
    ):
        return checkpoint_obj["model_state_dict"]
    if isinstance(checkpoint_obj, dict):
        return checkpoint_obj
    raise RuntimeError(
        f"Unsupported checkpoint object type at {checkpoint_path}: {type(checkpoint_obj)}"
    )


def _load_resume_checkpoint(network: NetworkWrapper, supervisor: TrainingSupervisor, checkpoint_path: Path, device: str, logger: logging.Logger) -> None:
    import torch

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint_obj = torch.load(checkpoint_path, map_location=device)
    checkpoint_dict = checkpoint_obj if isinstance(checkpoint_obj, dict) else {}
    model_state = _extract_model_state(checkpoint_obj, checkpoint_path)

    _load_model_state_compatible(network.network, model_state, logger)
    _restore_optimizer_state(supervisor, checkpoint_dict, logger)
    logger.info("Checkpoint loaded successfully")


def _load_init_checkpoint(network: NetworkWrapper, checkpoint_path: Path, device: str, logger: logging.Logger) -> None:
    """Warm-start a fresh run from an existing checkpoint's weights only.

    Unlike ``_load_resume_checkpoint``, this skips optimizer and scheduler
    state — the run starts at iteration 0 in a new run directory with fresh
    Adam moments / SGD momentum and a fresh LR schedule. Use when you want
    the network to inherit a learned prior (e.g. from supervised pretraining
    or a prior self-play run) without inheriting its optimizer history.
    """
    import torch

    logger.info(f"Warm-starting from init checkpoint: {checkpoint_path}")
    checkpoint_obj = torch.load(checkpoint_path, map_location=device)
    model_state = _extract_model_state(checkpoint_obj, checkpoint_path)
    _load_model_state_compatible(network.network, model_state, logger)
    logger.info("Init checkpoint weights loaded (optimizer / scheduler state skipped)")


def main() -> None:
    parser = argparse.ArgumentParser(description='Run AlphaZero-style training for YinshML')
    parser.add_argument('-c', '--config', type=str, default=str(ROOT / 'configs' / 'training.yaml'), help='Path to YAML config')
    parser.add_argument('-n', '--iterations', type=int, default=None, help='Override num_iterations')
    parser.add_argument('--save-dir', type=str, default=None, help='Override save_dir')
    parser.add_argument('--export-every', type=int, default=None, help='Override export cadence')
    parser.add_argument('--resume', type=str, default=None, help='Resume from existing run directory (e.g., runs/20260216_094801)')
    parser.add_argument(
        '--init-checkpoint',
        type=str,
        default=None,
        help=(
            'Warm-start a fresh run from an existing checkpoint. Unlike --resume, '
            'this creates a new timestamped run directory, starts at iteration 0, '
            'and loads ONLY model weights (optimizer / scheduler state are reset). '
            'Use for warm-starting from a supervised-pretrained or earlier-run '
            'checkpoint. Mutually exclusive with --resume.'
        ),
    )
    args = parser.parse_args()

    if args.resume and args.init_checkpoint:
        parser.error("--resume and --init-checkpoint are mutually exclusive")

    cfg = load_config(Path(args.config))

    # Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('run_training')

    # Resolve settings
    device = select_device(cfg.get('device', 'auto'))
    num_iterations = args.iterations or int(cfg.get('num_iterations', 50))

    # Handle resume vs init-checkpoint vs fresh run.
    # `start_iteration` and `checkpoint_to_load` flow into the loader below;
    # `init_checkpoint_to_load` (if set) goes through the weights-only path
    # so optimizer + iteration counter reset to fresh.
    start_iteration = 0
    checkpoint_to_load = None
    init_checkpoint_to_load: Optional[Path] = None

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

        if args.init_checkpoint:
            init_path = Path(args.init_checkpoint)
            if not init_path.exists():
                logger.error(f"--init-checkpoint path does not exist: {init_path}")
                sys.exit(1)
            init_checkpoint_to_load = init_path
            logger.info(f"Warm-starting new run {run_dir} from {init_path}")

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
    network_cfg = cfg.get('network', {})

    use_enhanced_encoding = encoding_cfg.get('type', 'basic') == 'enhanced'
    enable_augmentation = augmentation_cfg.get('enabled', False) or trainer_cfg.get('enable_augmentation', False)
    max_augmentations = augmentation_cfg.get('max_augmentations', trainer_cfg.get('max_augmentations', 12))
    # Branch D.1: value_head_type='gap' swaps the ~4M-param spatial-flatten
    # value head for a ~17K-param GAP head. None = auto-detect from
    # --init-checkpoint, else fall back to wrapper default ('spatial').
    value_head_type = network_cfg.get('value_head_type', None)
    if value_head_type is not None and value_head_type not in ('spatial', 'gap', 'gap_v2'):
        parser.error(f"network.value_head_type must be 'spatial', 'gap', or 'gap_v2', got {value_head_type!r}")

    if use_enhanced_encoding:
        logger.info(f"Using ENHANCED encoding (15 channels)")
    else:
        logger.info(f"Using BASIC encoding (6 channels)")
    if value_head_type is not None:
        logger.info(f"Using value head: {value_head_type}")

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
        # Optional: path to a re-fit heuristic weights JSON (WeightManager
        # format). Omit / null => hardcoded default weights. Accepts either
        # `weight_config_file` or the more explicit `heuristic_weight_config_file`.
        'heuristic_weight_config_file': sp.get('heuristic_weight_config_file', sp.get('weight_config_file')),
        'num_workers': sp.get('num_workers', 'auto'),
        'late_simulations': sp.get('late_simulations'),
        'simulation_switch_ply': sp.get('simulation_switch_ply', 20),
        # Probabilistic fast-sim split (alphazero-general style). Default off.
        'fast_simulations': int(sp.get('fast_simulations', 0)),
        'fast_sim_prob': float(sp.get('fast_sim_prob', 0.0)),
        'c_puct': float(sp.get('c_puct', 1.0)),
        'dirichlet_alpha': float(sp.get('dirichlet_alpha', 0.3)),
        'value_weight': float(sp.get('value_weight', 1.0)),
        'max_depth': int(sp.get('max_depth', 300)),
        'use_batched_mcts': bool(sp.get('use_batched_mcts', True)),
        'mcts_batch_size': int(sp.get('mcts_batch_size', 32)),
        # PR #12 Phase 2: shared BatchedEvaluator across N MCTS threads.
        # Default off — explicit opt-in per config (see cloud_run_v1.yaml).
        'use_shared_evaluator': bool(sp.get('use_shared_evaluator', False)),
        # E20 throughput: process-based inference server. Workers stay
        # separate processes (no GIL) and route leaf-eval batches to one
        # GPU-resident server that coalesces across all workers. Default off;
        # mutually exclusive with use_shared_evaluator. Needs num_workers > 0.
        'use_inference_server': bool(sp.get('use_inference_server', False)),
        'inference_server_max_wait_ms': float(sp.get('inference_server_max_wait_ms', 1.0)),
        # Opt-in to the C++ bitboard engine (yinsh_ml/game_cpp). Default off
        # until each training config has been A/B'd against the Python
        # engine. When true, the GameStatePool is bypassed because the
        # C++ State.clone() is faster than any pool reuse mechanism.
        'use_cpp_engine': bool(sp.get('use_cpp_engine', False)),
        # E22 cross-teacher: path to a FIXED opponent model. When set, self-play
        # is learner-vs-opponent (color-balanced) and only learner positions
        # train. None / unset = ordinary mirror self-play.
        'opponent_model_path': sp.get('opponent_model_path', None),
        'enable_subtree_reuse': bool(sp.get('enable_subtree_reuse', True)),
        'fpu_reduction': float(sp.get('fpu_reduction', 0.25)),
        'epsilon_mix_start': float(sp.get('epsilon_mix_start', 0.25)),
        'epsilon_mix_end': float(sp.get('epsilon_mix_end', 0.0)),
        'epsilon_mix_taper_moves': int(sp.get('epsilon_mix_taper_moves', 20)),
        # Root policy temperature: reshape root child priors before noise.
        # 1.0 = no-op; >1 flattens (more exploration); <1 sharpens.
        'root_policy_temp': float(sp.get('root_policy_temp', 1.0)),
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
        # EMA eval target. `ema_decay=None` disables the shadow entirely; set it
        # to e.g. 0.999 to have the trainer track a smoothed copy that the
        # tournament plays with instead of the single-step-noisy live weights.
        'ema_decay': (float(trainer_cfg['ema_decay'])
                      if trainer_cfg.get('ema_decay') is not None else None),
        'use_ema_for_eval': bool(trainer_cfg.get('use_ema_for_eval', True)),
        # Gaussian soft value targets. σ in class-widths; 0 = hard one-hot CE.
        'soft_value_target_sigma': float(trainer_cfg.get('soft_value_target_sigma', 0.5)),
        # LR schedule: 'cosine' (default) or 'step' for the legacy StepLR(step_size=10, gamma=0.9).
        'lr_schedule': trainer_cfg.get('lr_schedule', 'cosine'),
        'warmup_epochs': int(trainer_cfg.get('warmup_epochs', 0)),
        # bf16 autocast for training forward + loss. Auto-disabled on CPU.
        'enable_autocast': bool(trainer_cfg.get('enable_autocast', True)),
        # Search-consistency probe (Track B §5). Off by default. Reads from a
        # nested `trainer.search_consistency:` block when present so the YAML
        # stays organized; falls back to flat `trainer.search_consistency_*`
        # keys for ergonomics.
        'enable_search_consistency': bool(
            (trainer_cfg.get('search_consistency') or {}).get(
                'enabled', trainer_cfg.get('enable_search_consistency', False)
            )
        ),
        'search_consistency_weight': float(
            (trainer_cfg.get('search_consistency') or {}).get(
                'policy_weight', trainer_cfg.get('search_consistency_weight', 0.1)
            )
        ),
        'search_consistency_value_weight': float(
            (trainer_cfg.get('search_consistency') or {}).get(
                'value_weight', trainer_cfg.get('search_consistency_value_weight', 1.0)
            )
        ),
        'search_consistency_every_k_steps': int(
            (trainer_cfg.get('search_consistency') or {}).get(
                'every_k_steps', trainer_cfg.get('search_consistency_every_k_steps', 10)
            )
        ),
        'search_consistency_long_sims': int(
            (trainer_cfg.get('search_consistency') or {}).get(
                'long_sims', trainer_cfg.get('search_consistency_long_sims', 64)
            )
        ),
        'search_consistency_batch_size': int(
            (trainer_cfg.get('search_consistency') or {}).get(
                'batch_size', trainer_cfg.get('search_consistency_batch_size', 32)
            )
        ),
        'search_consistency_warmup_iters': int(
            (trainer_cfg.get('search_consistency') or {}).get(
                'warmup_iters', trainer_cfg.get('search_consistency_warmup_iters', 3)
            )
        ),
        # E2: placement-only value distillation (grounds the value head where it's
        # blind). Pair with policy_weight: 0.0 for value-only.
        'search_consistency_placement_only': bool(
            (trainer_cfg.get('search_consistency') or {}).get(
                'placement_only', trainer_cfg.get('search_consistency_placement_only', False)
            )
        ),
        # E16 symmetric-weight regularizer. Off by default. Reads from a nested
        # `trainer.symmetric_reg:` block when present, with flat-key fallback.
        # value_weight default 20.0 is measured (investigate_e16_value_weight.py).
        'enable_symmetric_reg': bool(
            (trainer_cfg.get('symmetric_reg') or {}).get(
                'enabled', trainer_cfg.get('enable_symmetric_reg', False)
            )
        ),
        'symmetric_reg_weight': float(
            (trainer_cfg.get('symmetric_reg') or {}).get(
                'weight', trainer_cfg.get('symmetric_reg_weight', 0.1)
            )
        ),
        'symmetric_reg_value_weight': float(
            (trainer_cfg.get('symmetric_reg') or {}).get(
                'value_weight', trainer_cfg.get('symmetric_reg_value_weight', 20.0)
            )
        ),
        'symmetric_reg_every_k_steps': int(
            (trainer_cfg.get('symmetric_reg') or {}).get(
                'every_k_steps', trainer_cfg.get('symmetric_reg_every_k_steps', 10)
            )
        ),
        # Cosine horizon = full training run in epochs. Scaled below when we
        # know num_iterations × epochs_per_iteration.
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
        # Deterministic tournament seed. Leave unset (or null) for stochastic play.
        'eval_seed': arena_cfg.get('eval_seed', None),
        # Gate revert (RiverNewbury-style). Default off preserves the
        # AlphaZero-continuous-training behavior. Recommended on for
        # warm-start runs from a strong supervised checkpoint.
        'revert_self_play_on_gate_failure': bool(arena_cfg.get('revert_self_play_on_gate_failure', False)),
        'reset_optimizer_on_revert': bool(arena_cfg.get('reset_optimizer_on_revert', True)),
        # Anchor eval (CLOUD_TRAINING_PLAN §1.3). Reads from a nested
        # `anchor:` block in the YAML for organization, with sensible
        # defaults for any run that doesn't specify one.
        'anchor_enabled': bool((cfg.get('anchor') or {}).get('enabled', True)),
        'anchor_num_games': int((cfg.get('anchor') or {}).get('num_games', 40)),
        'anchor_depth': int((cfg.get('anchor') or {}).get('depth', 3)),
        'anchor_seed': int((cfg.get('anchor') or {}).get('seed', 1337)),
        'anchor_max_moves_per_game': int((cfg.get('anchor') or {}).get('max_moves_per_game', 200)),
        'anchor_skip_first_n_iterations': int((cfg.get('anchor') or {}).get('skip_first_n_iterations', 1)),
        # MCTS-vs-anchor (primary strength metric). Off by default to keep
        # legacy configs unchanged; raw-policy anchor still runs and stays
        # the diagnostic. When on, the candidate also plays the anchor with
        # pure-neural MCTS (`mcts_simulations` per move, subtree-reuse on,
        # root noise off) so anchor numbers reflect deployment-style play.
        'anchor_mcts_enabled': bool((cfg.get('anchor') or {}).get('mcts_enabled', False)),
        'anchor_mcts_simulations': int((cfg.get('anchor') or {}).get('mcts_simulations', 64)),
        # Checkpoint retention: 0 = keep all, N>0 = keep top-N by Elo, N<0 =
        # legacy "delete rejected immediately." Default 5 prunes most of a
        # long run's history; for runs where you want to head-to-head every
        # iteration later, set to 0 in the YAML.
        'checkpoint_retention_count': int(trainer_cfg.get('checkpoint_retention_count', 5)),
    }

    games_per_iteration = int(sp.get('games_per_iteration', 50))
    epochs_per_iteration = int(trainer_cfg.get('epochs_per_iteration', 40))  # INCREASED: from 4 to 40 for better training
    # Compute the cosine horizon from the outer training loop's full epoch count
    # so the LR curve is scaled to the run, not to a single iteration.
    mode_settings['total_epochs'] = num_iterations * epochs_per_iteration
    # Tier 3 #6: Supervisor needs total_iterations to compute iteration_progress
    # for iteration-aware Dirichlet noise tapering. Stored in mode_settings so
    # the runner stays the only seat that knows the loop bounds.
    mode_settings['total_iterations'] = num_iterations

    # Instantiate network and supervisor. When warm-starting via
    # --init-checkpoint, pass model_path so NetworkWrapper auto-detects
    # num_channels / num_blocks / input_channels from the state_dict
    # (e.g. for a 256x18 supervised init checkpoint). Without this, the
    # network gets built at the default 256x12 and load_model rejects
    # the larger checkpoint.
    init_path_for_construct = init_checkpoint_to_load if init_checkpoint_to_load is not None else None
    network = NetworkWrapper(
        model_path=str(init_path_for_construct) if init_path_for_construct else None,
        device=device,
        use_enhanced_encoding=use_enhanced_encoding,
        value_head_type=value_head_type,
    )
    # Attach difficulty presets for export metadata
    network.difficulty_presets = cfg.get('difficulty_presets', {})
    supervisor = TrainingSupervisor(
        network=network,
        save_dir=str(run_dir),
        device=device,
        tournament_games=int(arena_cfg.get('games_per_match', 200)),
        mcts_simulations=int(sp.get('num_simulations', 96)),
        mode_settings=mode_settings,
        full_config=cfg,
    )

    # Load checkpoint if resuming
    if checkpoint_to_load:
        _load_resume_checkpoint(network, supervisor, checkpoint_to_load, device, logger)
        # Sync supervisor's internal iteration counter so the next call to
        # train_iteration runs as `start_iteration` (not overwriting iteration_0/).
        # Supervisor's _load_best_model_state already restores this from
        # best_model_state.json in the common case, but we set it explicitly
        # to guarantee correctness when that file is missing or stale.
        try:
            supervisor.set_resume_iteration(start_iteration)
        except Exception as e:
            logger.warning(f"Failed to sync supervisor iteration counter: {e}")
    elif init_checkpoint_to_load is not None:
        # Warm-start: weights only. Iteration counter stays at 0 (the new run
        # starts fresh), and supervisor.set_resume_iteration is intentionally
        # NOT called so iteration_0 / best_model bookkeeping behaves as a new
        # run that just happens to inherit a learned prior.
        _load_init_checkpoint(network, init_checkpoint_to_load, device, logger)

    start_time = time.time()
    last_completed_iteration = start_iteration  # tracks successful iterations for manifest_final
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
        last_completed_iteration = it + 1  # 1-based count of completed iterations

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

    # Training loop completed normally — write the final manifest.
    # final_anchor_win_rate comes from the last iteration's anchor eval
    # (CLOUD_TRAINING_PLAN §1.3). None if anchor eval was disabled/skipped.
    try:
        supervisor.finalize_manifest(
            iterations_completed=last_completed_iteration,
            final_anchor_win_rate=getattr(supervisor, '_latest_anchor_win_rate', None),
        )
    except Exception as e:
        logger.warning(f'finalize_manifest failed: {e}')


if __name__ == '__main__':
    main()
