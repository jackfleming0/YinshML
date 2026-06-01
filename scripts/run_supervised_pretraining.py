#!/usr/bin/env python3
"""Supervised pre-training from expert game data.

Produces a model checkpoint pre-trained on expert games that can then
feed into the self-play training loop as a warm start.

Usage:
    # Pre-train from converted game data (.npz) — eager load, needs RAM ≥ corpus
    python scripts/run_supervised_pretraining.py --data expert_games/training_data.npz

    # Pre-train from a directory of .npy files (memory-mapped, RAM-light)
    # See scripts/convert_npz_to_mmap_shards.py for the npz → .npy conversion.
    python scripts/run_supervised_pretraining.py --data-dir expert_games/yngine_volume_mmap/

    # Pre-train from raw JSON game files
    python scripts/run_supervised_pretraining.py --games-dir expert_games/json/ --min-rating 1500

    # With custom hyperparameters
    python scripts/run_supervised_pretraining.py --data data.npz --epochs 50 --batch-size 128 --lr 0.001
"""

import argparse
import logging
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.network.model import YinshNetwork
from yinsh_ml.data.converter import GameConverter
from yinsh_ml.utils.encoding import StateEncoder
from yinsh_ml.utils.enhanced_encoding import EnhancedStateEncoder
from yinsh_ml.training import symmetric_reg

logger = logging.getLogger(__name__)


class ExpertDataset(Dataset):
    """PyTorch Dataset for expert game training data.

    Accepts numpy arrays OR numpy memmaps (same indexing API). All
    dtype conversion is per-item so the same code handles eager npz
    loads (full arrays in RAM) and lazy memmap loads (paged from disk).

    Policy targets may be either:
      - (N, total_moves) float32 — soft/one-hot distribution
      - (N,) int{32,64}           — argmax move index (volume corpora)

    Downstream loss code branches on tensor dimensionality.
    """

    def __init__(self, states, policies, values):
        assert len(states) == len(policies) == len(values), (
            f"length mismatch: states={len(states)} policies={len(policies)} "
            f"values={len(values)}"
        )
        self.states = states
        self.policies = policies
        self.values = values

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        # np.ascontiguousarray triggers a copy out of any underlying mmap,
        # which keeps torch.from_numpy / pin_memory well-behaved across workers.
        state = torch.from_numpy(np.ascontiguousarray(self.states[idx])).float()
        policy_raw = self.policies[idx]
        if np.ndim(policy_raw) == 0:
            policy = torch.tensor(int(policy_raw), dtype=torch.long)
        else:
            policy = torch.from_numpy(np.ascontiguousarray(policy_raw)).float()
        value = torch.tensor(float(self.values[idx]), dtype=torch.float32)
        return state, policy, value


def load_data(args) -> tuple:
    """Load training data from one of: .npz file, .npy directory, raw games dir.

    Returns (states, policy_targets, values). Arrays may be backed by RAM
    (npz / games-dir paths) or by mmap (data-dir path). ExpertDataset is
    backing-agnostic.

    `policy_targets` is one-hot/soft (N, total_moves) float32 OR int{32,64}
    indices (N,) — schema determined by what keys/files are present.
    """
    if args.data_dir:
        data_dir = Path(args.data_dir)
        logger.info(f"Memory-mapping arrays from {data_dir}/")
        states = np.load(data_dir / 'states.npy', mmap_mode='r')
        values = np.load(data_dir / 'values.npy', mmap_mode='r')
        if (data_dir / 'policy_indices.npy').exists():
            policy_targets = np.load(data_dir / 'policy_indices.npy', mmap_mode='r')
            logger.info(
                f"Policy schema: integer indices "
                f"(shape={policy_targets.shape}, dtype={policy_targets.dtype}); "
                f"loss will use F.cross_entropy"
            )
        elif (data_dir / 'policies.npy').exists():
            policy_targets = np.load(data_dir / 'policies.npy', mmap_mode='r')
            logger.info(
                f"Policy schema: soft/one-hot distribution "
                f"(shape={policy_targets.shape}); loss will use soft-target NLL"
            )
        else:
            raise RuntimeError(
                f"{data_dir} must contain policy_indices.npy or policies.npy"
            )
    elif args.data:
        logger.info(f"Loading pre-converted data from {args.data}")
        data = np.load(args.data)
        states = data['states']
        values = data['values']
        if 'policy_indices' in data.files:
            policy_targets = data['policy_indices']
            logger.info(
                f"Policy schema: integer indices "
                f"(shape={policy_targets.shape}, dtype={policy_targets.dtype}); "
                f"loss will use F.cross_entropy"
            )
        elif 'policies' in data.files:
            policy_targets = data['policies']
            logger.info(
                f"Policy schema: soft/one-hot distribution "
                f"(shape={policy_targets.shape}); loss will use soft-target NLL"
            )
        else:
            raise RuntimeError(
                f"{args.data} must contain either 'policy_indices' or "
                f"'policies' key (found: {list(data.files)})"
            )
    elif args.games_dir:
        logger.info(f"Converting games from {args.games_dir}")
        encoder = EnhancedStateEncoder() if args.use_enhanced_encoding else StateEncoder()
        converter = GameConverter(encoder=encoder)
        pairs = converter.convert_directory(args.games_dir,
                                            min_rating=args.min_rating)
        if not pairs:
            raise RuntimeError(f"No valid training pairs from {args.games_dir}")

        states = np.array([p['state'] for p in pairs], dtype=np.float32)
        policy_targets = np.array([p['policy'] for p in pairs], dtype=np.float32)
        values = np.array([p['value'] for p in pairs], dtype=np.float32)

        # Save for future use
        save_path = Path(args.games_dir) / 'training_data.npz'
        np.savez_compressed(save_path, states=states, policies=policy_targets,
                           values=values)
        logger.info(f"Saved converted data to {save_path}")
    else:
        raise RuntimeError("Must specify --data or --games-dir")

    logger.info(f"Loaded {len(states)} positions "
                f"(states: {states.shape}, policy_targets: {policy_targets.shape})")
    return states, policy_targets, values


def create_model(args) -> tuple:
    """Create model and move to appropriate device."""
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    )

    input_channels = 15 if args.use_enhanced_encoding else 6
    model_kwargs = dict(
        num_channels=args.num_channels,
        num_blocks=args.num_blocks,
        input_channels=input_channels,
        value_head_type=args.value_head_type,
        value_mode=args.value_mode,
    )
    if args.value_mode == 'classification' and args.num_value_classes is not None:
        model_kwargs['num_value_classes'] = args.num_value_classes
    model = YinshNetwork(**model_kwargs).to(device)

    # Mutual-exclusion: --resume and --checkpoint do different things.
    if args.resume and args.checkpoint:
        raise ValueError(
            "--resume and --checkpoint are mutually exclusive. "
            "--checkpoint warm-starts model weights only (optimizer + epoch reset); "
            "--resume continues a prior run (restores model + optimizer + epoch from "
            "<output_dir>/last_resume_state.pt)."
        )

    resume_bundle = None
    if args.resume:
        resume_path = Path(args.output_dir) / 'last_resume_state.pt'
        if not resume_path.exists():
            raise FileNotFoundError(
                f"--resume specified but no state file at {resume_path}. "
                "Remove --resume to start fresh, or check --output-dir."
            )
        logger.info(f"Resuming from rich bundle: {resume_path}")
        resume_bundle = torch.load(resume_path, map_location=device,
                                   weights_only=False)
        model.load_state_dict(resume_bundle['model'])

    elif args.checkpoint:
        logger.info(f"Warm-starting from checkpoint: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device,
                                weights_only=True)
        model.load_state_dict(state_dict)

    return model, device, resume_bundle


def _value_targets_to_classes(values: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Map scalar targets in [-1, 1] to class indices in [0, num_classes-1].

    Matches the discretization used in yinsh_ml/training/trainer.py so the
    supervised and self-play value heads optimize the same objective.
    Expert outcomes {-1, 0, +1} → classes {0, mid, num_classes-1}.
    """
    normalized = (values + 1.0) / 2.0 * (num_classes - 1)
    return torch.round(normalized).long().clamp(0, num_classes - 1)


def train(model, device, train_loader, val_loader, args, resume_bundle=None):
    """Run supervised training loop.

    If resume_bundle is provided (from --resume), restores optimizer state and
    continues from saved_epoch + 1. The cosine scheduler is recreated fresh with
    T_max=args.epochs, then advanced saved_epoch steps — this lets a resume
    pass --epochs N+M to extend the original N-epoch run by M epochs cleanly.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    # LR schedule: cosine annealing. Built against the CURRENT args.epochs so an
    # extension call (--resume --epochs N+M) recomputes the curve over the new
    # total budget. We then advance the scheduler saved_epoch steps to land on
    # the LR appropriate for the new schedule at that point.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 100
    )

    start_epoch = 1
    best_val_loss = float('inf')

    if resume_bundle is not None:
        if 'optimizer' in resume_bundle:
            optimizer.load_state_dict(resume_bundle['optimizer'])
        saved_epoch = resume_bundle.get('epoch', 0)
        start_epoch = saved_epoch + 1
        best_val_loss = resume_bundle.get('best_val_loss', float('inf'))
        for _ in range(saved_epoch):
            scheduler.step()
        logger.info(
            f"Resumed at epoch {start_epoch}/{args.epochs} | "
            f"best_val_loss carry-over = {best_val_loss:.4f} | "
            f"LR = {scheduler.get_last_lr()[0]:.6f}"
        )
        if start_epoch > args.epochs:
            logger.warning(
                f"Resume state already past --epochs (saved_epoch={saved_epoch} "
                f">= args.epochs={args.epochs}). Nothing to do. Increase --epochs "
                f"to extend the run."
            )
            return Path(args.output_dir) / 'best_supervised.pt'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_value_classes = model.num_value_classes
    is_regression = (model.value_mode == 'regression')

    # E16 symmetric-weight regularizer (shared with self-play trainer.py). Built
    # once; applied every K steps. Off unless --enable-symmetric-reg.
    sym_reg_tensors = None
    sym_reg_encoder = None
    sym_reg_step = 0
    if args.enable_symmetric_reg:
        sym_reg_encoder = EnhancedStateEncoder() if args.use_enhanced_encoding else StateEncoder()
        sym_reg_tensors = symmetric_reg.build_reg_tensors(sym_reg_encoder, device)
        logger.info(
            f"E16 symmetric regularizer ON: weight={args.symmetric_reg_weight}, "
            f"value_weight={args.symmetric_reg_value_weight}, "
            f"every_k_steps={args.symmetric_reg_every_k_steps}"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        # --- Training ---
        model.train()
        train_policy_loss = 0.0
        train_value_loss = 0.0
        train_value_correct = 0
        train_correct = 0
        train_total = 0

        batch_count = 0
        epoch_start = time.time()
        for states, policies, values in train_loader:
            states = states.to(device)
            policies = policies.to(device)
            values = values.to(device)

            optimizer.zero_grad()

            pred_logits, value_pred = model(states)

            # Policy loss: cross-entropy against expert moves.
            # Branch on target schema: 1-D int targets → integer-target CE;
            # 2-D soft/one-hot targets → soft-target NLL.
            if policies.dim() == 1:
                expert_moves = policies.long()
                policy_loss = F.cross_entropy(pred_logits, expert_moves, label_smoothing=args.label_smoothing)
            else:
                log_probs = F.log_softmax(pred_logits, dim=1)
                policy_loss = -(policies * log_probs).sum(dim=1).mean()
                expert_moves = policies.argmax(dim=1)

            # Value loss: branch on value_mode.
            if is_regression:
                # Scalar tanh head against raw value target. value_pred is
                # already (B,) from YinshNetwork.forward in regression mode.
                value_loss = F.mse_loss(value_pred, values.float())
            else:
                # Cross-entropy on discretized outcome classes. Mirrors
                # yinsh_ml/training/trainer.py — same loss surface as self-play
                # so the warm-start checkpoint isn't washed out on iter 1.
                value_logits = model._value_logits  # populated in forward()
                target_class = _value_targets_to_classes(values, num_value_classes)
                value_loss = F.cross_entropy(value_logits, target_class)

            # Combined loss
            loss = policy_loss + args.value_weight * value_loss

            # E16: every K steps add the D2 symmetric-weight regularizer. Hard
            # expert targets give no MCTS visit support, so the valid-move mask is
            # decoded from the batch states (only on regularized steps). Reuses
            # this step's forward (pred_logits/value_pred) as the identity.
            if sym_reg_tensors is not None:
                sym_reg_step += 1
                if sym_reg_step % args.symmetric_reg_every_k_steps == 0:
                    sym_mask = symmetric_reg.valid_move_mask(sym_reg_encoder, states)
                    sym_loss, sym_diag = symmetric_reg.symmetric_reg_term(
                        model, states, pred_logits, value_pred, sym_mask,
                        sym_reg_tensors,
                        value_weight=args.symmetric_reg_value_weight,
                        weight=args.symmetric_reg_weight,
                    )
                    loss = loss + sym_loss
                    if sym_reg_step % (args.symmetric_reg_every_k_steps * 10) == 0:
                        logger.info(
                            f"E16 sym-reg: kl={sym_diag['sym_kl']:.5f} "
                            f"value_asym={sym_diag['sym_value_asym']:.5f}"
                        )
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_policy_loss += policy_loss.item() * len(states)
            train_value_loss += value_loss.item() * len(states)

            # Top-1 accuracy
            pred_moves = pred_logits.argmax(dim=1)
            train_correct += (pred_moves == expert_moves).sum().item()

            # Value accuracy: in classification mode = argmax class match;
            # in regression mode = sign accuracy (proxy — does the model
            # at least predict the right side of zero?). Classification's
            # argmax accuracy is the metric flagged as "argmax-VAcc plateau"
            # in the A4 hypothesis; sign accuracy is the regression analog.
            if is_regression:
                train_value_correct += (
                    torch.sign(value_pred) == torch.sign(values.float())
                ).sum().item()
            else:
                pred_class = model._value_logits.argmax(dim=1)
                target_class = _value_targets_to_classes(values, num_value_classes)
                train_value_correct += (pred_class == target_class).sum().item()
            train_total += len(states)

            batch_count += 1
            if batch_count % 50 == 0:
                elapsed = time.time() - epoch_start
                rate = batch_count / elapsed
                eta = (len(train_loader) - batch_count) / rate
                logger.info(
                    f"  Epoch {epoch} batch {batch_count}/{len(train_loader)} "
                    f"({rate:.1f} batch/s, eta {eta:.0f}s) "
                    f"loss={loss.item():.4f}"
                )

        scheduler.step()

        n_train = len(train_loader.dataset)
        avg_policy = train_policy_loss / n_train
        avg_value = train_value_loss / n_train
        accuracy = train_correct / train_total if train_total > 0 else 0
        value_accuracy = train_value_correct / train_total if train_total > 0 else 0

        # --- Validation ---
        val_policy, val_value, val_acc, val_value_acc = evaluate(
            model, device, val_loader, num_value_classes,
            label_smoothing=args.label_smoothing,
        )

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train: P={avg_policy:.4f} V={avg_value:.4f} "
            f"PAcc={accuracy:.3f} VAcc={value_accuracy:.3f} | "
            f"Val: P={val_policy:.4f} V={val_value:.4f} "
            f"PAcc={val_acc:.3f} VAcc={val_value_acc:.3f} | "
            f"LR={scheduler.get_last_lr()[0]:.6f}"
        )

        # Save best model
        val_loss = val_policy + args.value_weight * val_value
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = output_dir / 'best_supervised.pt'
            torch.save(model.state_dict(), save_path)
            logger.info(f"  New best model saved to {save_path}")

        # Save periodic checkpoints
        if epoch % args.save_every == 0:
            save_path = output_dir / f'supervised_epoch_{epoch}.pt'
            torch.save(model.state_dict(), save_path)

        # Rich resume bundle — overwritten every epoch. Includes optimizer state
        # and epoch counter so `--resume` continues without reset. Existing
        # state_dict-only checkpoints above stay untouched for downstream
        # compatibility (self-play loads them as plain state_dicts).
        resume_path = output_dir / 'last_resume_state.pt'
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'args': {
                k: v for k, v in vars(args).items()
                if isinstance(v, (str, int, float, bool, type(None)))
            },
        }, resume_path)

    # Save final model
    save_path = output_dir / 'supervised_final.pt'
    torch.save(model.state_dict(), save_path)
    logger.info(f"Final model saved to {save_path}")

    return save_path


def evaluate(model, device, data_loader, num_value_classes, label_smoothing: float = 0.0) -> tuple:
    """Evaluate model on a dataset.

    Returns (policy_loss, value_loss, policy_top1_accuracy, value_accuracy).
    In classification mode, value_loss is CE and value_accuracy is class argmax match.
    In regression mode, value_loss is MSE and value_accuracy is sign accuracy.
    """
    is_regression = (model.value_mode == 'regression')
    model.eval()
    total_policy = 0.0
    total_value = 0.0
    correct = 0
    value_correct = 0
    total = 0

    with torch.no_grad():
        for states, policies, values in data_loader:
            states = states.to(device)
            policies = policies.to(device)
            values = values.to(device)

            pred_logits, value_pred = model(states)

            if policies.dim() == 1:
                expert_moves = policies.long()
                policy_loss = F.cross_entropy(pred_logits, expert_moves, label_smoothing=label_smoothing)
            else:
                log_probs = F.log_softmax(pred_logits, dim=1)
                policy_loss = -(policies * log_probs).sum(dim=1).mean()
                expert_moves = policies.argmax(dim=1)

            if is_regression:
                value_loss = F.mse_loss(value_pred, values.float())
            else:
                value_logits = model._value_logits
                target_class = _value_targets_to_classes(values, num_value_classes)
                value_loss = F.cross_entropy(value_logits, target_class)

            total_policy += policy_loss.item() * len(states)
            total_value += value_loss.item() * len(states)

            pred_moves = pred_logits.argmax(dim=1)
            correct += (pred_moves == expert_moves).sum().item()

            if is_regression:
                value_correct += (
                    torch.sign(value_pred) == torch.sign(values.float())
                ).sum().item()
            else:
                pred_class = model._value_logits.argmax(dim=1)
                target_class = _value_targets_to_classes(values, num_value_classes)
                value_correct += (pred_class == target_class).sum().item()
            total += len(states)

    n = len(data_loader.dataset)
    policy_top1 = correct / total if total > 0 else 0
    value_top1 = value_correct / total if total > 0 else 0
    return total_policy / n, total_value / n, policy_top1, value_top1


def main():
    parser = argparse.ArgumentParser(
        description='Supervised pre-training from expert YINSH games'
    )

    # Data source (one required)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--data', type=str,
                           help='Path to .npz file with training data')
    data_group.add_argument('--data-dir', type=str,
                           help='Directory of .npy files (memory-mapped at '
                                'load time; required for corpora that exceed '
                                'RAM, e.g. yngine_volume — see '
                                'scripts/convert_npz_to_mmap_shards.py)')
    data_group.add_argument('--games-dir', type=str,
                           help='Directory of JSON game files to convert')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs (default: 30)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size (default: 256)')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='DataLoader worker processes (default 2; raise on many-core boxes to keep the GPU fed)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Initial learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--value-weight', type=float, default=1.0,
                       help='Weight for value loss relative to policy (default: 1.0)')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                       help='L2: policy label smoothing for hard-target CE. Keeps '
                            'entropy in the policy so training on sharp expert '
                            'targets (under Dropout=0) does not over-concentrate '
                            'to a single modal opening. Validated at 0.1; set 0 '
                            'to disable. No-op for soft/distribution targets.')
    parser.add_argument('--enable-symmetric-reg', action='store_true',
                       help='E16: add the D2 symmetric-weight regularizer (shared '
                            'with the self-play trainer). Recommended ON — the '
                            'supervised pretrain is where most of the network '
                            'representation (and its D2 asymmetry) is learned, so '
                            'enforcing weight symmetry here keeps the self-play '
                            'loop from inheriting an already-asymmetric net.')
    parser.add_argument('--symmetric-reg-weight', type=float, default=0.1,
                       help='E16 outer weight α (default 0.1).')
    parser.add_argument('--symmetric-reg-value-weight', type=float, default=20.0,
                       help='E16 value-asymmetry weight (default 20, measured — see '
                            'scripts/investigate_e16_value_weight.py).')
    parser.add_argument('--symmetric-reg-every-k-steps', type=int, default=10,
                       help='E16 cadence: regularize every K batches (default 10).')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split fraction (default: 0.1)')

    # Data options
    parser.add_argument('--min-rating', type=int, default=0,
                       help='Minimum player rating for game filtering')

    # Model options
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to a state_dict to warm-start from (model '
                            'weights only; optimizer/epoch reset to fresh). '
                            'For mid-training resume of a prior --output-dir, '
                            'use --resume instead. Mutually exclusive with --resume.')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from <output_dir>/last_resume_state.pt — '
                            'restores model + optimizer state and continues from '
                            'saved_epoch + 1. Extends cleanly: pass --epochs '
                            'higher than the prior run to add more epochs '
                            '(cosine schedule recomputes over the new total). '
                            'Mutually exclusive with --checkpoint.')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/mps/cpu)')
    parser.add_argument('--use-enhanced-encoding', action='store_true',
                       help='Use 15-channel enhanced encoding (default: 6-channel basic). '
                            'Must match the channel count of any --checkpoint loaded.')
    parser.add_argument('--num-channels', type=int, default=256,
                       help='ResNet channel width (default: 256)')
    parser.add_argument('--num-blocks', type=int, default=12,
                       help='Number of residual/attention blocks (default: 12)')
    parser.add_argument('--value-mode', type=str, default='classification',
                       choices=['classification', 'regression'],
                       help="Value-head loss structure. 'classification' "
                            "(default) discretizes the outcome into "
                            "num_value_classes bins and trains with "
                            "F.cross_entropy — matches the self-play "
                            "trainer so the warm-start isn't washed out. "
                            "'regression' trains a scalar tanh head with "
                            "F.mse_loss against the raw value target — "
                            "Branch A4 hypothesis: the 3-class CE plateau "
                            "is target-discretization, not capacity. "
                            "Self-play trainer must match if you intend "
                            "to chain pretrain -> self-play.")
    parser.add_argument('--num-value-classes', type=int, default=None,
                       help='Override num_value_classes for classification mode '
                            '(default: model default of 7). Ignored in regression mode.')
    parser.add_argument('--value-head-type', type=str, default='spatial',
                       choices=['spatial', 'gap', 'gap_v2'],
                       help="Value head architecture: 'spatial' (legacy ~4M "
                            "params, Flatten(64*11*11)->512->256->out), 'gap' "
                            "(~17K params, 1x1 conv->GAP->Linear(64, out) — "
                            "direct projection, lost to spatial head in SPRT), "
                            "or 'gap_v2' (~22K params, adds hidden layer: 1x1 "
                            "conv->GAP->Linear(64,80)->ReLU->Linear(80,out), "
                            "KataGo-canonical). Defaults to 'spatial' for "
                            "back-compat. Use 'gap_v2' to train a GAP-native "
                            "supervised init from scratch — the parallel-track "
                            "alternative to warm-starting a spatial-head "
                            "checkpoint into a fresh head via run_training.py's "
                            "--init-checkpoint.")

    # Output options
    parser.add_argument('--output-dir', type=str, default='models/supervised',
                       help='Directory for checkpoints (default: models/supervised)')
    parser.add_argument('--save-every', type=int, default=10,
                       help='Save checkpoint every N epochs (default: 10)')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    # Load data
    states, policies, values = load_data(args)

    # Create dataset and split
    dataset = ExpertDataset(states, policies, values)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logger.info(f"Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0,
                              prefetch_factor=4 if args.num_workers > 0 else None)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0,
                            prefetch_factor=4 if args.num_workers > 0 else None)

    # Create model
    model, device, resume_bundle = create_model(args)
    logger.info(f"Model on {device}, "
                f"{sum(p.numel() for p in model.parameters()):,} parameters")

    # Train
    t0 = time.time()
    best_path = train(model, device, train_loader, val_loader, args,
                      resume_bundle=resume_bundle)
    elapsed = time.time() - t0

    logger.info(f"Training complete in {elapsed:.0f}s. "
                f"Best model: {best_path}")


if __name__ == '__main__':
    main()
