#!/usr/bin/env python3
"""Supervised pre-training from expert game data.

Produces a model checkpoint pre-trained on expert games that can then
feed into the self-play training loop as a warm start.

Usage:
    # Pre-train from converted game data (.npz)
    python scripts/run_supervised_pretraining.py --data expert_games/training_data.npz

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

logger = logging.getLogger(__name__)


class ExpertDataset(Dataset):
    """PyTorch Dataset for expert game training data."""

    def __init__(self, states: np.ndarray, policies: np.ndarray,
                 values: np.ndarray):
        self.states = torch.from_numpy(states)
        self.policies = torch.from_numpy(policies)
        self.values = torch.from_numpy(values)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]


def load_data(args) -> tuple:
    """Load training data from .npz file or raw game directory."""
    if args.data:
        logger.info(f"Loading pre-converted data from {args.data}")
        states, policies, values = GameConverter.load_training_data(args.data)
    elif args.games_dir:
        logger.info(f"Converting games from {args.games_dir}")
        encoder = EnhancedStateEncoder() if args.use_enhanced_encoding else StateEncoder()
        converter = GameConverter(encoder=encoder)
        pairs = converter.convert_directory(args.games_dir,
                                            min_rating=args.min_rating)
        if not pairs:
            raise RuntimeError(f"No valid training pairs from {args.games_dir}")

        states = np.array([p['state'] for p in pairs], dtype=np.float32)
        policies = np.array([p['policy'] for p in pairs], dtype=np.float32)
        values = np.array([p['value'] for p in pairs], dtype=np.float32)

        # Save for future use
        save_path = Path(args.games_dir) / 'training_data.npz'
        np.savez_compressed(save_path, states=states, policies=policies,
                           values=values)
        logger.info(f"Saved converted data to {save_path}")
    else:
        raise RuntimeError("Must specify --data or --games-dir")

    logger.info(f"Loaded {len(states)} positions "
                f"(states: {states.shape}, policies: {policies.shape})")
    return states, policies, values


def create_model(args) -> tuple:
    """Create model and move to appropriate device."""
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    )

    input_channels = 15 if args.use_enhanced_encoding else 6
    model = YinshNetwork(
        num_channels=256,
        num_blocks=12,
        input_channels=input_channels,
    ).to(device)

    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device,
                                weights_only=True)
        model.load_state_dict(state_dict)

    return model, device


def _value_targets_to_classes(values: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Map scalar targets in [-1, 1] to class indices in [0, num_classes-1].

    Matches the discretization used in yinsh_ml/training/trainer.py so the
    supervised and self-play value heads optimize the same objective.
    Expert outcomes {-1, 0, +1} → classes {0, mid, num_classes-1}.
    """
    normalized = (values + 1.0) / 2.0 * (num_classes - 1)
    return torch.round(normalized).long().clamp(0, num_classes - 1)


def train(model, device, train_loader, val_loader, args):
    """Run supervised training loop."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    # LR schedule: cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 100
    )

    best_val_loss = float('inf')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_value_classes = model.num_value_classes

    for epoch in range(1, args.epochs + 1):
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

            pred_logits, _ = model(states)
            value_logits = model._value_logits  # populated in forward()

            # Policy loss: cross-entropy against expert moves
            log_probs = F.log_softmax(pred_logits, dim=1)
            policy_loss = -(policies * log_probs).sum(dim=1).mean()

            # Value loss: cross-entropy on discretized outcome classes.
            # Mirrors yinsh_ml/training/trainer.py — same loss surface as
            # self-play so the warm-start checkpoint isn't washed out on iter 1.
            target_class = _value_targets_to_classes(values, num_value_classes)
            value_loss = F.cross_entropy(value_logits, target_class)

            # Combined loss
            loss = policy_loss + args.value_weight * value_loss
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_policy_loss += policy_loss.item() * len(states)
            train_value_loss += value_loss.item() * len(states)

            # Top-1 accuracy
            pred_moves = pred_logits.argmax(dim=1)
            expert_moves = policies.argmax(dim=1)
            train_correct += (pred_moves == expert_moves).sum().item()

            pred_class = value_logits.argmax(dim=1)
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
            model, device, val_loader, num_value_classes
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

    # Save final model
    save_path = output_dir / 'supervised_final.pt'
    torch.save(model.state_dict(), save_path)
    logger.info(f"Final model saved to {save_path}")

    return save_path


def evaluate(model, device, data_loader, num_value_classes) -> tuple:
    """Evaluate model on a dataset.

    Returns (policy_loss, value_ce_loss, policy_top1_accuracy, value_class_accuracy).
    """
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

            pred_logits, _ = model(states)
            value_logits = model._value_logits

            log_probs = F.log_softmax(pred_logits, dim=1)
            policy_loss = -(policies * log_probs).sum(dim=1).mean()

            target_class = _value_targets_to_classes(values, num_value_classes)
            value_loss = F.cross_entropy(value_logits, target_class)

            total_policy += policy_loss.item() * len(states)
            total_value += value_loss.item() * len(states)

            pred_moves = pred_logits.argmax(dim=1)
            expert_moves = policies.argmax(dim=1)
            correct += (pred_moves == expert_moves).sum().item()

            pred_class = value_logits.argmax(dim=1)
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
    data_group.add_argument('--games-dir', type=str,
                           help='Directory of JSON game files to convert')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs (default: 30)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Initial learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--value-weight', type=float, default=1.0,
                       help='Weight for value loss relative to policy (default: 1.0)')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split fraction (default: 0.1)')

    # Data options
    parser.add_argument('--min-rating', type=int, default=0,
                       help='Minimum player rating for game filtering')

    # Model options
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/mps/cpu)')
    parser.add_argument('--use-enhanced-encoding', action='store_true',
                       help='Use 15-channel enhanced encoding (default: 6-channel basic). '
                            'Must match the channel count of any --checkpoint loaded.')

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
                              shuffle=True, num_workers=2, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True,
                            persistent_workers=True)

    # Create model
    model, device = create_model(args)
    logger.info(f"Model on {device}, "
                f"{sum(p.numel() for p in model.parameters()):,} parameters")

    # Train
    t0 = time.time()
    best_path = train(model, device, train_loader, val_loader, args)
    elapsed = time.time() - t0

    logger.info(f"Training complete in {elapsed:.0f}s. "
                f"Best model: {best_path}")


if __name__ == '__main__':
    main()
