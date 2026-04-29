"""Sparse Autoencoder for value-head feature interpretation (Track B §8).

Standard top-K-sparse-AE formulation following Anthropic's "Towards
Monosemanticity" recipe:

  encoder(x) = ReLU(W_enc @ (x - bias_dec) + bias_enc)
  decoder(z) = W_dec @ z + bias_dec
  loss      = MSE(x, decoder(encoder(x))) + λ_L1 · ||encoder(x)||_1

The pre-bias subtraction (`x - bias_dec`) is what makes the geometry sensible:
the decoder bias becomes an interpretable "average activation" point, and
features encode directions away from it.

For the 256-dim value-head activation we use `num_features = 8 × 256 = 2048`,
matching the prompt and standard 8× expansion conventions for
medium-dimensional SAEs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEConfig:
    """Hyperparameters for the SAE."""
    input_dim: int = 256
    expansion: int = 8                 # → num_features = expansion * input_dim
    l1_coefficient: float = 1e-3       # sparsity weight
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 50
    # Re-init dead features (those that haven't fired in `dead_feature_window`
    # batches) by setting their decoder weight to a top-eigenvector of the
    # current reconstruction residual. Off by default; turn on if dead-feature
    # rates exceed ~30%.
    dead_feature_resampling: bool = False
    dead_feature_window: int = 1000
    seed: int = 0

    @property
    def num_features(self) -> int:
        return self.expansion * self.input_dim


class SparseAutoencoder(nn.Module):
    """Single-layer SAE with tied-bias-via-pre-subtraction geometry."""

    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        n, d = cfg.num_features, cfg.input_dim

        # Decoder bias doubles as the "centroid" for the pre-subtraction term.
        # Initialized to zero — first batches will pull it toward the
        # activation mean.
        self.bias_dec = nn.Parameter(torch.zeros(d))
        self.bias_enc = nn.Parameter(torch.zeros(n))

        # Encoder/decoder weights. Encoder rows are unit-normalized at init
        # to make L1 magnitudes comparable across features. Decoder columns
        # are initialized to encoder rows transposed (tied init, then untied
        # during training).
        torch.manual_seed(cfg.seed)
        W = torch.randn(n, d)
        W = W / W.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_enc = nn.Parameter(W.clone())
        self.W_dec = nn.Parameter(W.t().clone())

        # For dead-feature tracking.
        self.register_buffer('feature_activity', torch.zeros(n))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d) → z: (B, n)"""
        return F.relu(F.linear(x - self.bias_dec, self.W_enc, self.bias_enc))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, n) → x_hat: (B, d)"""
        return F.linear(z, self.W_dec) + self.bias_dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def loss(self, x: torch.Tensor) -> dict:
        x_hat, z = self.forward(x)
        recon_loss = F.mse_loss(x_hat, x)
        l1_loss = z.abs().sum(dim=1).mean()
        total = recon_loss + self.cfg.l1_coefficient * l1_loss
        return {
            'total': total,
            'recon': recon_loss.detach(),
            'l1': l1_loss.detach(),
            'sparsity': (z > 0).float().mean().detach(),
            'z': z,
        }


@dataclass
class SAETrainingStats:
    """Per-epoch stats — also persists to disk alongside the trained SAE."""
    epoch: list = field(default_factory=list)
    recon_loss: list = field(default_factory=list)
    l1_loss: list = field(default_factory=list)
    total_loss: list = field(default_factory=list)
    sparsity: list = field(default_factory=list)  # fraction of active features per sample
    dead_feature_count: list = field(default_factory=list)


def train_sae(
    sae: SparseAutoencoder,
    activations: np.ndarray,
    cfg: Optional[SAEConfig] = None,
    device: str = 'cpu',
    verbose: bool = True,
) -> SAETrainingStats:
    """Train the SAE on captured activations.

    Args:
        sae: SparseAutoencoder instance (already constructed with a config).
        activations: (N, input_dim) numpy array of captured activations.
        cfg: Optional override config; defaults to `sae.cfg`.
        device: Torch device string.

    Returns:
        SAETrainingStats with per-epoch loss/sparsity/dead-feature numbers.
    """
    cfg = cfg or sae.cfg
    if activations.shape[1] != cfg.input_dim:
        raise ValueError(
            f"activations dim {activations.shape[1]} != cfg.input_dim {cfg.input_dim}"
        )
    sae = sae.to(device)
    sae.train()

    x = torch.from_numpy(activations).float().to(device)
    n_samples = x.shape[0]
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.learning_rate)
    stats = SAETrainingStats()

    rng = np.random.default_rng(cfg.seed)
    for epoch in range(cfg.epochs):
        # Shuffle indices each epoch.
        perm = rng.permutation(n_samples)
        epoch_recon = 0.0
        epoch_l1 = 0.0
        epoch_total = 0.0
        epoch_sparsity = 0.0
        n_batches = 0

        # Reset feature activity counter at start of epoch so the dead-
        # feature metric reflects this epoch's pattern.
        sae.feature_activity.zero_()

        for start in range(0, n_samples, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            batch = x[idx]

            losses = sae.loss(batch)
            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()

            with torch.no_grad():
                # Track which features fired this batch.
                active_mask = (losses['z'] > 0).any(dim=0)
                sae.feature_activity += active_mask.float()

            epoch_recon += losses['recon'].item()
            epoch_l1 += losses['l1'].item()
            epoch_total += losses['total'].item()
            epoch_sparsity += losses['sparsity'].item()
            n_batches += 1

        # Per-epoch stats.
        dead = int((sae.feature_activity == 0).sum().item())
        stats.epoch.append(epoch)
        stats.recon_loss.append(epoch_recon / n_batches)
        stats.l1_loss.append(epoch_l1 / n_batches)
        stats.total_loss.append(epoch_total / n_batches)
        stats.sparsity.append(epoch_sparsity / n_batches)
        stats.dead_feature_count.append(dead)

        if verbose and (epoch == 0 or (epoch + 1) % max(1, cfg.epochs // 10) == 0):
            print(
                f"[SAE epoch {epoch+1:3d}/{cfg.epochs}] "
                f"recon={stats.recon_loss[-1]:.5f} "
                f"l1={stats.l1_loss[-1]:.3f} "
                f"sparsity={stats.sparsity[-1]:.3f} "
                f"dead={dead}/{cfg.num_features}"
            )

    return stats


def save_sae(sae: SparseAutoencoder, stats: SAETrainingStats, output_dir: str | Path) -> None:
    """Persist the trained SAE + training stats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'state_dict': sae.state_dict(),
        'config': sae.cfg.__dict__,
    }, output_dir / 'sae.pt')
    import json
    with open(output_dir / 'training_stats.json', 'w') as f:
        json.dump({
            'epoch': stats.epoch,
            'recon_loss': stats.recon_loss,
            'l1_loss': stats.l1_loss,
            'total_loss': stats.total_loss,
            'sparsity': stats.sparsity,
            'dead_feature_count': stats.dead_feature_count,
        }, f, indent=2)


def load_sae(path: str | Path) -> SparseAutoencoder:
    """Reconstruct an SAE from a saved checkpoint."""
    blob = torch.load(path, map_location='cpu')
    cfg = SAEConfig(**blob['config'])
    sae = SparseAutoencoder(cfg)
    sae.load_state_dict(blob['state_dict'])
    return sae
