"""Find and characterize what each SAE feature has learned (Track B §8).

After training the SAE, we want to know what each of the 2048 features
fires on. The standard recipe:

  1. Run the captured activations through the SAE encoder → (N, num_features)
     activations.
  2. For each feature, find the top-K positions where it fires most strongly.
  3. Cross-reference those positions to game-state properties (ring count,
     marker patterns, game phase, score differential, threat structure) to
     hand-label the feature.

This module produces the raw artifact (per-feature top-position indices +
per-feature firing-rate stats); hand-labeling is a separate report-writing
step. Also identifies "confident error" positions — where the network's
value prediction is far from the actual outcome — so we can ask "what board
features do these positions share that no SAE feature captures?"
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .sae import SparseAutoencoder


@dataclass
class FeatureSummary:
    """Per-feature stats. One entry per SAE feature."""
    feature_id: int
    firing_rate: float           # fraction of positions where activation > 0
    mean_activation: float       # mean over positions where it's active
    max_activation: float
    top_k_indices: list          # position indices in `activations`
    top_k_activations: list


@dataclass
class FeatureReport:
    """Aggregate analysis of an SAE over a captured activation set."""
    num_positions: int
    num_features: int
    dead_feature_count: int      # features that never fire
    sparse_feature_count: int    # features firing on < 1% of positions
    dense_feature_count: int     # features firing on > 50% of positions
    features: list = field(default_factory=list)


class FeatureAnalyzer:
    """Compute per-feature top-position indices + firing-rate stats."""

    def __init__(self, sae: SparseAutoencoder, device: str = 'cpu'):
        self.sae = sae.to(device).eval()
        self.device = device

    def encode(self, activations: np.ndarray) -> np.ndarray:
        """Run captured activations through the SAE encoder. Returns
        (N, num_features) sparse encoding."""
        x = torch.from_numpy(activations).float().to(self.device)
        with torch.no_grad():
            z = self.sae.encode(x)
        return z.cpu().numpy()

    def per_feature_summary(
        self,
        activations: np.ndarray,
        top_k: int = 20,
    ) -> FeatureReport:
        """For each SAE feature, summarize its firing pattern over the
        captured activation set. Returns a `FeatureReport`."""
        z = self.encode(activations)  # (N, num_features)
        n_positions, n_features = z.shape

        firing_rates = (z > 0).mean(axis=0)  # per-feature
        mean_acts = np.zeros(n_features, dtype=np.float32)
        max_acts = z.max(axis=0)
        # Mean over only the active positions — guards against dilution by
        # zeros, which would conflate "feature fires rarely but strongly"
        # with "feature fires often but weakly."
        for f in range(n_features):
            active = z[:, f] > 0
            if active.any():
                mean_acts[f] = z[active, f].mean()

        feature_summaries = []
        for f in range(n_features):
            col = z[:, f]
            # Argsort descending; take top_k. Skip features that never fire.
            if (col > 0).sum() == 0:
                top_indices = []
                top_acts_list = []
            else:
                k = min(top_k, int((col > 0).sum()))
                top_indices = np.argsort(-col)[:k].tolist()
                top_acts_list = [float(col[i]) for i in top_indices]

            feature_summaries.append(FeatureSummary(
                feature_id=f,
                firing_rate=float(firing_rates[f]),
                mean_activation=float(mean_acts[f]),
                max_activation=float(max_acts[f]),
                top_k_indices=top_indices,
                top_k_activations=top_acts_list,
            ))

        return FeatureReport(
            num_positions=n_positions,
            num_features=n_features,
            dead_feature_count=int((firing_rates == 0).sum()),
            sparse_feature_count=int((firing_rates < 0.01).sum()),
            dense_feature_count=int((firing_rates > 0.5).sum()),
            features=feature_summaries,
        )

    @staticmethod
    def find_confident_errors(
        value_preds: np.ndarray,
        actual_outcomes: np.ndarray,
        confidence_threshold: float = 0.7,
    ) -> np.ndarray:
        """Return indices of positions where the network's value prediction
        was confidently wrong — |v_pred| > threshold but sign(v_pred) !=
        sign(actual). These are the positions to mine for "what concept did
        the value head miss?" (Track B §8 step 3).

        Args:
            value_preds: (N,) array of network value predictions in [-1, 1].
            actual_outcomes: (N,) array of true outcomes in [-1, 1].
            confidence_threshold: |v_pred| must exceed this.

        Returns:
            (M,) array of indices into value_preds.
        """
        if value_preds.shape != actual_outcomes.shape:
            raise ValueError("value_preds and actual_outcomes must have same shape")
        confident = np.abs(value_preds) > confidence_threshold
        wrong_sign = np.sign(value_preds) != np.sign(actual_outcomes)
        # Exclude actual==0 (true draw — sign undefined) from the "wrong sign"
        # set. A confident +0.8 prediction on a draw is interesting but the
        # "confidently wrong" framing wants a clear miss.
        nontrivial_truth = actual_outcomes != 0
        mask = confident & wrong_sign & nontrivial_truth
        return np.where(mask)[0]


def save_report(report: FeatureReport, output_path: str | Path) -> None:
    """Persist a feature report as JSON. Lists are kept short so the file
    stays human-readable (each feature has top_k indices/activations only,
    not the full activation column)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'num_positions': report.num_positions,
        'num_features': report.num_features,
        'dead_feature_count': report.dead_feature_count,
        'sparse_feature_count': report.sparse_feature_count,
        'dense_feature_count': report.dense_feature_count,
        'features': [asdict(f) for f in report.features],
    }
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)
