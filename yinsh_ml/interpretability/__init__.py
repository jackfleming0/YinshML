"""Interpretability tools for YinshML value/policy heads.

Track B §8 — value-head SAE probe. Identifies what YINSH concepts the value
head has learned vs. failed to learn. Pairs with §5 (search-consistency):
§5 tests whether the training signal is the bottleneck; §8 tests whether
the representation is.

Modules:
  * `activation_capture` — hook-based activation extraction from the value
    head's penultimate ReLU (model.py:153, 256-dim).
  * `sae` — Sparse Autoencoder model + training loop (256 → 2048 features).
  * `feature_analysis` — top-activating-position finder, confident-error
    dataset, label helpers.
"""

from .activation_capture import ValueHeadActivationCapture
from .sae import SparseAutoencoder, train_sae
from .feature_analysis import FeatureAnalyzer

__all__ = [
    'ValueHeadActivationCapture',
    'SparseAutoencoder',
    'train_sae',
    'FeatureAnalyzer',
]
