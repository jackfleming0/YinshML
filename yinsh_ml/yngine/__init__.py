"""Bridge to the external `temhelk/yngine` C++ MCTS engine.

Used by ``scripts/eval_vs_yngine.py`` to measure our trained models against
yngine as an independent strength reference. yngine is a pure-MCTS, no-NN
engine; it is also the corpus generator that produced the 200K-game volume
pretraining dataset (see VOLUME_PRETRAIN_RESULTS.md).

The library has no native CLI, so we drive it through a small stdin/stdout
protocol implemented by ``third_party/yngine_driver/yngine_driver.cpp``. Build
it once with ``bash third_party/yngine_driver/build.sh`` and the bridge will
find the resulting binary automatically.
"""

from .bridge import Yngine, YngineError, default_binary_path

__all__ = ["Yngine", "YngineError", "default_binary_path"]
