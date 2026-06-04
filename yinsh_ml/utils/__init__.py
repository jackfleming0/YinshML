"""Utility functions and classes."""

from .state_conversion import StateConverter

# Visualizer classes are exported lazily so that importing anything else
# from yinsh_ml.utils (e.g. EnhancedStateEncoder in self-play workers) does
# NOT eagerly pull in matplotlib/seaborn. Workers without viz deps then fail
# loudly only if they actually touch the visualizer classes, instead of
# crash-looping during pool init. Same pattern as yinsh_ml.tracking.
__all__ = ['StateConverter', 'TrainingVisualizer', 'GameVisualizer']


def __getattr__(name):
    if name in ('TrainingVisualizer', 'GameVisualizer'):
        from .visualization import TrainingVisualizer, GameVisualizer
        return {'TrainingVisualizer': TrainingVisualizer,
                'GameVisualizer': GameVisualizer}[name]
    raise AttributeError(f"module '{__name__}' has no attribute {name!r}")
