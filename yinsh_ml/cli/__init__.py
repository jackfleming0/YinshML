"""
CLI interface for YinshML experiment tracking.

Provides command-line tools for managing experiments, including:
- Creating and starting experiments
- Listing and searching experiments
- Comparing experiment results
- Reproducing experiments
"""

__version__ = "0.1.0"

__all__ = ['cli']

# Lazy import to avoid circular dependencies
def __getattr__(name):
    if name == 'cli':
        from .main import cli
        return cli
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'") 