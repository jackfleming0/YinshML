"""Data pipeline for scraping, parsing, and converting expert YINSH games."""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ['GameConverter', 'GameValidator']

# GameConverter/GameValidator pull in torch (via the state encoder). Import
# them lazily (PEP 562) so importing a lightweight sibling — e.g.
# ``yinsh_ml.data.human_games``, used by torch-free engine/heuristic tests —
# does not force a torch import. The public API is unchanged: callers can
# still do ``from yinsh_ml.data import GameConverter``.
if TYPE_CHECKING:  # pragma: no cover - typing only
    from .converter import GameConverter
    from .validator import GameValidator


def __getattr__(name):
    if name == 'GameConverter':
        return import_module('.converter', __name__).GameConverter
    if name == 'GameValidator':
        return import_module('.validator', __name__).GameValidator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Lazy imports for scrapers/parsers (avoid import overhead when not needed)
def get_lg_scraper():
    from .scrapers.little_golem import LittleGolemScraper
    return LittleGolemScraper

def get_cg_scraper():
    from .scrapers.codingame import CodinGameScraper
    return CodinGameScraper
