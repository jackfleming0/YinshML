"""Data pipeline for scraping, parsing, and converting expert YINSH games."""

from .converter import GameConverter
from .validator import GameValidator

__all__ = ['GameConverter', 'GameValidator']

# Lazy imports for scrapers/parsers (avoid import overhead when not needed)
def get_lg_scraper():
    from .scrapers.little_golem import LittleGolemScraper
    return LittleGolemScraper

def get_cg_scraper():
    from .scrapers.codingame import CodinGameScraper
    return CodinGameScraper
