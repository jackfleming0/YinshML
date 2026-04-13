"""Scrapers for external YINSH game sources."""

from .little_golem import LittleGolemScraper, load_gipf_notation_file
from .codingame import CodinGameScraper
from .boardspace import BoardspaceScraper
from .bga import BGAScraper

__all__ = [
    'LittleGolemScraper', 'CodinGameScraper', 'BoardspaceScraper',
    'BGAScraper', 'load_gipf_notation_file',
]
