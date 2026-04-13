"""Scraper/downloader for Boardspace.net YINSH games.

Downloads SGF files from the open Apache directory listing at
boardspace.net/yinsh/yinshgames/ and converts them to the
standardized game format.

The archive contains ~30,000 games spanning 2005-2026, stored as:
- Loose .sgf files in the root directory (recent games)
- Yearly archive-YYYY/ directories containing .zip files of batched games

Usage:
    scraper = BoardspaceScraper(output_dir='expert_games/boardspace')
    games = scraper.scrape_all()
"""

import json
import logging
import os
import re
import time
import zipfile
from pathlib import Path
from typing import List, Dict, Optional, Set
import ssl
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

import certifi

from ..parsers.boardspace_sgf import parse_boardspace_sgf

_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

logger = logging.getLogger(__name__)

BASE_URL = "https://www.boardspace.net/yinsh/yinshgames/"
DEFAULT_DELAY = 2.0


class BoardspaceScraper:
    """Downloads and parses YINSH games from Boardspace.net.

    Args:
        output_dir: Local directory for storing downloaded files and parsed games.
        delay: Seconds to wait between HTTP requests.
    """

    def __init__(self, output_dir: str = 'expert_games/boardspace',
                 delay: float = DEFAULT_DELAY):
        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / 'raw'
        self.json_dir = self.output_dir / 'json'
        self.delay = delay
        self._last_request_time = 0.0
        self._manifest_path = self.output_dir / 'manifest.json'
        self._manifest: Dict = {}

    def scrape_all(self, max_games: Optional[int] = None) -> List[Dict]:
        """Full scrape: discover files, download, parse, return games.

        Args:
            max_games: Stop after this many successfully parsed games.
                       None = no limit.

        Returns:
            List of standardized game dicts.
        """
        self._ensure_dirs()
        self._load_manifest()

        # Discover all SGF/zip URLs
        urls = self.discover_files()
        logger.info(f"Discovered {len(urls)} files to process")

        games = []
        sgf_count = 0

        for url in urls:
            if max_games and len(games) >= max_games:
                break

            if url.endswith('.zip'):
                zip_path = self._download(url)
                if zip_path:
                    new_games = self._process_zip(zip_path, max_games and max_games - len(games))
                    games.extend(new_games)
            elif url.endswith('.sgf'):
                sgf_path = self._download(url)
                if sgf_path:
                    game = self._process_sgf(sgf_path)
                    if game:
                        games.append(game)

            sgf_count += 1
            if sgf_count % 50 == 0:
                logger.info(f"Progress: {len(games)} games parsed from {sgf_count} files")
                self._save_manifest()

        self._save_manifest()
        logger.info(f"Scrape complete: {len(games)} games from {sgf_count} files")
        return games

    def discover_files(self) -> List[str]:
        """Crawl directory listing to find all SGF and zip file URLs.

        Returns:
            List of full URLs to download.
        """
        urls = []

        # Fetch root directory
        root_html = self._fetch(BASE_URL)
        if not root_html:
            return urls

        # Find loose .sgf files
        sgf_links = re.findall(r'href="([^"]+\.sgf)"', root_html, re.IGNORECASE)
        for link in sgf_links:
            urls.append(BASE_URL + link)

        # Find archive-YYYY/ directories
        archive_dirs = re.findall(r'href="(archive-\d{4}/)"', root_html)
        for archive_dir in sorted(archive_dirs):
            archive_url = BASE_URL + archive_dir
            archive_html = self._fetch(archive_url)
            if not archive_html:
                continue

            # Find .zip files in archive
            zip_links = re.findall(r'href="([^"]+\.zip)"', archive_html, re.IGNORECASE)
            for link in zip_links:
                urls.append(archive_url + link)

        # Find snapshot directories (games-Mon-DD-YYYY/)
        snapshot_dirs = re.findall(r'href="(games-[^/"]+/)"', root_html)
        for snap_dir in snapshot_dirs:
            snap_url = BASE_URL + snap_dir
            snap_html = self._fetch(snap_url)
            if not snap_html:
                continue
            sgf_links = re.findall(r'href="([^"]+\.sgf)"', snap_html, re.IGNORECASE)
            for link in sgf_links:
                urls.append(snap_url + link)

        logger.info(f"Discovered {len(urls)} files "
                    f"({sum(1 for u in urls if u.endswith('.sgf'))} SGFs, "
                    f"{sum(1 for u in urls if u.endswith('.zip'))} ZIPs)")
        return urls

    def _process_zip(self, zip_path: Path,
                     max_games: Optional[int] = None) -> List[Dict]:
        """Extract and parse SGFs from a zip archive."""
        games = []
        extract_dir = self.raw_dir / 'extracted' / zip_path.stem

        try:
            with zipfile.ZipFile(zip_path) as zf:
                sgf_names = [n for n in zf.namelist() if n.endswith('.sgf')]

                for name in sgf_names:
                    if max_games and len(games) >= max_games:
                        break

                    # Extract to disk
                    extract_path = extract_dir / name
                    if not extract_path.exists():
                        extract_path.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(name) as src, open(extract_path, 'wb') as dst:
                            dst.write(src.read())

                    game = self._process_sgf(extract_path)
                    if game:
                        games.append(game)

        except (zipfile.BadZipFile, OSError) as e:
            logger.error(f"Failed to process zip {zip_path}: {e}")

        return games

    def _process_sgf(self, sgf_path: Path) -> Optional[Dict]:
        """Parse a single SGF file into standardized format."""
        sgf_key = str(sgf_path.name)

        # Check manifest for already-processed files
        if sgf_key in self._manifest and self._manifest[sgf_key].get('parsed'):
            json_path = self.json_dir / f"{sgf_path.stem}.json"
            if json_path.exists():
                try:
                    with open(json_path) as f:
                        return json.load(f)
                except json.JSONDecodeError:
                    pass  # Re-parse if JSON is corrupted

        try:
            with open(sgf_path, encoding='utf-8', errors='replace') as f:
                sgf_text = f.read()
        except OSError as e:
            logger.error(f"Failed to read {sgf_path}: {e}")
            return None

        game = parse_boardspace_sgf(sgf_text)
        if game is None:
            self._manifest[sgf_key] = {'parsed': False, 'error': 'parse_failed'}
            return None

        # Save parsed JSON
        json_path = self.json_dir / f"{sgf_path.stem}.json"
        try:
            with open(json_path, 'w') as f:
                json.dump(game, f, indent=2)
        except OSError as e:
            logger.error(f"Failed to save JSON for {sgf_path}: {e}")

        self._manifest[sgf_key] = {
            'parsed': True,
            'game_id': game.get('game_id', ''),
            'result': game.get('result', ''),
            'n_moves': len(game.get('moves', [])),
        }

        return game

    def _download(self, url: str) -> Optional[Path]:
        """Download a file, skip if already exists locally.

        Returns the local path, or None on failure.
        """
        # Determine local path
        relative = url.replace(BASE_URL, '')
        local_path = self.raw_dir / relative

        if local_path.exists():
            return local_path

        local_path.parent.mkdir(parents=True, exist_ok=True)

        data = self._fetch_bytes(url)
        if data is None:
            return None

        try:
            with open(local_path, 'wb') as f:
                f.write(data)
            return local_path
        except OSError as e:
            logger.error(f"Failed to save {local_path}: {e}")
            return None

    def _fetch(self, url: str) -> Optional[str]:
        """Fetch a URL as text with rate limiting."""
        data = self._fetch_bytes(url)
        if data is None:
            return None
        return data.decode('utf-8', errors='replace')

    def _fetch_bytes(self, url: str) -> Optional[bytes]:
        """Fetch a URL as bytes with rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

        try:
            req = Request(url, headers={
                'User-Agent': 'YinshML-Research/1.0 (academic research)',
            })
            with urlopen(req, timeout=60, context=_SSL_CONTEXT) as resp:
                self._last_request_time = time.time()
                return resp.read()
        except (HTTPError, URLError, OSError) as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def _ensure_dirs(self):
        """Create output directories."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.json_dir.mkdir(parents=True, exist_ok=True)

    def _load_manifest(self):
        """Load the download manifest."""
        if self._manifest_path.exists():
            try:
                with open(self._manifest_path) as f:
                    self._manifest = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._manifest = {}
        else:
            self._manifest = {}

    def _save_manifest(self):
        """Save the download manifest."""
        try:
            with open(self._manifest_path, 'w') as f:
                json.dump(self._manifest, f, indent=2)
        except OSError as e:
            logger.error(f"Failed to save manifest: {e}")
