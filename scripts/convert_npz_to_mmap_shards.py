#!/usr/bin/env python3
"""Convert a single .npz training corpus to a directory of .npy files
suitable for memory-mapped loading.

Why: large corpora (e.g. yngine_volume.npz, 13.6M positions) decompress
to ~40 GB for the 'states' array alone. Eagerly loading the npz blows
out RAM. Plain .npy files support np.load(..., mmap_mode='r'), so the
OS handles paging and RAM usage stays at batch-size × tensor-size.

The conversion itself streams through zipfile.ZipFile.extractall, which
decompresses to disk in chunks without loading the full array into RAM.

Usage:
    python scripts/convert_npz_to_mmap_shards.py \\
        --input expert_games/yngine_volume.npz \\
        --output expert_games/yngine_volume_mmap/

Disk footprint: ~40 GB for yngine_volume (mostly the states array).
"""

import argparse
import logging
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def convert(npz_path: Path, out_dir: Path) -> None:
    if not npz_path.exists():
        raise FileNotFoundError(f"input npz not found: {npz_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    logger.info(f"Reading {npz_path} ({npz_path.stat().st_size / 1e9:.2f} GB compressed)")

    with zipfile.ZipFile(npz_path) as zf:
        members = [n for n in zf.namelist() if n.endswith('.npy')]
        logger.info(f"Found {len(members)} arrays: {members}")
        for name in members:
            t1 = time.time()
            target = out_dir / name
            with zf.open(name) as src, open(target, 'wb') as dst:
                # 4 MB chunks — keeps RAM flat regardless of array size
                while True:
                    buf = src.read(4 * 1024 * 1024)
                    if not buf:
                        break
                    dst.write(buf)
            size_gb = target.stat().st_size / 1e9
            logger.info(f"  {name}: {size_gb:.2f} GB in {time.time() - t1:.1f}s")

    # Sanity: open each output file as mmap and report shape/dtype
    logger.info("Verifying outputs via mmap_mode='r' ...")
    for npy in sorted(out_dir.glob('*.npy')):
        arr = np.load(npy, mmap_mode='r')
        if arr.ndim == 0:
            logger.info(f"  {npy.name}: scalar = {arr.item()}")
        else:
            logger.info(f"  {npy.name}: shape={arr.shape} dtype={arr.dtype}")
        del arr  # release mmap

    logger.info(f"Done in {time.time() - t0:.1f}s. Output: {out_dir}/")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', type=Path, required=True,
                   help='Path to input .npz file')
    p.add_argument('--output', type=Path, required=True,
                   help='Output directory for .npy files')
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    convert(args.input, args.output)


if __name__ == '__main__':
    main()
