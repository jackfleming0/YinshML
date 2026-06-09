#!/usr/bin/env python
"""Stream the first N rows of arrays out of a (possibly huge, compressed) .npz —
WITHOUT materializing the whole thing.

Why: `expert_games/yngine_volume.npz` is 600 MB compressed but ~40 GB
decompressed (13.6M x 6x11x11 float32). A plain `np.load(...)['states']`
materializes all 40 GB and thrashes any machine with <40 GB RAM. This reads the
DEFLATE stream of each member sequentially and keeps only the first N rows, so
peak memory is ~`N * rowbytes` (plus one chunk), not the full array.

The corpus is game-ordered (consecutive positions are the same game, side-to-move
POV labels alternate +/-1), so a contiguous prefix is a coherent set of whole
games — exactly what `value_ceiling_probe.py`'s contiguous train/test game-split
expects.

Usage:
  python scripts/subsample_npz_prefix.py \
      --src expert_games/yngine_volume.npz \
      --out expert_games/yngine_volume_2M_6ch.npz \
      --n 2000000                              # states + values by default

Sizing (states float32, 6x11x11 = 726 floats/row): N rows -> N*2904 bytes.
  500k -> 1.45 GB   2M -> 5.8 GB   5M -> 14.5 GB.  Pick N to fit the box's RAM
  with headroom for training (a CUDA box with 64 GB RAM handles 5M comfortably).
"""
import argparse
import os
import time
import zipfile

import numpy as np
from numpy.lib import format as npf


def read_prefix(zf, member, n, chunk=200_000):
    with zf.open(member) as f:
        ver = npf.read_magic(f)
        shape, _, dtype = (npf.read_array_header_1_0(f) if ver == (1, 0)
                           else npf.read_array_header_2_0(f))
        rowsize = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        n = min(n, shape[0])
        out = np.empty((n,) + tuple(shape[1:]), dtype=dtype)
        flat = out.reshape(n, rowsize)
        bpr = rowsize * dtype.itemsize
        read = 0
        while read < n:
            k = min(chunk, n - read)
            flat[read:read + k] = np.frombuffer(
                f.read(k * bpr), dtype=dtype, count=k * rowsize).reshape(k, rowsize)
            read += k
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True, help="number of leading rows to keep")
    ap.add_argument("--members", default="states,values",
                    help="comma-separated array names to extract (default: states,values)")
    args = ap.parse_args()

    t0 = time.time()
    zf = zipfile.ZipFile(args.src)
    members = [m.strip() for m in args.members.split(",")]
    arrs = {}
    for m in members:
        member_file = m if m.endswith(".npy") else m + ".npy"
        arrs[m] = read_prefix(zf, member_file, args.n)
        print(f"  {m}: {arrs[m].shape} {arrs[m].dtype} ({(time.time()-t0):.0f}s)", flush=True)

    if "values" in arrs:
        v = arrs["values"]
        print(f"  decisive={int((v != 0).sum())} draws={int((v == 0).sum())}")
    np.savez_compressed(args.out, **arrs)
    sz = os.path.getsize(args.out) / 1e6
    load_gb = sum(a.nbytes for a in arrs.values()) / 1e9
    print(f"wrote {args.out} ({sz:.0f} MB on disk, ~{load_gb:.1f} GB when loaded) "
          f"in {(time.time()-t0):.0f}s")


if __name__ == "__main__":
    main()
