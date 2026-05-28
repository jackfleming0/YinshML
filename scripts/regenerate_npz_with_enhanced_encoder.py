#!/usr/bin/env python3
"""Regenerate a 6-channel volume corpus into a 15-channel one (Branch D.2 Path B).

Reads an existing npz produced by `yngine_corpus_to_npz.py` (6-channel states,
encoded via StateEncoder), decodes each position back to a GameState, and
re-encodes via EnhancedStateEncoder. Writes a directory of .npy files in
the same mmap-compatible format that `run_supervised_pretraining.py
--data-dir` consumes (matching `convert_npz_to_mmap_shards.py`'s output).

KNOWN INFORMATION LOSS — channel 13 (turn_number, normalized move count
capped at 100 plies): NOT recoverable from a 6-channel encoded state.
StateEncoder.decode_state recovers pieces, phase, scores, rings_placed —
but NOT game_state.move_count, which the enhanced encoder needs for
channel 13. The script falls back to 0 for that channel (per
EnhancedStateEncoder.encode_state's existing hasattr guard at
yinsh_ml/utils/enhanced_encoding.py:220). All other 14 channels round-trip
cleanly. See D2_PREP.md §"Path B" for the tradeoff.

If raw yngine shard files are available, prefer the cleaner Path A:
    python scripts/yngine_corpus_to_npz.py \\
        --corpus-dir <shards/> --output ... --use-enhanced-encoding

Usage:
    python scripts/regenerate_npz_with_enhanced_encoder.py \\
        --input  expert_games/yngine_volume.npz \\
        --output expert_games/yngine_volume_15ch_mmap/ \\
        --workers 16

The output directory can be passed directly to:
    python scripts/run_supervised_pretraining.py \\
        --data-dir expert_games/yngine_volume_15ch_mmap/ \\
        --use-enhanced-encoding ...

Disk footprint for the full 13.6M-position corpus:
  - input (compressed npz):  ~10-15 GB
  - output (uncompressed npy): ~10 GB states + small metadata = ~10-12 GB
  - Use a vast.ai instance with ≥80 GB free disk (≥100 GB total).
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

# Repo root on path for direct script run
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yinsh_ml.utils.encoding import StateEncoder  # noqa: E402
from yinsh_ml.utils.enhanced_encoding import EnhancedStateEncoder  # noqa: E402

logger = logging.getLogger("regen_15ch")


# --- worker -----------------------------------------------------------------
# Encoders are stateful (precomputed maps, optional stats); construct once
# per worker, not per call. multiprocessing.Pool's `initializer` runs once
# per worker so this is the right hook.

_basic_enc: StateEncoder | None = None
_enh_enc: EnhancedStateEncoder | None = None


def _init_worker() -> None:
    """Per-worker setup. Constructs the two encoders once."""
    global _basic_enc, _enh_enc
    _basic_enc = StateEncoder()
    _enh_enc = EnhancedStateEncoder()


def _reencode_chunk(chunk: np.ndarray) -> np.ndarray:
    """Decode each 6ch state in `chunk`, re-encode to 15ch.

    Args:
        chunk: (M, 6, 11, 11) float32 array of 6-channel encoded states.

    Returns:
        (M, 15, 11, 11) float32 array of 15-channel re-encoded states.
        Channel 13 (turn_number) will be all zeros — see module docstring.
    """
    assert _basic_enc is not None and _enh_enc is not None, "worker not init'd"
    out = np.zeros((len(chunk), 15, 11, 11), dtype=np.float32)
    for i, state_6ch in enumerate(chunk):
        gs = _basic_enc.decode_state(state_6ch.astype(np.float32))
        out[i] = _enh_enc.encode_state(gs)
    return out


# --- driver -----------------------------------------------------------------


def regenerate(input_path: Path, output_dir: Path, workers: int,
               chunk_size: int, max_positions: int | None = None) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    logger.info(f"Opening {input_path} ({input_path.stat().st_size / 1e9:.2f} GB)")

    # Load input lazily where possible. np.load on .npz is lazy per-key.
    with np.load(input_path) as data:
        keys = list(data.keys())
        logger.info(f"  arrays in input: {keys}")
        if 'states' not in keys:
            raise ValueError(f"input npz missing 'states' key (found: {keys})")
        states_in = data['states']
        if states_in.ndim != 4 or states_in.shape[1] != 6:
            raise ValueError(
                f"input states have shape {states_in.shape} — "
                f"expected (N, 6, 11, 11). This script regenerates 6→15 "
                f"channels; input must be 6-channel."
            )
        N = states_in.shape[0]
        if max_positions is not None:
            N = min(N, max_positions)
            states_in = states_in[:N]
            logger.info(f"  truncated to {N} positions (--max-positions)")
        else:
            logger.info(f"  total positions: {N:,d}")

        policy_indices = data['policy_indices'][:N].astype(np.int32)
        values = data['values'][:N].astype(np.float32)
        total_moves = int(data['total_moves'].item()) if 'total_moves' in keys else 7433
        # Load `states` fully into RAM so we can slice for the pool. The 6ch
        # input is ~6 GB for the 13.6M corpus — fits on the ≥64 GB box.
        # If RAM becomes the bottleneck we could chunk-read via zipfile, but
        # that adds I/O re-amortization complexity for a one-time conversion.
        logger.info(f"  materializing states array into RAM "
                    f"({states_in.nbytes / 1e9:.2f} GB)...")
        states_in_ram = np.ascontiguousarray(states_in[:])

    # Pre-allocate output memmap. Writes go straight to disk, RAM stays flat.
    out_states_path = output_dir / 'states.npy'
    out_states = np.lib.format.open_memmap(
        str(out_states_path), mode='w+',
        dtype=np.float32, shape=(N, 15, 11, 11))
    logger.info(f"  output memmap: {out_states_path} "
                f"({out_states.nbytes / 1e9:.2f} GB)")

    # Process in chunks via Pool.
    n_chunks = (N + chunk_size - 1) // chunk_size
    logger.info(f"  workers={workers}  chunk_size={chunk_size}  n_chunks={n_chunks}")

    # imap to preserve ordering — we write into the memmap by position
    # so the chunk index determines the write offset.
    def chunk_iter():
        for ci in range(n_chunks):
            lo = ci * chunk_size
            hi = min((ci + 1) * chunk_size, N)
            yield states_in_ram[lo:hi]

    t_process = time.time()
    completed = 0
    with mp.Pool(workers, initializer=_init_worker) as pool:
        for ci, chunk_out in enumerate(pool.imap(_reencode_chunk, chunk_iter())):
            lo = ci * chunk_size
            hi = min((ci + 1) * chunk_size, N)
            out_states[lo:hi] = chunk_out
            completed += len(chunk_out)
            if (ci + 1) % max(1, n_chunks // 20) == 0 or ci + 1 == n_chunks:
                elapsed = time.time() - t_process
                rate = completed / elapsed if elapsed > 0 else 0
                eta_s = (N - completed) / rate if rate > 0 else 0
                logger.info(f"    chunk {ci + 1}/{n_chunks}  "
                            f"positions {completed:,d}/{N:,d}  "
                            f"rate={rate:,.0f}/s  eta={eta_s:.0f}s")

    out_states.flush()
    del out_states  # release mmap

    # Save the small metadata arrays
    np.save(output_dir / 'policy_indices.npy', policy_indices)
    np.save(output_dir / 'values.npy', values)
    np.save(output_dir / 'total_moves.npy', np.int32(total_moves))

    # Provenance / channel-13 warning note for downstream consumers
    notes_path = output_dir / 'NOTES.md'
    with open(notes_path, 'w') as fh:
        fh.write(
            "# Regenerated 15-channel corpus (Branch D.2 Path B)\n\n"
            f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
            f"Source npz: {input_path}\n"
            f"Positions: {N:,d}\n"
            f"Total moves (encoder vocab): {total_moves}\n\n"
            "## Known limitation: channel 13 (turn_number) is all zeros\n\n"
            "StateEncoder.decode_state recovers pieces, phase, scores, and\n"
            "rings_placed — but NOT game_state.move_count, which the\n"
            "EnhancedStateEncoder uses for the turn_number plane (channel 13,\n"
            "normalized 0-1 over moves 0..100). Re-encoded positions in this\n"
            "corpus all have channel 13 = 0.\n\n"
            "Channels 0-12 and channel 14 (score_differential) round-trip\n"
            "cleanly.\n\n"
            "If turn_number matters downstream, regenerate the corpus the\n"
            "clean way: from raw yngine shards via\n"
            "`scripts/yngine_corpus_to_npz.py --use-enhanced-encoding`.\n"
        )

    logger.info(f"Wrote outputs to {output_dir}/")
    logger.info(f"  states.npy: ({N}, 15, 11, 11) float32  "
                f"({(N * 15 * 11 * 11 * 4) / 1e9:.2f} GB)")
    logger.info(f"  policy_indices.npy: ({N},) int32")
    logger.info(f"  values.npy: ({N},) float32")
    logger.info(f"  total_moves.npy: scalar int32 = {total_moves}")
    logger.info(f"  NOTES.md: provenance + channel-13 warning")
    logger.info(f"Total wall time: {time.time() - t0:.1f}s")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', type=Path, required=True,
                   help='Input 6-channel npz (e.g. expert_games/yngine_volume.npz)')
    p.add_argument('--output', type=Path, required=True,
                   help='Output directory for mmap-compatible .npy files')
    p.add_argument('--workers', type=int, default=mp.cpu_count(),
                   help=f'Parallel workers (default: cpu_count={mp.cpu_count()})')
    p.add_argument('--chunk-size', type=int, default=2048,
                   help='Positions per worker chunk (default: 2048)')
    p.add_argument('--max-positions', type=int, default=None,
                   help='Cap on positions to process (debug; default: all)')
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    try:
        regenerate(args.input, args.output, args.workers,
                   args.chunk_size, args.max_positions)
    except Exception as e:
        logger.error(f"regeneration failed: {e}", exc_info=args.verbose)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
