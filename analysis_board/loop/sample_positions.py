"""Sample positions from a replay buffer, save as JSONL ready for measure.py.

Reads ``replay_buffer.pkl.gz`` (15-channel encoded state tensors + metadata),
decodes a random sample back to GameStates, and serializes each as the same
JSON payload the analysis-board server already accepts (``pieces``, ``phase``,
``side_to_move``, ``scores``). The output is portable: feed it to
``measure.py``, or POST individual lines to ``/api/evaluate``, or replay them
in the UI by hand.

Usage:
    python analysis_board/loop/sample_positions.py \
        --replay-buffer experiments/.../replay_buffer.pkl.gz \
        --n 200 \
        --out analysis_board/loop/runs/<ts>/positions.jsonl
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yinsh_ml.game.constants import PieceType, Player  # noqa: E402
from yinsh_ml.game.game_state import GameState  # noqa: E402
from yinsh_ml.utils.enhanced_encoding import EnhancedStateEncoder  # noqa: E402
from yinsh_ml.utils.encoding import StateEncoder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("sample_positions")


def state_to_payload(state: GameState) -> Dict[str, Any]:
    """Serialize a GameState as the analysis-board evaluate-payload schema."""
    pieces: List[Dict[str, str]] = []
    for col in "ABCDEFGHIJK":
        for row in range(1, 12):
            from yinsh_ml.game.constants import Position, is_valid_position
            pos = Position(col, row)
            if not is_valid_position(pos):
                continue
            piece = state.board.get_piece(pos)
            if piece is None or piece == PieceType.EMPTY:
                continue
            pieces.append({"pos": str(pos), "piece": piece.name})
    return {
        "pieces": pieces,
        "phase": state.phase.name,
        "side_to_move": state.current_player.name,
        "scores": {
            "WHITE": int(state.white_score),
            "BLACK": int(state.black_score),
        },
    }


def _stable_id(payload: Dict[str, Any]) -> str:
    """Deterministic ID for a position — content hash of the canonical form."""
    canon = json.dumps(
        {
            "pieces": sorted(payload["pieces"], key=lambda p: p["pos"]),
            "phase": payload["phase"],
            "side_to_move": payload["side_to_move"],
            "scores": payload["scores"],
        },
        sort_keys=True,
    )
    return hashlib.md5(canon.encode()).hexdigest()[:12]


def load_replay(path: Path) -> Dict[str, Any]:
    log.info("loading replay buffer: %s", path)
    with gzip.open(path, "rb") as f:
        buf = pickle.load(f)
    required = {"states", "phases", "move_numbers", "values"}
    missing = required - set(buf.keys())
    if missing:
        raise ValueError(f"replay buffer missing keys: {missing}")
    log.info("replay buffer has %d positions", len(buf["states"]))
    return buf


def pick_encoder(state_tensor: np.ndarray):
    """Return the encoder whose channel count matches the tensor's."""
    ch = int(state_tensor.shape[0])
    if ch == 6:
        return StateEncoder()
    if ch == 15:
        return EnhancedStateEncoder()
    raise ValueError(f"unsupported channel count: {ch}")


def sample(
    buf: Dict[str, Any],
    n: int,
    *,
    phase_filter: Iterable[str] | None = None,
    min_move_number: int = 0,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Randomly sample N positions matching filters."""
    total = len(buf["states"])
    indices = list(range(total))
    if phase_filter:
        wanted = set(phase_filter)
        indices = [i for i in indices if buf["phases"][i] in wanted]
    if min_move_number > 0:
        indices = [i for i in indices if buf["move_numbers"][i] >= min_move_number]
    log.info("%d candidates after filters", len(indices))
    if not indices:
        return []

    rng = random.Random(seed)
    chosen = rng.sample(indices, min(n, len(indices)))
    chosen.sort()  # deterministic order in the output

    encoder = pick_encoder(buf["states"][chosen[0]])
    out: List[Dict[str, Any]] = []
    for idx in chosen:
        tensor = buf["states"][idx]
        try:
            gs = encoder.decode_state(tensor)
        except Exception as e:  # noqa: BLE001
            log.warning("decode failed at idx %d: %s", idx, e)
            continue
        payload = state_to_payload(gs)
        payload["meta"] = {
            "replay_index": int(idx),
            "phase_label": buf["phases"][idx],
            "move_number": int(buf["move_numbers"][idx]),
            "self_play_target_value": float(buf["values"][idx]),
        }
        payload["id"] = _stable_id(payload)
        out.append(payload)
    log.info("decoded %d / %d positions", len(out), len(chosen))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--replay-buffer", required=True, type=Path)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--phase", action="append", default=None,
                   help="filter to specific phases (repeatable)")
    p.add_argument("--min-move", type=int, default=0,
                   help="skip positions with move_number < this")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    buf = load_replay(args.replay_buffer)
    positions = sample(
        buf, args.n,
        phase_filter=args.phase, min_move_number=args.min_move, seed=args.seed,
    )
    with args.out.open("w") as f:
        for pos in positions:
            f.write(json.dumps(pos) + "\n")
    log.info("wrote %d positions to %s", len(positions), args.out)


if __name__ == "__main__":
    main()
