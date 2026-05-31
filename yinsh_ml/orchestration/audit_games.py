"""Record real candidate self-play games so the offense-only check can fire.

Training self-play writes training tensors, not replayable move-records, so the
offense-only-equilibrium check skips on real candidates (it has no trajectories to
read). This plays a small batch of self-play games with the trained candidate
network through the real MCTS engine and records them via the standard
``GameRecorder`` → ``ParquetDataStorage`` pipeline, into ``<output_dir>/parquet_data``
— exactly where ``_default_panel_input`` looks. The result: the offense-only
detector runs on the candidate's *own* play.

It's an opt-in audit (``--audit-games N`` on ``yinsh-track schedule``): a few games
is enough signal, and each game is an MCTS self-play game, so keep N small.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _auto_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def record_candidate_games(
    checkpoint: str,
    num_games: int,
    output_dir: str,
    device: Optional[str] = None,
    num_simulations: int = 50,
    max_moves: int = 200,
    seed: int = 0,
) -> Optional[str]:
    """Play ``num_games`` MCTS self-play games with the candidate and record them.

    Writes replayable parquet to ``<output_dir>/parquet_data`` and returns that path
    (or ``None`` on failure — the audit is best-effort and must not break the run).
    Heavy deps are imported lazily so the orchestration package stays light.
    """
    try:
        import numpy as np
        import random
        import torch

        from ..game.game_state import GameState
        from ..network.wrapper import NetworkWrapper
        from ..self_play.data_storage import ParquetDataStorage, StorageConfig
        from ..self_play.game_recorder import GameRecorder
        from ..training.self_play import MCTS
        from ..utils.encoding import StateEncoder

        device = device or _auto_device()
        network = NetworkWrapper(device=device)
        network.load_model(str(checkpoint))
        encoder = StateEncoder()

        storage = ParquetDataStorage(
            StorageConfig(output_dir=str(output_dir), parquet_dir="parquet_data", batch_size=1)
        )
        recorder = GameRecorder(
            output_dir=str(Path(output_dir) / "_audit_scratch"), save_json=False
        )

        recorded = 0
        for g in range(num_games):
            s = seed + g
            random.seed(s)
            np.random.seed(s)
            torch.manual_seed(s)
            # Fresh MCTS per game so no tree state carries across games.
            mcts = MCTS(
                network=network,
                evaluation_mode="pure_neural",
                num_simulations=num_simulations,
                max_depth=max_moves,
            )
            state = GameState()
            recorder.start_game(f"audit_{s:06d}")
            move_number = 0
            while not state.is_terminal() and move_number < max_moves:
                valid = state.get_valid_moves()
                if not valid:
                    break
                # Mirror the tested self-play move selection (self_play._play_game_inner):
                # take the MCTS visit policy, restrict to valid moves, fall back to
                # uniform if it has no mass (avoids a NaN), then temperature-sample.
                move_probs = mcts.search(state, move_number)
                temp = mcts.get_temperature(move_number)
                valid_probs = move_probs[[encoder.move_to_index(m) for m in valid]].astype(np.float64)
                total = valid_probs.sum()
                if total <= 1e-8 or not np.isfinite(total):
                    valid_probs = np.ones(len(valid)) / len(valid)
                else:
                    valid_probs = valid_probs / total
                if temp < 0.01:
                    idx = int(np.argmax(valid_probs))
                else:
                    idx = int(np.random.choice(len(valid), p=valid_probs))
                move = valid[idx]
                recorder.record_turn(state, move, state.current_player)
                state.make_move(move)
                mcts.advance_root(move)
                move_number += 1
            winner = state.get_winner() if state.is_terminal() else None
            record = recorder.end_game(state, winner=winner)
            if record is not None:
                storage.store_game_record(record)
                recorded += 1

        logger.info("Recorded %d candidate audit games under %s", recorded, output_dir)
        return str(Path(output_dir) / "parquet_data")
    except Exception as exc:  # noqa: BLE001 - audit is best-effort
        logger.warning("Candidate game audit failed (%s); offense-only check will skip.", exc)
        return None
