"""Annotator pattern for adding per-turn metadata to a ``GameReplay``.

The replay layer (``game_replay.py``) is intentionally source-agnostic
— it produces ``Board`` snapshots from whatever serialized moves you
hand it (parquet, BGA JSON, in-memory move list, etc.). What metadata
you want to attach to each turn depends on the consumer:

  - Heuristic audit (the offense-only equilibrium investigation):
    the 7 differential heuristic features, threat counts, capture
    events.
  - Neural self-play review: MCTS visit distributions, network value
    predictions, expected vs. realised outcome.
  - Expert game review with a model: what the network would have
    played at each position vs. what the human did, and by how much.
  - Custom: anything else a consumer dreams up.

Rather than baking any one of these into the loader, an ``Annotator``
is just a callable ``(replay, turn_idx, state) -> Dict[str, Any]``.
Run one or more via :func:`annotate`; results land in
``replay.annotations[turn_idx]`` as a flat dict, ready to be displayed
by the dashboard or read by downstream analysis.

This module ships a couple of starter annotators (heuristic features,
threat counting). The expensive ones (network value, MCTS) live close
to those subsystems and are added on demand — see the README for
sketches.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..game.constants import PieceType, Player
from ..game.game_state import GameState
from .game_replay import GameReplay

logger = logging.getLogger(__name__)


# An annotator is a stateless-or-stateful callable that produces a dict
# of metadata for one turn. Stateful annotators can hold their own
# context (e.g. previous scores) via a class or closure. Returning {}
# is fine if the annotator has nothing to add for this turn.
Annotator = Callable[[GameReplay, int, GameState], Dict[str, Any]]


def annotate(
    replay: GameReplay,
    annotators: Iterable[Annotator],
    *,
    overwrite_keys: bool = True,
) -> GameReplay:
    """Run one or more annotators over ``replay``, populating ``replay.annotations``.

    Single forward pass through ``replay.iter_states()`` — all annotators
    run on the same yielded ``GameState`` per turn. Their output dicts
    are merged into a single per-turn annotation dict; with
    ``overwrite_keys=True`` (default) later annotators win on key
    collisions, with ``False`` collisions are logged and the first
    value is kept.

    Mutates ``replay.annotations`` in place AND returns ``replay`` for
    pipeline-style chaining.
    """
    annotators = list(annotators)
    n_turns = len(replay.moves)
    out: List[Dict[str, Any]] = [{} for _ in range(n_turns)]

    for turn_idx, state in replay.iter_states():
        if turn_idx >= n_turns:
            break  # defensive: iter_states could in principle yield extra
        for ann in annotators:
            try:
                produced = ann(replay, turn_idx, state) or {}
            except Exception as e:
                logger.warning(
                    "Annotator %s failed at turn %d (%s); skipping that "
                    "annotator for this turn", ann, turn_idx, e,
                )
                continue
            for k, v in produced.items():
                if k in out[turn_idx] and not overwrite_keys:
                    logger.debug(
                        "Annotation key collision at turn %d (key=%s); "
                        "keeping first value (overwrite_keys=False)",
                        turn_idx, k,
                    )
                    continue
                out[turn_idx][k] = v

    replay.annotations = out
    return replay


# ----------------------------------------------------------------------
# Starter annotators. Cheap; depend only on game-engine + heuristics.
# ----------------------------------------------------------------------
def heuristic_features_annotator(
    *, player: Player = Player.WHITE, feature_keys: Optional[Iterable[str]] = None
) -> Annotator:
    """Annotator that adds differential heuristic features per turn.

    ``player`` controls POV — defaults to White (positive = White
    favoured). ``feature_keys`` optionally filters which features to
    emit; defaults to all.
    """
    # Lazy import — pulls in the heuristics package only if this
    # annotator is actually used.
    from ..heuristics.features import extract_all_features

    keys_filter = set(feature_keys) if feature_keys is not None else None

    def _ann(replay: GameReplay, turn_idx: int, state: GameState) -> Dict[str, Any]:
        feats = extract_all_features(state, player)
        if keys_filter is not None:
            feats = {k: feats[k] for k in feats if k in keys_filter}
        return {f"hf_{k}": float(v) for k, v in feats.items()}

    return _ann


def captures_and_threats_annotator() -> Annotator:
    """Annotator that adds capture events (score-delta) and threat counts.

    Stateful — closure-captures the prior turn's scores to detect
    captures. Per turn emits:

      capture: ``"WHITE" / "BLACK" / ""``
      white_score, black_score: running scores
      white_threats, black_threats: count of length-4 rows
      defensive_miss: opponent had a 4-row at the start of the
        player's turn AND it survived the move
    """
    state_dict = {"prev_white": 0, "prev_black": 0,
                  "prev_white_threats": 0, "prev_black_threats": 0}

    def _threats(state: GameState, marker: PieceType) -> int:
        return sum(1 for r in state.board.find_marker_rows(marker)
                   if r.length == 4)

    def _ann(replay: GameReplay, turn_idx: int, state: GameState) -> Dict[str, Any]:
        capture = ""
        if state.white_score > state_dict["prev_white"]:
            capture = "WHITE"
        elif state.black_score > state_dict["prev_black"]:
            capture = "BLACK"

        white_threats = _threats(state, PieceType.WHITE_MARKER)
        black_threats = _threats(state, PieceType.BLACK_MARKER)

        # state.current_player at state-after-move is whoever moves NEXT,
        # i.e. opponent of the player who just moved.
        opponent_of_mover = state.current_player
        prev_threats = (state_dict["prev_white_threats"]
                        if opponent_of_mover == Player.WHITE
                        else state_dict["prev_black_threats"])
        threats_after = (white_threats if opponent_of_mover == Player.WHITE
                         else black_threats)
        defensive_miss = bool(prev_threats > 0 and threats_after >= prev_threats)

        out = {
            "capture": capture,
            "white_score": state.white_score,
            "black_score": state.black_score,
            "white_threats": white_threats,
            "black_threats": black_threats,
            "defensive_miss": defensive_miss,
        }
        state_dict["prev_white"] = state.white_score
        state_dict["prev_black"] = state.black_score
        state_dict["prev_white_threats"] = white_threats
        state_dict["prev_black_threats"] = black_threats
        return out

    return _ann


__all__ = [
    "Annotator",
    "annotate",
    "captures_and_threats_annotator",
    "heuristic_features_annotator",
]
