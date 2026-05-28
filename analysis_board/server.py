"""Flask backend for the YINSH analysis board.

Serves a single-page app that lets you compose arbitrary positions and ask
the network (and optionally MCTS) what it thinks. Stateless: every
``POST /api/evaluate`` rebuilds a GameState from the request body.

Run with::

    python analysis_board/server.py            # http://127.0.0.1:5173
    YNS_DEVICE=cuda python analysis_board/server.py   # force CUDA
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yinsh_ml.game.constants import (  # noqa: E402
    PieceType,
    Player,
    Position,
    RINGS_PER_PLAYER,
    is_valid_position,
)
from yinsh_ml.game.game_state import GameState  # noqa: E402
from yinsh_ml.game.types import GamePhase, Move, MoveType  # noqa: E402
from yinsh_ml.network.wrapper import NetworkWrapper  # noqa: E402
from yinsh_ml.utils.encoding import StateEncoder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("analysis_board")

MODELS_DIR = ROOT / "models"
# Preferred filenames in priority order. The first match in a model_dir wins.
_PT_CANDIDATES = ("best_supervised.pt", "best.pt", "final.pt", "supervised_final.pt")
DEVICE = os.environ.get("YNS_DEVICE")  # None = auto-detect

# Hard cap on MCTS sims per request — protects a shared/public deployment
# from a single client queuing a 25+ second eval. Default 0 = no cap (local
# dev). On the deployed Mac mini we set YNS_MAX_NUM_SIMS=1600 via launchd
# so a friend on the public URL can't accidentally queue 3200-sim runs.
MAX_NUM_SIMS = int(os.environ.get("YNS_MAX_NUM_SIMS", "0"))

# Serialize MCTS searches across concurrent requests. The MCTS instances in
# _mcts_cache are shared by all callers hitting the same (model_id, num_sims,
# ...) tuple, so two concurrent users would race on reset_tree() / search() /
# _cached_root. Enforce single-MCTS-at-a-time: queue requests behind whoever
# is currently searching. Acceptable at the ~5-user scale (worst case 5 *
# 1600-sim eval ≈ 1 minute for the last user).
_mcts_lock = threading.Lock()


def _pick_checkpoint(model_dir: Path) -> Optional[Path]:
    """Pick the best checkpoint from a model directory.

    Order: canonical names > files starting with 'best' > '*_ema.pt' >
    single .pt > most-recent .pt by mtime. Returns None for empty dirs.
    """
    for name in _PT_CANDIDATES:
        pt = model_dir / name
        if pt.exists():
            return pt
    pts = sorted(model_dir.glob("*.pt"))
    if not pts:
        return None
    for pt in pts:
        if pt.name.startswith("best"):
            return pt
    for pt in pts:
        if pt.name.endswith("_ema.pt"):
            return pt
    if len(pts) == 1:
        return pts[0]
    return max(pts, key=lambda p: p.stat().st_mtime)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_models: List[Dict[str, Any]] = []
_wrapper_cache: Dict[str, NetworkWrapper] = {}
_mcts_cache: Dict[Tuple[str, int], Any] = {}
_encoder = StateEncoder()


def discover_models() -> List[Dict[str, Any]]:
    """Scan ``models/`` for one *.pt per subdirectory.

    Picks the first match in ``_PT_CANDIDATES`` order so ``best_supervised.pt``
    wins over ``supervised_final.pt`` when both exist.
    """
    out: List[Dict[str, Any]] = []
    if not MODELS_DIR.exists():
        log.warning("models/ directory not found at %s", MODELS_DIR)
        return out
    for model_dir in sorted(MODELS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        # Skip archived / hidden directories. Use models/_archive/<name>/
        # to park older checkpoints you don't want surfaced in the dropdown
        # without deleting them.
        if model_dir.name.startswith(("_", ".")):
            continue
        pt = _pick_checkpoint(model_dir)
        if pt is None:
            continue
        out.append({
            "id": f"{model_dir.name}/{pt.name}",
            "label": model_dir.name,
            "checkpoint": pt.name,
            "path": str(pt),
        })
    log.info("discovered %d models under %s", len(out), MODELS_DIR)
    return out


def _model_info(model_id: str) -> Dict[str, Any]:
    for m in _models:
        if m["id"] == model_id:
            return m
    raise KeyError(f"unknown model_id: {model_id}")


def get_wrapper(model_id: str) -> NetworkWrapper:
    if model_id not in _wrapper_cache:
        info = _model_info(model_id)
        log.info("loading wrapper for %s", info["path"])
        _wrapper_cache[model_id] = NetworkWrapper(
            model_path=info["path"],
            device=DEVICE,
        )
    return _wrapper_cache[model_id]


def get_mcts(
    model_id: str,
    num_sims: int,
    *,
    c_puct: float = 1.0,
    fpu_reduction: float = 0.25,
    evaluation_mode: str = "pure_neural",
    heuristic_weight: float = 0.5,
):
    """Cached MCTS instance keyed by all params that change search behaviour.

    Cache size is bounded in practice — users explore a small set of
    (model × num_sims × c_puct × fpu × mode) combinations per session.
    Each instance holds a network reference (shared via the wrapper cache)
    and a small policy tree, so memory growth per cache entry is modest.
    """
    key = (
        model_id, num_sims,
        round(float(c_puct), 3),
        round(float(fpu_reduction), 3),
        str(evaluation_mode),
        round(float(heuristic_weight), 3),
    )
    if key not in _mcts_cache:
        # Lazy import — MCTS pulls a wide chain of training-side deps.
        from yinsh_ml.training.self_play import MCTS  # noqa: WPS433

        wrapper = get_wrapper(model_id)
        heuristic_evaluator = None
        if evaluation_mode in ("pure_heuristic", "hybrid"):
            from yinsh_ml.heuristics.evaluator import YinshHeuristics  # noqa: WPS433
            # Per evaluator.py:60 docstring: MCTS callers should disable
            # forced-sequence detection since MCTS does the lookahead itself,
            # ~30× heuristic speedup with no search-quality loss.
            heuristic_evaluator = YinshHeuristics(enable_forced_sequence_detection=False)
        log.info(
            "constructing MCTS(model=%s, sims=%d, c_puct=%.2f, fpu=%.2f, mode=%s, hw=%.2f)",
            model_id, num_sims, c_puct, fpu_reduction, evaluation_mode, heuristic_weight,
        )
        _mcts_cache[key] = MCTS(
            network=wrapper,
            evaluation_mode=evaluation_mode,
            heuristic_evaluator=heuristic_evaluator,
            heuristic_weight=heuristic_weight,
            num_simulations=num_sims,
            late_simulations=num_sims,
            simulation_switch_ply=999,
            c_puct=c_puct,
            fpu_reduction=fpu_reduction,
            initial_temp=1.0,
            final_temp=1.0,
            annealing_steps=1,
            dirichlet_alpha=0.0,
            epsilon_mix_start=0.0,
            epsilon_mix_end=0.0,
            # Subtree reuse must be ENABLED for the analysis board so the
            # root tree survives `search()` and we can extract principal
            # variations afterward. We compensate by calling
            # `mcts.reset_tree()` before every /api/evaluate search so each
            # call is from a fresh root — no stale subtree carryover.
            enable_subtree_reuse=True,
            mcts_metrics=None,
        )
    return _mcts_cache[key]


def _extract_pv_for_move(root, top_move, depth: int = 5, min_visits: int = 10):
    """Extract the principal variation starting with ``top_move``.

    Returns a list of (Move, visit_count) tuples — the chosen top move first,
    then descending through its subtree by highest-visit child. Stops at
    ``depth`` plies or when a node's best child has fewer than ``min_visits``
    visits (search hasn't explored enough to trust that step).
    """
    pv = []
    first_child = root.children.get(top_move)
    if first_child is None:
        return pv
    pv.append((top_move, int(first_child.visit_count)))
    current = first_child
    for _ in range(depth - 1):
        if not current.children:
            break
        best_move, best_child = max(
            current.children.items(),
            key=lambda kv: kv[1].visit_count,
        )
        if best_child.visit_count < min_visits:
            break
        pv.append((best_move, int(best_child.visit_count)))
        current = best_child
    return pv


def _serialize_pv(base_state: GameState, pv_steps):
    """For each PV step, apply the move to a copy of the state and emit
    a serialized position-after payload. The frontend uses these to swap
    the board to each step instantly when the user walks the line.

    The Move objects from MCTS may have ``player=None`` (engine internals
    don't always populate it). Replace with the state's current player
    before applying, so make_move's player check passes.
    """
    out = []
    if not pv_steps:
        return out
    s = base_state.copy()
    for move, visits in pv_steps:
        # Force the move's player to match the state's current player —
        # MCTS may have stored a generic Move with player=None.
        from dataclasses import replace as _replace
        moving_player = s.current_player
        applied_move = _replace(move, player=moving_player)
        try:
            ok = s.make_move(applied_move)
        except Exception:  # noqa: BLE001
            break
        if not ok:
            break
        out.append({
            "move": _serialize_move(applied_move),
            "visits": int(visits),
            "player": moving_player.name,
            "position_after": _serialize_state(s),
        })
    return out


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------

_PIECE_TYPE_MAP = {
    "WHITE_RING": PieceType.WHITE_RING,
    "BLACK_RING": PieceType.BLACK_RING,
    "WHITE_MARKER": PieceType.WHITE_MARKER,
    "BLACK_MARKER": PieceType.BLACK_MARKER,
}


def build_state(payload: Dict[str, Any]) -> GameState:
    """Construct a GameState from the JSON payload.

    Raises ``ValueError`` on schema problems; downstream legality is checked
    by calling ``get_valid_moves`` after construction.

    The optional ``move_maker`` field preserves the engine's internal
    ``_move_maker`` across the three-move capture sequence (MOVE_RING →
    REMOVE_MARKERS → REMOVE_RING). The engine sets this when a row-completion
    sequence begins and reads it at the end of RING_REMOVAL to decide whose
    turn comes next (the opponent of the original mover). Because the server
    rebuilds GameState from JSON on every /api/move call, this field would be
    lost without explicit threading — silently breaking the turn-flip at the
    end of every capture. See /tmp/test_stateless_capture.py for the
    reproducer.
    """
    gs = GameState()

    side = str(payload.get("side_to_move", "WHITE")).upper()
    if side not in {"WHITE", "BLACK"}:
        raise ValueError(f"side_to_move must be WHITE or BLACK, got {side!r}")
    gs.current_player = Player[side]

    phase_str = str(payload.get("phase", "MAIN_GAME")).upper()
    if phase_str not in GamePhase.__members__:
        raise ValueError(f"unknown phase {phase_str!r}")
    gs.phase = GamePhase[phase_str]

    # Restore _move_maker if the payload carries it (mid-capture-sequence states).
    mm = payload.get("move_maker")
    if mm is not None:
        mm_str = str(mm).upper()
        if mm_str not in {"WHITE", "BLACK"}:
            raise ValueError(f"move_maker must be WHITE or BLACK, got {mm!r}")
        gs._move_maker = Player[mm_str]

    scores = payload.get("scores") or {}
    gs.white_score = int(scores.get("WHITE", 0))
    gs.black_score = int(scores.get("BLACK", 0))

    pieces = payload.get("pieces") or []
    white_rings_on_board = 0
    black_rings_on_board = 0
    for entry in pieces:
        pos_str = str(entry["pos"])
        piece_name = str(entry["piece"]).upper()
        if piece_name not in _PIECE_TYPE_MAP:
            raise ValueError(f"unknown piece type {piece_name!r}")
        pos = Position.from_string(pos_str)
        ok = gs.board.place_piece(pos, _PIECE_TYPE_MAP[piece_name])
        if not ok:
            raise ValueError(f"could not place {piece_name} at {pos_str}")
        if piece_name == "WHITE_RING":
            white_rings_on_board += 1
        elif piece_name == "BLACK_RING":
            black_rings_on_board += 1

    gs.rings_placed[Player.WHITE] = white_rings_on_board + gs.white_score
    gs.rings_placed[Player.BLACK] = black_rings_on_board + gs.black_score

    if gs.rings_placed[Player.WHITE] > RINGS_PER_PLAYER:
        raise ValueError(
            f"WHITE has too many rings: {white_rings_on_board} on board + "
            f"{gs.white_score} captured > {RINGS_PER_PLAYER}"
        )
    if gs.rings_placed[Player.BLACK] > RINGS_PER_PLAYER:
        raise ValueError(
            f"BLACK has too many rings: {black_rings_on_board} on board + "
            f"{gs.black_score} captured > {RINGS_PER_PLAYER}"
        )
    return gs


def _best_move_value(wrapper: NetworkWrapper, state: GameState, top_move) -> Optional[float]:
    """Network's value estimate AFTER playing ``top_move``, in original side's POV.

    The headline "value" we display (mcts root.Q) is a visit-weighted average
    of every backed-up rollout — at high sim budgets this drifts toward zero
    because the long tail of suboptimal moves contributes mass. This function
    returns a complementary number: "if we play the move MCTS picked, what
    does the network think the resulting position is worth?" Much closer to
    the tournament-style "is this winning" intuition.

    Returns None if make_move fails (illegal move snuck through validation).
    """
    s = state.copy()
    try:
        if not s.make_move(top_move):
            return None
    except Exception:  # noqa: BLE001
        return None
    if s.is_terminal():
        # Score-delta is the cleanest terminal signal — whoever has more
        # captured rows won. Encode as +1 (state's side won), -1 (lost), 0 (draw).
        diff = (s.white_score - s.black_score)
        if state.current_player == Player.WHITE:
            return float(np.sign(diff))
        return float(np.sign(-diff))
    _, value_t = wrapper.predict_from_state(s)
    raw = float(value_t.detach().cpu().reshape(-1)[0].item())
    # `predict_from_state` returns value in the new state's POV. If the
    # mover changed (the common case for MOVE_RING in MAIN_GAME), flip
    # to recover the original side's POV. YINSH capture sequences keep the
    # same player on move for 3+ plies — don't flip in that case.
    if s.current_player == state.current_player:
        return raw
    return -raw


def _serialize_move(move) -> Dict[str, Any]:
    return {
        "type": move.type.name,
        "source": str(move.source) if move.source else None,
        "destination": str(move.destination) if move.destination else None,
        "markers": [str(p) for p in move.markers] if move.markers else None,
        "description": str(move),
    }


def _serialize_state(gs: GameState) -> Dict[str, Any]:
    """GameState → the same payload shape build_state consumes (round-trip)."""
    pieces: List[Dict[str, str]] = []
    for col in "ABCDEFGHIJK":
        for row in range(1, 12):
            pos = Position(col, row)
            if not is_valid_position(pos):
                continue
            piece = gs.board.get_piece(pos)
            if piece is None or piece == PieceType.EMPTY:
                continue
            pieces.append({"pos": str(pos), "piece": piece.name})
    out: Dict[str, Any] = {
        "pieces": pieces,
        "phase": gs.phase.name,
        "side_to_move": gs.current_player.name,
        "scores": {"WHITE": int(gs.white_score), "BLACK": int(gs.black_score)},
    }
    # Surface _move_maker for round-trip in mid-capture-sequence states.
    # Always emit (null when not in a sequence) so the frontend can
    # unconditionally pass it back without nullable-key gymnastics.
    out["move_maker"] = gs._move_maker.name if gs._move_maker else None
    return out


def _construct_move(move_spec: Dict[str, Any], player: Player) -> Move:
    """Construct a Move from a JSON spec. Raises ValueError on schema errors."""
    type_name = str(move_spec.get("type", "")).upper()
    if type_name not in MoveType.__members__:
        raise ValueError(f"unknown move type {type_name!r}")
    mtype = MoveType[type_name]
    src = move_spec.get("source")
    dst = move_spec.get("destination")
    markers_raw = move_spec.get("markers")
    return Move(
        type=mtype,
        player=player,
        source=Position.from_string(src) if src else None,
        destination=Position.from_string(dst) if dst else None,
        markers=tuple(Position.from_string(m) for m in markers_raw) if markers_raw else None,
    )


def _legal_moves_payload(valid_moves: List[Move]) -> List[Dict[str, Any]]:
    """Frontend-friendly list of legal moves (same schema as _serialize_move)."""
    return [_serialize_move(m) for m in valid_moves]


def _match_entry_to_move(entry: Dict[str, Any], candidate_moves: List) -> Optional[Any]:
    """Find the Move object in `candidate_moves` that corresponds to a
    serialized top-moves entry (matched by type + source + destination + markers).
    Used both for PV traversal (against root.children) and best_move_value
    computation (against valid_moves)."""
    target_type = entry.get("type")
    target_src = entry.get("source")
    target_dst = entry.get("destination")
    target_markers = entry.get("markers")
    target_marker_set = set(target_markers) if target_markers else None
    for m in candidate_moves:
        if m.type.name != target_type:
            continue
        if target_src is not None and str(m.source) != target_src:
            continue
        if target_dst is not None and (m.destination is None or str(m.destination) != target_dst):
            continue
        if target_marker_set is not None:
            mset = set(str(p) for p in (m.markers or ()))
            if mset != target_marker_set:
                continue
        return m
    return None


def _top_moves(
    probs: np.ndarray,
    valid_moves: List,
    top_k: int,
    include_visits: bool = False,
    visit_counts: Optional[Dict[int, int]] = None,
) -> List[Dict[str, Any]]:
    pairs: List[Tuple[int, Any]] = []
    for m in valid_moves:
        idx = _encoder.move_to_index(m)
        pairs.append((idx, m))
    pairs.sort(key=lambda x: -float(probs[x[0]]))
    out = []
    for idx, m in pairs[:top_k]:
        entry = _serialize_move(m)
        entry["prob"] = float(probs[idx])
        if include_visits and visit_counts is not None:
            entry["visits"] = int(visit_counts.get(idx, 0))
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder="static", static_url_path="")


@app.route("/")
def index():  # type: ignore[no-untyped-def]
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/models")
def api_models():  # type: ignore[no-untyped-def]
    return jsonify(_models)


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():  # type: ignore[no-untyped-def]
    payload = request.get_json(force=True, silent=False) or {}
    model_id = payload.get("model_id")
    if not model_id:
        return jsonify({"ok": False, "errors": ["model_id is required"]}), 400

    num_sims = int(payload.get("num_sims", 0))
    capped_from = None
    if MAX_NUM_SIMS > 0 and num_sims > MAX_NUM_SIMS:
        capped_from = num_sims
        num_sims = MAX_NUM_SIMS
        log.info("capping num_sims %d → %d (YNS_MAX_NUM_SIMS)", capped_from, num_sims)
    top_k = int(payload.get("top_k", 8))
    # Advanced MCTS knobs — defaults match training so the headline reading
    # reflects what the trained agent "thinks." Override to stress-test the
    # position from a different angle.
    c_puct = float(payload.get("c_puct", 1.0))
    fpu_reduction = float(payload.get("fpu_reduction", 0.25))
    evaluation_mode = str(payload.get("evaluation_mode", "pure_neural"))
    if evaluation_mode not in ("pure_neural", "pure_heuristic", "hybrid"):
        return jsonify({"ok": False, "errors": [
            f"evaluation_mode must be pure_neural / pure_heuristic / hybrid, got {evaluation_mode!r}"
        ]}), 400
    heuristic_weight = float(payload.get("heuristic_weight", 0.5))

    # Build state
    try:
        gs = build_state(payload)
    except (KeyError, ValueError) as e:
        return jsonify({"ok": False, "errors": [str(e)]}), 200

    # Validate
    try:
        valid_moves = gs.get_valid_moves()
    except Exception as e:  # noqa: BLE001
        return jsonify({"ok": False, "errors": [f"invalid position: {e}"]}), 200

    if not valid_moves:
        return jsonify({
            "ok": False,
            "errors": ["no legal moves available from this position"],
        }), 200

    # Evaluate
    if num_sims > 0:
        mcts = get_mcts(
            model_id, num_sims,
            c_puct=c_puct,
            fpu_reduction=fpu_reduction,
            evaluation_mode=evaluation_mode,
            heuristic_weight=heuristic_weight,
        )
        # Hold the lock from reset_tree through PV extraction — the cached
        # MCTS instance + its _cached_root are shared across users, so a
        # concurrent search would race with this one's tree state. Acquired
        # here, released either on the MCTS-failure early-return path below
        # or in the PV extraction's `finally` at the end of this branch.
        _mcts_lock.acquire()
        # Reset before each call — subtree reuse is enabled so we can read
        # the tree post-search, but each /api/evaluate is logically a fresh
        # position; we don't want last search's tree biasing this one.
        try:
            mcts.reset_tree()
        except AttributeError:
            mcts._cached_root = None  # fallback for older engine versions
        except Exception as e:  # noqa: BLE001
            log.exception("MCTS reset_tree raised")
            _mcts_lock.release()
            return jsonify({"ok": False, "errors": [f"MCTS reset failed: {e}"]}), 500
        try:
            probs = mcts.search(gs, move_number=1)
        except Exception as e:  # noqa: BLE001
            log.exception("MCTS search failed")
            _mcts_lock.release()
            return jsonify({"ok": False, "errors": [f"MCTS failed: {e}"]}), 500
        value = float(getattr(mcts, "last_root_value", 0.0))
        # MCTS with enable_subtree_reuse=False clears root.children after
        # search, so we can't read visit_count directly. Back-compute from
        # the returned distribution: with initial_temp=final_temp=1.0, the
        # MCTS engine returns probs proportional to visit counts, so
        # visits ≈ round(prob * num_sims). Off by at most 1 in edge cases.
        visit_counts: Dict[int, int] = {}
        for m in valid_moves:
            i = _encoder.move_to_index(m)
            visit_counts[i] = int(round(float(probs[i]) * num_sims))
        top = _top_moves(probs, valid_moves, top_k, include_visits=True, visit_counts=visit_counts)
        mode = "mcts"
        # Best-move-value: a complementary signal to root.Q. Compute for
        # every entry so the UI can flag opposite-sign divergences per-row
        # (where MCTS picked move N but the network value at the resulting
        # position disagrees with the search-averaged root.Q).
        wrapper_for_bm = get_wrapper(model_id)
        for entry in top:
            matched = _match_entry_to_move(entry, valid_moves)
            entry["best_move_value"] = (
                _best_move_value(wrapper_for_bm, gs, matched) if matched is not None else None
            )
        # Headline best_move_value = the top-1 move's. Convenient for the
        # existing single-value display; per-row values are also available.
        best_value = top[0].get("best_move_value") if top else None
        top_move = max(valid_moves, key=lambda m: float(probs[_encoder.move_to_index(m)]))
        # Principal variations — extract from the live MCTS tree before the
        # next call resets it. Attach each PV to its corresponding top_moves
        # entry by matching move_to_index.
        try:
            root = mcts._cached_root  # noqa: SLF001
            if root is not None:
                # Map move_idx → original Move object in root.children (lets us
                # look up the right Move for each top_moves entry).
                idx_to_root_move = {}
                for m in root.children.keys():
                    try:
                        idx_to_root_move[_encoder.move_to_index(m)] = m
                    except Exception:  # noqa: BLE001
                        continue
                for entry in top:
                    # Reconstruct the move's index from the description fields.
                    # We don't carry idx in entry, so re-derive from the
                    # serialized move; cheaper to find by move source/dest.
                    matched_move = None
                    for m in root.children.keys():
                        if m.type.name != entry["type"]:
                            continue
                        if entry.get("source") and str(m.source) != entry["source"]:
                            continue
                        if entry.get("destination") and (m.destination is None or str(m.destination) != entry["destination"]):
                            continue
                        if entry.get("markers"):
                            mset = set(str(p) for p in (m.markers or ()))
                            if mset != set(entry["markers"]):
                                continue
                        matched_move = m
                        break
                    if matched_move is None:
                        entry["principal_variation"] = []
                        continue
                    # depth=8, min_visits=4: explore deeper into the tree but
                    # cut off when search hasn't allocated enough budget to
                    # trust the "best child" pick. The frontend displays
                    # visit counts per ply so the user can spot when plies
                    # are getting tentative (single-digit visits).
                    pv_steps = _extract_pv_for_move(root, matched_move, depth=8, min_visits=4)
                    entry["principal_variation"] = _serialize_pv(gs, pv_steps)
        except Exception as e:  # noqa: BLE001
            log.warning("PV extraction failed: %s", e)
            for entry in top:
                entry.setdefault("principal_variation", [])
        finally:
            # Release the MCTS lock after PV extraction — the next queued
            # request can now reset_tree() and run its own search.
            _mcts_lock.release()
    else:
        wrapper = get_wrapper(model_id)
        try:
            move_probs_t, value_t = wrapper.predict_from_state(gs)
        except Exception as e:  # noqa: BLE001
            log.exception("network predict failed")
            return jsonify({"ok": False, "errors": [f"predict failed: {e}"]}), 500
        probs_np = move_probs_t.detach().cpu().numpy()
        if probs_np.ndim > 1:
            probs_np = probs_np[0]
        # Mask to valid moves, renormalize
        masked = np.zeros_like(probs_np)
        for m in valid_moves:
            i = _encoder.move_to_index(m)
            masked[i] = probs_np[i]
        s = masked.sum()
        probs = masked / s if s > 1e-12 else masked
        value = float(value_t.detach().cpu().reshape(-1)[0].item())
        top = _top_moves(probs, valid_moves, top_k, include_visits=False)
        mode = "policy"
        # Per-entry best_move_value (raw-policy mode also benefits — lets
        # the user see which low-ranked moves still have positive after-value).
        for entry in top:
            matched = _match_entry_to_move(entry, valid_moves)
            entry["best_move_value"] = (
                _best_move_value(wrapper, gs, matched) if matched is not None else None
            )
        best_value = top[0].get("best_move_value") if top else None
        top_move = max(valid_moves, key=lambda m: float(probs[_encoder.move_to_index(m)]))

    return jsonify({
        "ok": True,
        "mode": mode,
        "value": value,
        "best_move_value": best_value,
        "side_to_move": gs.current_player.name,
        "phase": gs.phase.name,
        "rings_placed": {
            "WHITE": int(gs.rings_placed[Player.WHITE]),
            "BLACK": int(gs.rings_placed[Player.BLACK]),
        },
        "num_valid_moves": len(valid_moves),
        "top_moves": top,
        "legal_moves": _legal_moves_payload(valid_moves),
    })


@app.route("/api/move", methods=["POST"])
def api_move():  # type: ignore[no-untyped-def]
    """Apply a move to a position; return the new position + new legal moves.

    Stateless — the caller (frontend) is responsible for tracking history.
    """
    payload = request.get_json(force=True, silent=False) or {}
    move_spec = payload.pop("move", None)
    if not move_spec:
        return jsonify({"ok": False, "errors": ["move is required"]}), 400

    # Build state, then construct + validate move.
    try:
        gs = build_state(payload)
    except (KeyError, ValueError) as e:
        return jsonify({"ok": False, "errors": [f"position: {e}"]}), 200

    try:
        move = _construct_move(move_spec, gs.current_player)
    except (KeyError, ValueError) as e:
        return jsonify({"ok": False, "errors": [f"move: {e}"]}), 200

    try:
        valid_moves = gs.get_valid_moves()
    except Exception as e:  # noqa: BLE001
        return jsonify({"ok": False, "errors": [f"invalid position: {e}"]}), 200

    if move not in valid_moves:
        return jsonify({
            "ok": False,
            "errors": [f"illegal move from this position: {move}"],
        }), 200

    # Apply.
    try:
        applied = gs.make_move(move)
    except Exception as e:  # noqa: BLE001
        log.exception("make_move raised")
        return jsonify({"ok": False, "errors": [f"make_move raised: {e}"]}), 500
    if not applied:
        return jsonify({"ok": False, "errors": ["make_move returned False"]}), 200

    new_payload = _serialize_state(gs)
    new_valid: List[Move] = []
    game_over = gs.is_terminal()
    winner: Optional[str] = None
    if game_over:
        if gs.white_score > gs.black_score:
            winner = "WHITE"
        elif gs.black_score > gs.white_score:
            winner = "BLACK"
        # equal scores in a terminal state = stalemate-style draw; leave None
    else:
        new_valid = gs.get_valid_moves()

    return jsonify({
        "ok": True,
        "new_position": new_payload,
        "applied_move": _serialize_move(move),
        "legal_moves": _legal_moves_payload(new_valid),
        "game_over": game_over,
        "winner": winner,
    })


def main() -> None:
    global _models  # noqa: PLW0603
    _models = discover_models()
    host = os.environ.get("YNS_HOST", "127.0.0.1")
    port = int(os.environ.get("YNS_PORT", "5173"))
    log.info("starting analysis board on http://%s:%d", host, port)
    app.run(host=host, port=port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
