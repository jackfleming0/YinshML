"""Flask backend for the YINSH analysis board.

Serves a single-page app that lets you compose arbitrary positions and ask
the network (and optionally MCTS) what it thinks. Stateless: every
``POST /api/evaluate`` rebuilds a GameState from the request body.

Run with::

    python analysis_board/server.py            # http://127.0.0.1:5173
    YNS_DEVICE=cuda python analysis_board/server.py   # force CUDA
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis_board.bga_import import (  # noqa: E402
    BGAImportError,
    cache_load,
    cache_save,
    check_rate_limit,
    parse_url_or_id,
    record_rate_limit,
    replay_to_steps,
)
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

# Aliased on import — both bga_import and screenshot_import export
# `check_rate_limit` / `record_rate_limit` against their own per-IP
# trackers (separate quotas, separate buckets). Without the rename
# the second import would shadow the BGA functions and the BGA
# handler would charge against the screenshot bucket.
from analysis_board.screenshot_import import (  # noqa: E402
    ScreenshotImportError,
    call_claude_vision,
    decode_and_validate_image,
    validate_claude_response,
)
from analysis_board.screenshot_import import (  # noqa: E402
    check_rate_limit as screenshot_check_rate_limit,
    record_rate_limit as screenshot_record_rate_limit,
)

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

# Owner bypass for MAX_NUM_SIMS. When set to a non-empty secret, a request
# carrying the matching token (X-Yns-Owner-Token header, or "owner_token" in
# the JSON body) skips the cap entirely — so the owner can queue arbitrarily
# deep searches (256000 sims if they feel like it) while anonymous public
# visitors stay capped and can't hang the shared, lock-serialized engine.
# Empty default = bypass disabled (no token grants extra budget).
OWNER_TOKEN = os.environ.get("YNS_OWNER_TOKEN", "")

# Serialize all network inference / MCTS search across threads. Two reasons:
# (1) the MCTS instances in _mcts_cache are shared by all callers hitting the
# same (model_id, num_sims, ...) tuple, so concurrent users would race on
# reset_tree() / search() / _cached_root; (2) the NetworkWrapper's TensorPool
# is not thread-safe, so a background async-job search and a request-thread
# eval must never do inference at the same time. Every inference path (sync
# MCTS eval, sync raw-policy eval, and the async worker) holds this lock — one
# search at a time, server-wide. Acceptable at the ~5-user scale.
_mcts_lock = threading.Lock()

# --- Async evaluation jobs ---------------------------------------------------
# Big owner searches (e.g. 128000 sims ≈ 15 min on the singleton search path)
# can't return within the Cloudflare tunnel's ~100s response window, so the
# synchronous /api/evaluate times out with an HTML 524 page. /api/evaluate_async
# instead runs the search on a background daemon thread and reports progress via
# /api/evaluate_result/<job_id>, which the SPA polls ~1Hz. The search still
# holds _mcts_lock (the TensorPool isn't thread-safe — see _run_eval_job), so
# only one search runs at a time; but the lightweight poll endpoint doesn't take
# that lock, so progress keeps flowing while a long job churns.
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()
# Keep finished jobs around briefly so the poller can fetch the result, then
# evict so the dict doesn't grow unbounded.
_JOB_TTL_SECONDS = 600

# Append-only per-day JSONL log of every successful /api/move event. Used
# to reconstruct friend-played games offline for qualitative analysis:
# where did the engine win / lose, what openings show up, what positions
# trip the model up. Path is relative to the repo root (ROOT). Logging is
# best-effort — a write failure must NOT break the response to the user.
GAME_LOG_DIR = ROOT / "analysis_board" / "multiplayer" / "deploy" / "games"

# Cache for BGA Review-mode imports. One file per table id; LRU-evicted by
# the import module so total on-disk footprint stays bounded. Path is server-
# private — exposed only via the Review-mode `/api/import_bga` endpoint, not
# served as static content.
BGA_CACHE_DIR = ROOT / "analysis_board" / "multiplayer" / "bga_imports"

# Default cookies path. ``~/.bga_cookies.json`` is what scripts/bga_fetch.py
# uses; honoring the same default keeps local-dev flow uniform between the
# bulk crawler and the Review-mode endpoint. Override with ``YNS_BGA_COOKIES``.
BGA_COOKIES_PATH = os.environ.get(
    "YNS_BGA_COOKIES",
    str(Path.home() / ".bga_cookies.json"),
)


def _log_move_event(
    payload: Dict[str, Any], move_spec: Optional[Dict[str, Any]], response: Dict[str, Any],
) -> None:
    """Append one /api/move event to today's JSONL file. Captures both the
    pre- and post-move position so a single line is self-contained — no need
    to chain events to reconstruct state. Groups into games via the
    client-supplied ``play_session_id`` (regenerated on Clear / startGame /
    setup→play transitions).
    """
    try:
        GAME_LOG_DIR.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        path = GAME_LOG_DIR / f"{now.strftime('%Y-%m-%d')}.jsonl"
        event = {
            "ts": now.isoformat(),
            "play_session_id": payload.get("play_session_id"),
            "pre_position": {
                "pieces": payload.get("pieces"),
                "phase": payload.get("phase"),
                "side_to_move": payload.get("side_to_move"),
                "scores": payload.get("scores"),
                "move_maker": payload.get("move_maker"),
            },
            "requested_move": move_spec,
            "applied_move": response.get("applied_move"),
            "new_position": response.get("new_position"),
            "game_over": response.get("game_over"),
            "winner": response.get("winner"),
        }
        with open(path, "a") as f:
            f.write(json.dumps(event, separators=(",", ":")) + "\n")
    except Exception as e:  # noqa: BLE001
        log.warning("game-log append failed: %s", e)


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
        _mcts_cache[key] = _construct_mcts(
            model_id, num_sims,
            c_puct=c_puct,
            fpu_reduction=fpu_reduction,
            evaluation_mode=evaluation_mode,
            heuristic_weight=heuristic_weight,
        )
    return _mcts_cache[key]


def _construct_mcts(
    model_id: str,
    num_sims: int,
    *,
    c_puct: float = 1.0,
    fpu_reduction: float = 0.25,
    evaluation_mode: str = "pure_neural",
    heuristic_weight: float = 0.5,
):
    """Build a fresh MCTS instance configured for the analysis board.

    Used directly (un-cached, private to the caller) by the async job runner
    so a long owner search doesn't share ``_cached_root`` / ``_mcts_lock`` with
    the interactive cache — and wrapped by ``get_mcts`` for the cached
    synchronous path.
    """
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
    return MCTS(
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
        # `mcts.reset_tree()` before every search so each call is from a
        # fresh root — no stale subtree carryover.
        enable_subtree_reuse=True,
        mcts_metrics=None,
    )


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


@app.errorhandler(Exception)
def _json_api_errors(e):  # type: ignore[no-untyped-def]
    """Return JSON (not Flask's default HTML page) for any unhandled error on
    an ``/api/*`` route.

    Without this, a server-side 500 hands the SPA an HTML error page and the
    frontend's ``res.json()`` dies with the cryptic "Unexpected token '<',
    \"<!DOCTYPE\"... is not valid JSON". With it, the real exception message
    reaches the user. (Note: this can't catch a *proxy* timeout — e.g. the
    Cloudflare tunnel's ~100s 524 page — since that HTML never originates
    from Flask; in that case the SPA still sees DOCTYPE, which itself confirms
    the request died upstream rather than in the app.)
    """
    status = e.code if isinstance(e, HTTPException) and e.code else 500
    if not request.path.startswith("/api/"):
        # Preserve default behavior for the SPA shell / static assets.
        if isinstance(e, HTTPException):
            return e
        raise e
    if status >= 500:
        log.exception("unhandled error on %s", request.path)
    return jsonify({"ok": False, "errors": [f"{type(e).__name__}: {e}"]}), status


@app.route("/")
def index():  # type: ignore[no-untyped-def]
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/models")
def api_models():  # type: ignore[no-untyped-def]
    return jsonify(_models)


def _mcts_result_from_probs(mcts, probs, gs, valid_moves, model_id, top_k, num_sims):
    """Turn a finished MCTS search into the response's move list.

    Assumes ``mcts.search(...)`` already ran (its tree is in
    ``mcts._cached_root``) and produced ``probs``. Returns
    ``(top_moves, value, top_move)``. Shared verbatim by the synchronous
    ``/api/evaluate`` route and the async job worker so both render identical
    top-move tables, best-move values, and principal variations.
    """
    value = float(getattr(mcts, "last_root_value", 0.0))
    # Back-compute visit counts from the distribution: with
    # initial_temp=final_temp=1.0 the engine returns probs proportional to
    # visit counts, so visits ≈ round(prob * num_sims) (off by at most 1).
    visit_counts: Dict[int, int] = {}
    for m in valid_moves:
        i = _encoder.move_to_index(m)
        visit_counts[i] = int(round(float(probs[i]) * num_sims))
    top = _top_moves(probs, valid_moves, top_k, include_visits=True, visit_counts=visit_counts)
    # Best-move-value: a complementary signal to root.Q, per entry, so the UI
    # can flag opposite-sign divergences (MCTS picked move N but the network
    # value at the resulting position disagrees with search-averaged root.Q).
    wrapper_for_bm = get_wrapper(model_id)
    for entry in top:
        matched = _match_entry_to_move(entry, valid_moves)
        entry["best_move_value"] = (
            _best_move_value(wrapper_for_bm, gs, matched) if matched is not None else None
        )
    top_move = max(valid_moves, key=lambda m: float(probs[_encoder.move_to_index(m)]))
    # Principal variations — read from the live MCTS tree, matching each
    # top_moves entry back to its root child by move descriptor.
    try:
        root = mcts._cached_root  # noqa: SLF001
        if root is not None:
            for entry in top:
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
                # depth=8, min_visits=4: explore deeper but cut off when search
                # hasn't allocated enough budget to trust the best-child pick.
                pv_steps = _extract_pv_for_move(root, matched_move, depth=8, min_visits=4)
                entry["principal_variation"] = _serialize_pv(gs, pv_steps)
    except Exception as e:  # noqa: BLE001
        log.warning("PV extraction failed: %s", e)
        for entry in top:
            entry.setdefault("principal_variation", [])
    return top, value, top_move


def _owner_token_ok(payload) -> bool:
    """True if the request carries the configured owner token (constant-time).

    Empty ``YNS_OWNER_TOKEN`` ⇒ always False (bypass disabled).
    """
    provided = request.headers.get("X-Yns-Owner-Token") or str(payload.get("owner_token", ""))
    return bool(OWNER_TOKEN) and hmac.compare_digest(provided, OWNER_TOKEN)


def _prepare_evaluation(payload, is_owner):
    """Parse + validate an evaluate request, shared by the sync and async paths.

    Returns ``(ctx, None)`` on success, or ``(None, (error_body, status))`` for
    a client-facing failure. ``ctx`` carries the parsed knobs (with the public
    cap already applied unless ``is_owner``) plus the built GameState and its
    legal moves — so both paths apply the cap and owner bypass identically.
    """
    model_id = payload.get("model_id")
    if not model_id:
        return None, ({"ok": False, "errors": ["model_id is required"]}, 400)

    num_sims = int(payload.get("num_sims", 0))
    capped_from = None
    if MAX_NUM_SIMS > 0 and not is_owner and num_sims > MAX_NUM_SIMS:
        capped_from = num_sims
        num_sims = MAX_NUM_SIMS
        log.info("capping num_sims %d → %d (YNS_MAX_NUM_SIMS)", capped_from, num_sims)
    top_k = int(payload.get("top_k", 8))
    # Advanced MCTS knobs — defaults match training so the headline reading
    # reflects what the trained agent "thinks."
    c_puct = float(payload.get("c_puct", 1.0))
    fpu_reduction = float(payload.get("fpu_reduction", 0.25))
    evaluation_mode = str(payload.get("evaluation_mode", "pure_neural"))
    if evaluation_mode not in ("pure_neural", "pure_heuristic", "hybrid"):
        return None, ({"ok": False, "errors": [
            f"evaluation_mode must be pure_neural / pure_heuristic / hybrid, got {evaluation_mode!r}"
        ]}, 400)
    heuristic_weight = float(payload.get("heuristic_weight", 0.5))

    try:
        gs = build_state(payload)
    except (KeyError, ValueError) as e:
        return None, ({"ok": False, "errors": [str(e)]}, 200)
    try:
        valid_moves = gs.get_valid_moves()
    except Exception as e:  # noqa: BLE001
        return None, ({"ok": False, "errors": [f"invalid position: {e}"]}, 200)
    if not valid_moves:
        return None, ({"ok": False, "errors": ["no legal moves available from this position"]}, 200)

    return {
        "model_id": model_id,
        "num_sims": num_sims,
        "capped_from": capped_from,
        "top_k": top_k,
        "c_puct": c_puct,
        "fpu_reduction": fpu_reduction,
        "evaluation_mode": evaluation_mode,
        "heuristic_weight": heuristic_weight,
        "gs": gs,
        "valid_moves": valid_moves,
    }, None


def _evaluate_policy(ctx):
    """num_sims==0 raw-policy path. Returns ``(top_moves, value, top_move)``."""
    gs, valid_moves = ctx["gs"], ctx["valid_moves"]
    wrapper = get_wrapper(ctx["model_id"])
    move_probs_t, value_t = wrapper.predict_from_state(gs)
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
    top = _top_moves(probs, valid_moves, ctx["top_k"], include_visits=False)
    # Per-entry best_move_value (raw-policy mode also benefits — lets the user
    # see which low-ranked moves still have positive after-value).
    for entry in top:
        matched = _match_entry_to_move(entry, valid_moves)
        entry["best_move_value"] = (
            _best_move_value(wrapper, gs, matched) if matched is not None else None
        )
    top_move = max(valid_moves, key=lambda m: float(probs[_encoder.move_to_index(m)]))
    return top, value, top_move


def _finalize_eval_response(gs, valid_moves, *, mode, value, best_value, top, num_sims, capped_from):
    """Assemble the JSON body returned by both the sync route and async job."""
    return {
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
        # Effective sims actually run, plus the pre-cap request if the public
        # cap clamped it (null when uncapped / owner-bypassed). Lets the UI
        # tell the user "ran 3200 of 256000 requested" instead of silently
        # lying about the search depth.
        "num_sims": num_sims,
        "capped_from": capped_from,
    }


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():  # type: ignore[no-untyped-def]
    payload = request.get_json(force=True, silent=False) or {}
    ctx, err = _prepare_evaluation(payload, _owner_token_ok(payload))
    if err is not None:
        body, status = err
        return jsonify(body), status

    gs, valid_moves, num_sims = ctx["gs"], ctx["valid_moves"], ctx["num_sims"]
    if num_sims > 0:
        mcts = get_mcts(
            ctx["model_id"], num_sims,
            c_puct=ctx["c_puct"],
            fpu_reduction=ctx["fpu_reduction"],
            evaluation_mode=ctx["evaluation_mode"],
            heuristic_weight=ctx["heuristic_weight"],
        )
        # Hold _mcts_lock from reset_tree through PV extraction — the cached
        # MCTS instance + its _cached_root are shared across users, so a
        # concurrent search would race on the tree state.
        _mcts_lock.acquire()
        try:
            # Reset before each call — subtree reuse is enabled so we can read
            # the tree post-search, but each /api/evaluate is logically a fresh
            # position; we don't want last search's tree biasing this one.
            try:
                mcts.reset_tree()
            except AttributeError:
                mcts._cached_root = None  # fallback for older engine versions
            probs = mcts.search(gs, move_number=1)
            top, value, _ = _mcts_result_from_probs(
                mcts, probs, gs, valid_moves, ctx["model_id"], ctx["top_k"], num_sims,
            )
        finally:
            _mcts_lock.release()
        mode = "mcts"
    else:
        # Raw-policy inference also touches the non-thread-safe TensorPool, so
        # serialize it on _mcts_lock too (an async job's background search may
        # be doing inference concurrently).
        with _mcts_lock:
            top, value, _ = _evaluate_policy(ctx)
        mode = "policy"

    best_value = top[0].get("best_move_value") if top else None
    return jsonify(_finalize_eval_response(
        gs, valid_moves, mode=mode, value=value, best_value=best_value,
        top=top, num_sims=num_sims, capped_from=ctx["capped_from"],
    ))


# --- Async evaluation: kick a long search onto a background thread ----------

def _evict_stale_jobs() -> None:
    now = time.time()
    with _jobs_lock:
        stale = [
            jid for jid, j in _jobs.items()
            if j.get("finished_at") and now - j["finished_at"] > _JOB_TTL_SECONDS
        ]
        for jid in stale:
            _jobs.pop(jid, None)


def _run_eval_job(job_id: str, ctx: Dict[str, Any]) -> None:
    """Worker body: run the (possibly very long) search on its own private
    MCTS instance, reporting progress, then stash the finished response."""
    def _progress(done, total):
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job["progress"] = {"done": int(done), "total": int(total)}

    started = time.time()
    try:
        gs, valid_moves, num_sims = ctx["gs"], ctx["valid_moves"], ctx["num_sims"]
        # Serialize the whole compute on _mcts_lock. The NetworkWrapper's
        # TensorPool is NOT thread-safe, so this background search's inference
        # must never overlap a request-thread's inference. We use a private
        # MCTS instance (own _cached_root) but still take the global lock so
        # only one search/inference runs server-wide at a time. Progress is
        # reported via _progress (guarded by the separate _jobs_lock), so the
        # poller keeps updating even while this lock is held.
        with _mcts_lock:
            if num_sims > 0:
                mcts = _construct_mcts(
                    ctx["model_id"], num_sims,
                    c_puct=ctx["c_puct"],
                    fpu_reduction=ctx["fpu_reduction"],
                    evaluation_mode=ctx["evaluation_mode"],
                    heuristic_weight=ctx["heuristic_weight"],
                )
                probs = mcts.search(gs, move_number=1, progress_callback=_progress)
                top, value, _ = _mcts_result_from_probs(
                    mcts, probs, gs, valid_moves, ctx["model_id"], ctx["top_k"], num_sims,
                )
                mode = "mcts"
            else:
                top, value, _ = _evaluate_policy(ctx)
                mode = "policy"
        best_value = top[0].get("best_move_value") if top else None
        result = _finalize_eval_response(
            gs, valid_moves, mode=mode, value=value, best_value=best_value,
            top=top, num_sims=num_sims, capped_from=ctx["capped_from"],
        )
        elapsed = time.time() - started
        rate = (num_sims / elapsed) if elapsed > 0 else 0.0
        log.info(
            "async eval job %s done: %d sims in %.1fs (%.0f sims/s)",
            job_id, num_sims, elapsed, rate,
        )
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job.update(status="done", result=result, finished_at=time.time())
                job["progress"] = {"done": num_sims, "total": num_sims}
    except Exception as e:  # noqa: BLE001
        log.exception("async eval job %s failed after %.1fs", job_id, time.time() - started)
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job.update(status="error", error=f"{type(e).__name__}: {e}", finished_at=time.time())


@app.route("/api/evaluate_async", methods=["POST"])
def api_evaluate_async():  # type: ignore[no-untyped-def]
    """Start a search on a background thread; return a job_id immediately.

    For searches too large to finish inside the proxy's response window
    (anything above ~10k sims over the Cloudflare tunnel). Poll
    /api/evaluate_result/<job_id> for progress and the eventual result.
    """
    payload = request.get_json(force=True, silent=False) or {}
    ctx, err = _prepare_evaluation(payload, _owner_token_ok(payload))
    if err is not None:
        body, status = err
        return jsonify(body), status

    _evict_stale_jobs()
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "running",
            "progress": {"done": 0, "total": ctx["num_sims"]},
            "result": None,
            "error": None,
            "created_at": time.time(),
            "finished_at": None,
        }
    threading.Thread(target=_run_eval_job, args=(job_id, ctx), daemon=True).start()
    log.info("async eval job %s started (sims=%d)", job_id, ctx["num_sims"])
    return jsonify({
        "ok": True,
        "job_id": job_id,
        "num_sims": ctx["num_sims"],
        "capped_from": ctx["capped_from"],
    })


@app.route("/api/evaluate_result/<job_id>")
def api_evaluate_result(job_id):  # type: ignore[no-untyped-def]
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return jsonify({"ok": False, "errors": ["unknown or expired job_id"]}), 404
        status = job["status"]
        progress = dict(job.get("progress") or {})
        error = job.get("error")
        result = job.get("result")
    if status == "running":
        return jsonify({"ok": True, "status": "running", "progress": progress})
    if status == "error":
        return jsonify({"ok": True, "status": "error", "errors": [error or "search failed"]})
    return jsonify({"ok": True, "status": "done", "progress": progress, "result": result})


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

    result = {
        "ok": True,
        "new_position": new_payload,
        "applied_move": _serialize_move(move),
        "legal_moves": _legal_moves_payload(new_valid),
        "game_over": game_over,
        "winner": winner,
    }
    # Best-effort: append to the per-day game log. Failures don't bubble up.
    _log_move_event(payload, move_spec, result)
    return jsonify(result)


def _client_ip() -> str:
    """Pick the most-trustworthy client IP available.

    Cloudflare adds ``CF-Connecting-IP`` for tunneled traffic; behind
    cloudflared the remote_addr is always 127.0.0.1, so we'd be
    rate-limiting "the tunnel" instead of "the user" without this.
    Falls back to ``X-Forwarded-For`` (first hop) and finally remote_addr.
    """
    cf = request.headers.get("CF-Connecting-IP")
    if cf:
        return cf.strip()
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _build_bga_scraper():
    """Construct a BGAScraper with cookies loaded. Returns None on auth
    failure. Pulled out so tests can monkey-patch the constructor without
    reaching into ``yinsh_ml.data.scrapers``.
    """
    # Lazy import — the scraper pulls ``certifi`` and a urllib stack we don't
    # want to load until somebody actually requests an import. Keeps server
    # startup fast and keeps the rest of the API independent of the scraper.
    from yinsh_ml.data.scrapers.bga import BGAScraper  # noqa: WPS433

    scraper = BGAScraper()
    cookies_path = BGA_COOKIES_PATH
    if not Path(cookies_path).exists():
        log.warning("BGA cookies file not found at %s", cookies_path)
        return None
    if not scraper.load_cookies(cookies_path):
        log.warning("BGA cookies at %s failed auth probe", cookies_path)
        return None
    return scraper


@app.route("/api/import_bga", methods=["POST"])
def api_import_bga():  # type: ignore[no-untyped-def]
    """Import a BGA YINSH replay for Review-mode step-through.

    Caches each successful import on disk so re-visits don't re-spend BGA's
    200/day per-account cap; rate-limits novel imports per remote IP so a
    single visitor can't burn the whole day's allowance.
    """
    payload = request.get_json(force=True, silent=True) or {}
    raw_input = payload.get("url_or_table_id")
    if raw_input is None:
        return jsonify({"ok": False, "errors": ["url_or_table_id is required"]}), 400

    try:
        table_id = parse_url_or_id(raw_input)
    except BGAImportError as e:
        return jsonify({"ok": False, "errors": [e.user_message]}), e.status

    # Cache hit short-circuits everything: no BGA fetch, no rate-limit charge.
    cached = cache_load(BGA_CACHE_DIR, table_id)
    if cached is not None:
        log.info("BGA cache hit for table %s", table_id)
        # ``cached: True`` after the spread so prior-iteration values can't
        # mask the cache-hit flag (the saved payload doesn't carry one).
        return jsonify({"ok": True, **cached, "table_id": table_id, "cached": True})

    # Novel import — check rate limit BEFORE doing any work. Cookies / scrape
    # / replay failures shouldn't burn the user's allowance, so the actual
    # record_rate_limit call happens after a successful import below.
    # Use the shared `_client_ip()` so we honor `CF-Connecting-IP` first
    # (deployed behind cloudflared, `remote_addr` is always 127.0.0.1).
    client_ip = _client_ip()
    if not check_rate_limit(client_ip):
        return jsonify({
            "ok": False,
            "errors": [
                f"Rate limit hit ({client_ip}). Max 5 novel BGA imports/hour "
                "per IP — cached games don't count."
            ],
        }), 429

    scraper = _build_bga_scraper()
    if scraper is None:
        return jsonify({
            "ok": False,
            "errors": [
                "BGA cookies missing or expired — see analysis_board/multiplayer/"
                "deploy/README.md for the Review-mode cookies setup."
            ],
        }), 200

    try:
        # BGACapHit is the only daily-cap signal that's worth distinguishing
        # from "this particular game failed." Catch it via the type name so
        # we don't have to thread the symbol through bga_import.py.
        parsed = scraper.scrape_game(table_id)
    except Exception as e:  # noqa: BLE001
        if e.__class__.__name__ == "BGACapHit":
            return jsonify({
                "ok": False,
                "errors": [
                    "BGA daily replay cap hit. Try again tomorrow, or import "
                    "a different game that's already cached."
                ],
            }), 200
        log.exception("BGA scrape raised for table %s", table_id)
        return jsonify({
            "ok": False,
            "errors": [f"BGA fetch failed: {e}"],
        }), 200

    if parsed is None:
        return jsonify({
            "ok": False,
            "errors": [
                f"BGA returned no replay for table {table_id} — table may not "
                "exist, may be private, or may be a different game."
            ],
        }), 200

    try:
        playback = replay_to_steps(parsed, _serialize_state, _serialize_move)
    except BGAImportError as e:
        return jsonify({"ok": False, "errors": [e.user_message]}), e.status

    # Save the bare playback to disk; ``cached`` is response-only metadata so
    # the on-disk value can't override the cache-hit flag on later reads.
    cache_save(BGA_CACHE_DIR, table_id, playback)
    record_rate_limit(client_ip)
    log.info("BGA import succeeded for table %s (%d steps)", table_id, len(playback["steps"]))
    return jsonify({"ok": True, "table_id": table_id, "cached": False, **playback})


@app.route("/api/import_screenshot", methods=["POST"])
def api_import_screenshot():  # type: ignore[no-untyped-def]
    """Parse a YINSH board image into a position payload via Claude vision.

    Request body::

        {"image_base64": "<base64>", "mime_type": "image/png"}

    Response (success)::

        {
          "ok": True,
          "position": {
            "pieces": [{"pos": "E5", "piece": "WHITE_RING"}, ...],
            "phase": "MAIN_GAME",
            "side_to_move": "WHITE",
            "scores": {"WHITE": 0, "BLACK": 0},
          },
          "confidence": "high" | "medium" | "low",
          "notes": "free-text caveats",
        }

    Failure modes — all body-shaped with an ``errors`` array (the
    frontend has a single parse path for these); HTTP status reflects
    severity (400 / 429 / 502 / 503).
    """
    payload = request.get_json(force=True, silent=False) or {}

    # Validate image inputs BEFORE checking rate limit — caller gets a
    # crisp 400 on malformed input instead of burning quota on a parse
    # we'd reject anyway.
    try:
        image_b64, mime_type = decode_and_validate_image(
            payload.get("image_base64"),
            payload.get("mime_type"),
        )
    except ScreenshotImportError as e:
        return jsonify({"ok": False, "errors": [e.user_message]}), e.status

    client_ip = _client_ip()
    # Aliased to keep a separate per-IP bucket from the BGA limiter
    # (different quotas, different services).
    if not screenshot_check_rate_limit(client_ip):
        return jsonify({
            "ok": False,
            "errors": [
                f"Rate limit hit ({client_ip}). Max 10 imports/hour per IP "
                "to keep public-deployment Claude API costs predictable. "
                "Try again in an hour."
            ],
        }), 429

    try:
        raw_parsed = call_claude_vision(image_b64, mime_type)
        usage = raw_parsed.pop("_usage", None)
        validated = validate_claude_response(raw_parsed)
    except ScreenshotImportError as e:
        # Failures don't count against the rate limit. Tells the user
        # to retry without sweating the quota.
        return jsonify({"ok": False, "errors": [e.user_message]}), e.status
    except Exception as e:  # noqa: BLE001
        log.exception("screenshot import unexpected failure")
        return jsonify({
            "ok": False,
            "errors": [f"Screenshot import failed: {e}"],
        }), 500

    # Success — burn one rate-limit slot.
    screenshot_record_rate_limit(client_ip)

    if usage is not None:
        log.info(
            "screenshot import: ip=%s confidence=%s pieces=%d "
            "tokens(in=%d/out=%d cache_create=%d cache_read=%d)",
            client_ip,
            validated["confidence"],
            len(validated["position"]["pieces"]),
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
            usage.get("cache_creation_input_tokens", 0),
            usage.get("cache_read_input_tokens", 0),
        )
    else:
        log.info(
            "screenshot import: ip=%s confidence=%s pieces=%d",
            client_ip,
            validated["confidence"],
            len(validated["position"]["pieces"]),
        )

    return jsonify({"ok": True, **validated})


def main() -> None:
    global _models  # noqa: PLW0603
    _models = discover_models()
    host = os.environ.get("YNS_HOST", "127.0.0.1")
    port = int(os.environ.get("YNS_PORT", "5173"))
    log.info("starting analysis board on http://%s:%d", host, port)
    app.run(host=host, port=port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
