"""Screenshot → position import via Claude vision.

The Flask layer (`/api/import_screenshot`) hands us a base64-encoded image;
we ask Claude Sonnet 4.6 to parse it into the same JSON shape
`currentPositionPayload()` produces on the frontend, validate the response,
and hand the parsed position back. The frontend then populates the Setup
composer so the user can review/correct before clicking Analyze.

Design notes:

* The Anthropic SDK is imported lazily so missing-dependency / missing-key
  setups don't blow up at module load — the endpoint reports 503 with a
  clear message instead.
* The system prompt is long and static; we mark it for prompt caching so
  repeat calls hit the ~0.1x read price after the first. Verify in usage:
  `cache_read_input_tokens` > 0 after the second call.
* Per-IP rate limiting (10/hour) is the cost guardrail. Same two-phase
  pattern as `bga_import.check_rate_limit` / `record_rate_limit` so a
  failed parse doesn't burn the user's quota.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

# Hard cap on raw image bytes the user can POST. 5MB easily covers a 4K
# screenshot at JPEG quality 90; well under Anthropic's per-image limit.
MAX_IMAGE_BYTES = 5 * 1024 * 1024

# Per-IP throttle. Each parse costs ~$0.01-0.03; the public URL has no auth,
# so an unthrottled attacker could rack up real money. Same window as
# bga_import for consistency.
RATE_LIMIT_MAX = 10
RATE_LIMIT_WINDOW_SECONDS = 3600

# Vision-eligible MIME types per Anthropic's docs. SVG/PDF are not images;
# rejected upstream so we don't surface a confusing API-side error.
ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}

# Model choice: Sonnet 4.6 — the best speed-quality balance for vision OCR.
# Opus 4.7 would be ~5x more expensive per call with no expected quality
# bump on this task (board layouts are visually simple, the labels are the
# hard part and Sonnet handles them fine in testing).
DEFAULT_MODEL = os.environ.get("YNS_SCREENSHOT_IMPORT_MODEL", "claude-sonnet-4-6")

# Valid game phases the model can return. ROW_COMPLETION and RING_REMOVAL
# are real engine phases but very hard to detect from a static image; we
# accept them on input but the frontend will rarely see them.
VALID_PHASES = {"RING_PLACEMENT", "MAIN_GAME", "ROW_COMPLETION", "RING_REMOVAL"}
VALID_SIDES = {"WHITE", "BLACK"}
VALID_PIECES = {"WHITE_RING", "BLACK_RING", "WHITE_MARKER", "BLACK_MARKER"}
VALID_CONFIDENCES = {"high", "medium", "low"}


class ScreenshotImportError(Exception):
    """Raised by the import pipeline with a user-facing message.

    Mirrors `BGAImportError` semantics — 200 by default so the frontend
    has a single body-shaped error path. 503 specifically for
    server-misconfiguration (no API key) so monitoring can distinguish
    "bad request" from "broken deploy."
    """

    def __init__(self, message: str, status: int = 200):
        super().__init__(message)
        self.user_message = message
        self.status = status


# ---------------------------------------------------------------------------
# Per-IP rate limiting
# ---------------------------------------------------------------------------

_rate_lock = threading.Lock()
_rate_tracker: Dict[str, List[float]] = {}


def check_rate_limit(ip: str, *, now: Optional[float] = None) -> bool:
    """True if ``ip`` is under the limit. Does NOT record — call
    ``record_rate_limit`` after a successful parse."""
    if now is None:
        now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    with _rate_lock:
        recent = [t for t in _rate_tracker.get(ip, []) if t >= cutoff]
        _rate_tracker[ip] = recent
        return len(recent) < RATE_LIMIT_MAX


def record_rate_limit(ip: str, *, now: Optional[float] = None) -> None:
    if now is None:
        now = time.time()
    with _rate_lock:
        recent = [t for t in _rate_tracker.get(ip, []) if t >= now - RATE_LIMIT_WINDOW_SECONDS]
        recent.append(now)
        _rate_tracker[ip] = recent


def _rate_reset_for_tests() -> None:
    with _rate_lock:
        _rate_tracker.clear()


# ---------------------------------------------------------------------------
# Image decode + validation
# ---------------------------------------------------------------------------

def decode_and_validate_image(image_b64: Any, mime_type: Any) -> Tuple[str, str]:
    """Validate and normalize the image inputs.

    Returns ``(image_b64_stripped, mime_type_lower)`` — both safe to pass
    straight to the Anthropic SDK. Raises ``ScreenshotImportError`` (400)
    on malformed input.
    """
    if not isinstance(image_b64, str) or not image_b64:
        raise ScreenshotImportError("image_base64 is required", status=400)
    if not isinstance(mime_type, str) or not mime_type:
        raise ScreenshotImportError("mime_type is required", status=400)

    # Accept data-URL-style prefixes (`data:image/png;base64,...`) by stripping
    # everything before the comma. Browsers' FileReader emits this format by
    # default; saves the frontend a `split(",")` step.
    if image_b64.startswith("data:"):
        comma_idx = image_b64.find(",")
        if comma_idx == -1:
            raise ScreenshotImportError("invalid data URL", status=400)
        image_b64 = image_b64[comma_idx + 1 :]

    mime_type_lower = mime_type.lower().strip()
    if mime_type_lower not in ALLOWED_MIME_TYPES:
        raise ScreenshotImportError(
            f"unsupported mime_type {mime_type!r}; "
            f"must be one of {sorted(ALLOWED_MIME_TYPES)}",
            status=400,
        )

    try:
        raw = base64.b64decode(image_b64, validate=True)
    except (binascii.Error, ValueError):
        raise ScreenshotImportError("image_base64 is not valid base64", status=400)

    if len(raw) == 0:
        raise ScreenshotImportError("image_base64 decoded to zero bytes", status=400)
    if len(raw) > MAX_IMAGE_BYTES:
        raise ScreenshotImportError(
            f"image too large: {len(raw)} bytes > {MAX_IMAGE_BYTES} bytes",
            status=400,
        )

    return image_b64, mime_type_lower


# ---------------------------------------------------------------------------
# Anthropic call
# ---------------------------------------------------------------------------

# Kept as a module constant so prompt-cache hashing is stable across calls.
# Any byte change here invalidates the cache; see `shared/prompt-caching.md`.
SYSTEM_PROMPT = """You are parsing a YINSH game position from a board image. \
YINSH is played on a hexagonal grid of 85 valid positions, labeled by \
column (A-K) and row (1-11). The valid rows per column are:

A: 2,3,4,5
B: 1,2,3,4,5,6,7
C: 1,2,3,4,5,6,7,8
D: 1,2,3,4,5,6,7,8,9
E: 1,2,3,4,5,6,7,8,9,10
F: 2,3,4,5,6,7,8,9,10
G: 2,3,4,5,6,7,8,9,10,11
H: 3,4,5,6,7,8,9,10,11
I: 4,5,6,7,8,9,10,11
J: 5,6,7,8,9,10,11
K: 7,8,9,10

Any other column/row combination is off the board.

Pieces:
- WHITE_RING: an open / hollow ring shape, light or white colored
- BLACK_RING: an open / hollow ring shape, dark or black colored
- WHITE_MARKER: a solid disc, light or white colored
- BLACK_MARKER: a solid disc, dark or black colored

Each player starts with 5 rings. The game has three phases:
- RING_PLACEMENT: rings being placed, no markers on board yet, each side \
has fewer than 5 rings on the board
- MAIN_GAME: both sides have 5 rings (or fewer if captures have happened), \
markers may be present
- RING_REMOVAL / ROW_COMPLETION: mid-capture sequences; hard to detect from \
a static image. Default to MAIN_GAME unless you have strong evidence \
otherwise.

Side to move is usually NOT visible in a static board photo. Return \
"unknown" for side_to_move unless the source clearly indicates it (e.g. an \
on-screen turn indicator in a digital screenshot).

Score (captured rings) may be visible as side-of-board scoreboards or as \
removed-ring indicators. If not visible, default to 0/0.

OUTPUT FORMAT: Respond with a JSON object only — no prose, no markdown \
code fences, no commentary. The exact schema:

{
  "pieces": [
    {"pos": "E5", "piece": "WHITE_RING"},
    {"pos": "F6", "piece": "BLACK_MARKER"},
    ...
  ],
  "phase": "MAIN_GAME",
  "side_to_move": "WHITE" | "BLACK" | "unknown",
  "scores": {"WHITE": 0, "BLACK": 0},
  "confidence": "high" | "medium" | "low",
  "notes": "any caveats about the parse, e.g. occluded areas, ambiguous \
pieces, glare"
}

Rules:
- "pos" must be a valid board position from the table above (e.g. "E5", \
"K7", "B1"). Lowercase letters are not valid; emit uppercase.
- Each (pos, piece) pair is one piece. Do not emit multiple pieces at \
the same position.
- "phase" must be exactly one of: RING_PLACEMENT, MAIN_GAME, ROW_COMPLETION, \
RING_REMOVAL.
- "side_to_move" must be exactly one of: WHITE, BLACK, unknown.
- "confidence" reflects YOUR overall confidence in the parse. Use "low" \
if many pieces are occluded, the image is heavily distorted, or you are \
guessing on phase.
- "notes" is a short free-text string; empty string "" if nothing to add.
- Do not invent pieces in heavily occluded areas — list those concerns in \
"notes" instead of guessing.

If the image is clearly not a YINSH board, return an empty pieces array \
with confidence "low" and notes explaining what you see instead.
"""

# Frontend hint embedded in the user turn — short, deterministic. The
# system prompt is the cached portion; this stays uncached but is tiny.
USER_DIRECTIVE = (
    "Parse the YINSH board position shown in this image. "
    "Respond with the JSON object specified in the system prompt and "
    "nothing else."
)


def _get_anthropic_client():
    """Construct an Anthropic client or raise ``ScreenshotImportError``
    with a 503 if the SDK / API key isn't available.

    Lazy-imported so the server starts cleanly when the SDK isn't
    installed (e.g. local dev without the feature) — only callers of
    this endpoint pay the cost of the missing dependency.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ScreenshotImportError(
            "Screenshot import is not configured (ANTHROPIC_API_KEY env var "
            "missing). See analysis_board/multiplayer/deploy/README.md.",
            status=503,
        )
    try:
        import anthropic  # type: ignore
    except ImportError:
        raise ScreenshotImportError(
            "Screenshot import is not available (anthropic SDK not "
            "installed). Run `pip install anthropic` to enable.",
            status=503,
        )
    return anthropic.Anthropic()


def call_claude_vision(
    image_b64: str,
    mime_type: str,
    *,
    client: Any = None,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Send the image to Claude and parse the JSON response.

    ``client`` is injected so tests can stub the Anthropic SDK without
    touching the network. In production it's constructed inside.

    Raises ``ScreenshotImportError`` (502) on API failures and (200) on
    malformed responses — distinguishing "Anthropic is down" from
    "Claude returned non-JSON" in the logs.
    """
    if client is None:
        client = _get_anthropic_client()

    try:
        message = client.messages.create(
            model=model,
            max_tokens=2048,
            # Top-level auto-caching: caches the largest cacheable prefix
            # (system prompt + tool definitions if any). Since we have no
            # tools, this caches just the system prompt. The actual image
            # bytes vary per request and live after the breakpoint, so
            # they don't invalidate the cache.
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": USER_DIRECTIVE},
                    ],
                }
            ],
        )
    except Exception as e:  # noqa: BLE001
        # Includes anthropic.APIStatusError, RateLimitError, network
        # errors, etc. Bubble up as 502 — the caller can retry.
        raise ScreenshotImportError(
            f"Claude vision call failed: {e}",
            status=502,
        )

    # Stitch all text blocks together. Sonnet should return one text
    # block but the API contract allows multiple.
    text_parts: List[str] = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            text_parts.append(getattr(block, "text", ""))
    raw_text = "".join(text_parts).strip()
    if not raw_text:
        raise ScreenshotImportError(
            "Claude returned an empty response",
            status=200,
        )

    # Tolerate accidental code-fence wrapping even though the prompt
    # asks for raw JSON — cheap defense against a regression.
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        # `json\n{...}\n` after stripping ticks
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ScreenshotImportError(
            f"Claude returned non-JSON response: {e}. Got: {raw_text[:200]!r}",
            status=200,
        )

    if not isinstance(parsed, dict):
        raise ScreenshotImportError(
            f"Claude returned a non-object JSON value: {type(parsed).__name__}",
            status=200,
        )

    # Surface usage in logs so we can verify cache hit rates after the
    # second call. Not returned to the client.
    usage = getattr(message, "usage", None)
    if usage is not None:
        parsed["_usage"] = {
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
        }
    return parsed


# ---------------------------------------------------------------------------
# Response validation
# ---------------------------------------------------------------------------

def _is_valid_position_string(pos: str) -> bool:
    """Cheap validator matching VALID_POSITIONS in app.js / yinsh_ml.

    Kept in sync with yinsh_ml/game/constants.py — duplicated here so we
    can validate without dragging in the engine's import chain.
    """
    if not isinstance(pos, str) or len(pos) < 2 or len(pos) > 3:
        return False
    col = pos[0]
    rest = pos[1:]
    if not rest.isdigit():
        return False
    row = int(rest)
    cols = {
        "A": (2, 5), "B": (1, 7), "C": (1, 8), "D": (1, 9), "E": (1, 10),
        "F": (2, 10), "G": (2, 11), "H": (3, 11), "I": (4, 11), "J": (5, 11),
        "K": (7, 10),
    }
    if col not in cols:
        return False
    lo, hi = cols[col]
    return lo <= row <= hi


def validate_claude_response(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce the Claude response into the frontend's expected shape.

    Drops anything that doesn't match the schema rather than throwing —
    Claude is usually close but occasionally adds garbage. The user
    reviews the result via the existing UI before clicking Analyze, so
    a partial parse with notes is more useful than a total failure.

    Returns the cleaned response dict (without the internal `_usage`
    key). Raises ``ScreenshotImportError`` only on structurally
    unfixable shapes.
    """
    pieces_in = parsed.get("pieces")
    if not isinstance(pieces_in, list):
        raise ScreenshotImportError(
            "Claude response missing 'pieces' array",
            status=200,
        )

    dropped: List[str] = []
    pieces_out: List[Dict[str, str]] = []
    seen_positions: set = set()
    for entry in pieces_in:
        if not isinstance(entry, dict):
            dropped.append(f"non-object piece entry: {entry!r}")
            continue
        pos = entry.get("pos")
        piece = entry.get("piece")
        if not isinstance(pos, str) or not isinstance(piece, str):
            dropped.append(f"missing pos/piece on entry {entry!r}")
            continue
        pos = pos.upper().strip()
        piece = piece.upper().strip()
        if piece not in VALID_PIECES:
            dropped.append(f"unknown piece {piece!r} at {pos}")
            continue
        if not _is_valid_position_string(pos):
            dropped.append(f"off-board position {pos!r}")
            continue
        if pos in seen_positions:
            dropped.append(f"duplicate position {pos!r}; kept first")
            continue
        seen_positions.add(pos)
        pieces_out.append({"pos": pos, "piece": piece})

    phase = str(parsed.get("phase", "MAIN_GAME")).upper().strip()
    if phase not in VALID_PHASES:
        dropped.append(f"unknown phase {phase!r}; defaulting to MAIN_GAME")
        phase = "MAIN_GAME"

    side = str(parsed.get("side_to_move", "unknown")).upper().strip()
    if side not in VALID_SIDES and side != "UNKNOWN":
        dropped.append(f"unknown side {side!r}; defaulting to WHITE")
        side = "WHITE"
    elif side == "UNKNOWN":
        # The frontend doesn't have an "unknown" state — pick WHITE and
        # surface in notes. WHITE moves first so it's the safer default
        # in RING_PLACEMENT.
        side = "WHITE"

    scores_in = parsed.get("scores") or {}
    try:
        white_score = int(scores_in.get("WHITE", 0))
        black_score = int(scores_in.get("BLACK", 0))
    except (TypeError, ValueError):
        white_score, black_score = 0, 0
        dropped.append(f"non-integer scores: {scores_in!r}; defaulting to 0/0")
    # Clamp to legal range; YINSH ends at 3 rings captured.
    white_score = max(0, min(5, white_score))
    black_score = max(0, min(5, black_score))

    confidence = str(parsed.get("confidence", "medium")).lower().strip()
    if confidence not in VALID_CONFIDENCES:
        confidence = "medium"

    notes = str(parsed.get("notes", "") or "")
    if dropped:
        validator_note = "Validator dropped: " + "; ".join(dropped)
        notes = (notes + " · " + validator_note).strip(" ·") if notes else validator_note

    return {
        "position": {
            "pieces": pieces_out,
            "phase": phase,
            "side_to_move": side,
            "scores": {"WHITE": white_score, "BLACK": black_score},
        },
        "confidence": confidence,
        "notes": notes,
    }
