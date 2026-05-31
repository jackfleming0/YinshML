"""Unit tests for the screenshot-import endpoint.

The Anthropic SDK is mocked end-to-end so these run without an API key,
without network access, and without the SDK installed (production CI may
not include `anthropic`). Test seams:

* ``decode_and_validate_image`` is pure — tested directly.
* ``validate_claude_response`` is pure — tested directly.
* ``call_claude_vision`` accepts an injected ``client`` so we hand it a
  stub that records arguments and returns canned responses.
* The Flask endpoint is exercised via the test client with a
  monkeypatched ``call_claude_vision`` so we cover the route's
  validation / rate-limiting / error-handling without touching the
  vision call internals.
"""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Make a 1x1 PNG once — base64 of `\x89PNG\r\n\x1a\n` IHDR + tiny chunks.
# Smallest valid PNG file. Reused everywhere we need a "real" image.
ONE_PX_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89"
    b"\x00\x00\x00\rIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01"
    b"\x18\xdd\x8db"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)
ONE_PX_PNG_B64 = base64.b64encode(ONE_PX_PNG).decode("ascii")


@pytest.fixture(autouse=True)
def _reset_rate_limit():
    """Each test starts with a clean rate-limit tracker."""
    from analysis_board.screenshot_import import _rate_reset_for_tests
    _rate_reset_for_tests()
    yield
    _rate_reset_for_tests()


# ---------------------------------------------------------------------------
# decode_and_validate_image
# ---------------------------------------------------------------------------

class TestDecodeAndValidateImage:
    def test_accepts_valid_png(self):
        from analysis_board.screenshot_import import decode_and_validate_image
        b64, mime = decode_and_validate_image(ONE_PX_PNG_B64, "image/png")
        assert mime == "image/png"
        # Stripped to bare base64 (no data: prefix).
        assert not b64.startswith("data:")
        assert base64.b64decode(b64) == ONE_PX_PNG

    def test_accepts_data_url_prefix(self):
        from analysis_board.screenshot_import import decode_and_validate_image
        data_url = f"data:image/png;base64,{ONE_PX_PNG_B64}"
        b64, mime = decode_and_validate_image(data_url, "image/png")
        assert base64.b64decode(b64) == ONE_PX_PNG

    def test_rejects_missing_base64(self):
        from analysis_board.screenshot_import import (
            ScreenshotImportError, decode_and_validate_image,
        )
        with pytest.raises(ScreenshotImportError) as exc:
            decode_and_validate_image("", "image/png")
        assert exc.value.status == 400

    def test_rejects_unknown_mime(self):
        from analysis_board.screenshot_import import (
            ScreenshotImportError, decode_and_validate_image,
        )
        with pytest.raises(ScreenshotImportError) as exc:
            decode_and_validate_image(ONE_PX_PNG_B64, "application/pdf")
        assert exc.value.status == 400
        assert "unsupported mime_type" in exc.value.user_message

    def test_rejects_invalid_base64(self):
        from analysis_board.screenshot_import import (
            ScreenshotImportError, decode_and_validate_image,
        )
        with pytest.raises(ScreenshotImportError) as exc:
            decode_and_validate_image("not!base64!@@@", "image/png")
        assert exc.value.status == 400
        assert "base64" in exc.value.user_message.lower()

    def test_rejects_oversize_image(self):
        from analysis_board.screenshot_import import (
            MAX_IMAGE_BYTES, ScreenshotImportError, decode_and_validate_image,
        )
        # 6MB of payload → 8MB-ish base64. Cap is 5MB raw.
        oversize_raw = b"\x00" * (MAX_IMAGE_BYTES + 1)
        big = base64.b64encode(oversize_raw).decode("ascii")
        with pytest.raises(ScreenshotImportError) as exc:
            decode_and_validate_image(big, "image/png")
        assert exc.value.status == 400
        assert "too large" in exc.value.user_message

    def test_mime_type_case_insensitive(self):
        from analysis_board.screenshot_import import decode_and_validate_image
        _, mime = decode_and_validate_image(ONE_PX_PNG_B64, "Image/PNG")
        assert mime == "image/png"


# ---------------------------------------------------------------------------
# validate_claude_response
# ---------------------------------------------------------------------------

class TestValidateClaudeResponse:
    def test_passes_clean_response(self):
        from analysis_board.screenshot_import import validate_claude_response
        result = validate_claude_response({
            "pieces": [
                {"pos": "E5", "piece": "WHITE_RING"},
                {"pos": "F6", "piece": "BLACK_MARKER"},
            ],
            "phase": "MAIN_GAME",
            "side_to_move": "WHITE",
            "scores": {"WHITE": 1, "BLACK": 0},
            "confidence": "high",
            "notes": "",
        })
        assert result["confidence"] == "high"
        assert result["position"]["phase"] == "MAIN_GAME"
        assert result["position"]["side_to_move"] == "WHITE"
        assert result["position"]["scores"] == {"WHITE": 1, "BLACK": 0}
        assert len(result["position"]["pieces"]) == 2

    def test_drops_off_board_positions(self):
        from analysis_board.screenshot_import import validate_claude_response
        result = validate_claude_response({
            "pieces": [
                {"pos": "E5", "piece": "WHITE_RING"},
                {"pos": "A1", "piece": "BLACK_RING"},  # off-board: A column starts row 2
                {"pos": "L1", "piece": "BLACK_RING"},  # off-board: no L column
            ],
            "phase": "MAIN_GAME",
            "side_to_move": "WHITE",
            "scores": {"WHITE": 0, "BLACK": 0},
            "confidence": "medium",
            "notes": "",
        })
        positions = {p["pos"] for p in result["position"]["pieces"]}
        assert positions == {"E5"}
        assert "off-board" in result["notes"]

    def test_drops_unknown_piece_types(self):
        from analysis_board.screenshot_import import validate_claude_response
        result = validate_claude_response({
            "pieces": [
                {"pos": "E5", "piece": "WHITE_RING"},
                {"pos": "F5", "piece": "PURPLE_RING"},  # bogus
            ],
            "phase": "MAIN_GAME",
            "side_to_move": "WHITE",
            "scores": {"WHITE": 0, "BLACK": 0},
            "confidence": "high",
            "notes": "",
        })
        assert len(result["position"]["pieces"]) == 1
        assert "unknown piece" in result["notes"]

    def test_dedups_duplicate_positions(self):
        from analysis_board.screenshot_import import validate_claude_response
        result = validate_claude_response({
            "pieces": [
                {"pos": "E5", "piece": "WHITE_RING"},
                {"pos": "E5", "piece": "BLACK_RING"},  # duplicate
            ],
            "phase": "MAIN_GAME",
            "side_to_move": "WHITE",
            "scores": {"WHITE": 0, "BLACK": 0},
            "confidence": "high",
            "notes": "",
        })
        # First entry wins.
        assert result["position"]["pieces"] == [{"pos": "E5", "piece": "WHITE_RING"}]
        assert "duplicate" in result["notes"]

    def test_uppercases_lowercase_input(self):
        from analysis_board.screenshot_import import validate_claude_response
        result = validate_claude_response({
            "pieces": [{"pos": "e5", "piece": "white_ring"}],
            "phase": "main_game",
            "side_to_move": "white",
            "scores": {"WHITE": 0, "BLACK": 0},
            "confidence": "high",
            "notes": "",
        })
        assert result["position"]["pieces"][0]["pos"] == "E5"
        assert result["position"]["pieces"][0]["piece"] == "WHITE_RING"
        assert result["position"]["phase"] == "MAIN_GAME"
        assert result["position"]["side_to_move"] == "WHITE"

    def test_unknown_side_defaults_to_white(self):
        from analysis_board.screenshot_import import validate_claude_response
        result = validate_claude_response({
            "pieces": [],
            "phase": "MAIN_GAME",
            "side_to_move": "unknown",
            "scores": {"WHITE": 0, "BLACK": 0},
            "confidence": "low",
            "notes": "blurry",
        })
        # Unknown isn't a real side; default to WHITE (moves first).
        assert result["position"]["side_to_move"] == "WHITE"

    def test_unknown_phase_defaults_to_main_game(self):
        from analysis_board.screenshot_import import validate_claude_response
        result = validate_claude_response({
            "pieces": [],
            "phase": "TIEBREAKER",  # not a real phase
            "side_to_move": "WHITE",
            "scores": {"WHITE": 0, "BLACK": 0},
            "confidence": "low",
            "notes": "",
        })
        assert result["position"]["phase"] == "MAIN_GAME"

    def test_clamps_scores(self):
        from analysis_board.screenshot_import import validate_claude_response
        result = validate_claude_response({
            "pieces": [],
            "phase": "MAIN_GAME",
            "side_to_move": "WHITE",
            "scores": {"WHITE": 99, "BLACK": -5},
            "confidence": "medium",
            "notes": "",
        })
        assert result["position"]["scores"] == {"WHITE": 5, "BLACK": 0}

    def test_missing_pieces_array_raises(self):
        from analysis_board.screenshot_import import (
            ScreenshotImportError, validate_claude_response,
        )
        with pytest.raises(ScreenshotImportError):
            validate_claude_response({
                "phase": "MAIN_GAME",
                "side_to_move": "WHITE",
            })


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimit:
    def test_check_and_record(self):
        from analysis_board.screenshot_import import (
            RATE_LIMIT_MAX, check_rate_limit, record_rate_limit,
        )
        ip = "10.0.0.1"
        for _ in range(RATE_LIMIT_MAX):
            assert check_rate_limit(ip)
            record_rate_limit(ip)
        # 11th check should fail.
        assert not check_rate_limit(ip)

    def test_per_ip_isolation(self):
        from analysis_board.screenshot_import import (
            RATE_LIMIT_MAX, check_rate_limit, record_rate_limit,
        )
        for _ in range(RATE_LIMIT_MAX):
            record_rate_limit("10.0.0.1")
        assert not check_rate_limit("10.0.0.1")
        # Different IP is untouched.
        assert check_rate_limit("10.0.0.2")

    def test_window_expiry(self):
        from analysis_board.screenshot_import import (
            RATE_LIMIT_MAX, RATE_LIMIT_WINDOW_SECONDS,
            check_rate_limit, record_rate_limit,
        )
        ip = "10.0.0.1"
        # Stuff in old entries past the window.
        for _ in range(RATE_LIMIT_MAX):
            record_rate_limit(ip, now=1.0)
        assert not check_rate_limit(ip, now=1.0)
        # Move clock forward past the window — old entries drop off.
        future = 1.0 + RATE_LIMIT_WINDOW_SECONDS + 1
        assert check_rate_limit(ip, now=future)


# ---------------------------------------------------------------------------
# call_claude_vision (injected client)
# ---------------------------------------------------------------------------

class StubClient:
    """Records calls and returns canned responses. Mimics
    ``anthropic.Anthropic`` enough for ``call_claude_vision``.

    With tool-use forcing, the real Anthropic SDK returns a tool_use
    block whose ``.input`` is the validated JSON. We mirror that shape.
    Pass ``tool_input=None`` to simulate a model that didn't call the
    tool (e.g. hit max_tokens mid-reasoning, or refused).
    """

    def __init__(
        self,
        *,
        tool_input: Any = "DEFAULT",
        raise_exception: Exception = None,
        stop_reason: str = "tool_use",
        extra_blocks: List[Any] = None,
    ):
        # Default to a minimal valid position so most tests don't have to
        # spell it out. Pass `tool_input=None` to omit the tool_use block
        # entirely.
        if tool_input == "DEFAULT":
            tool_input = {
                "pieces": [], "phase": "MAIN_GAME", "side_to_move": "WHITE",
                "scores": {"WHITE": 0, "BLACK": 0}, "confidence": "low",
                "notes": "",
            }
        self.tool_input = tool_input
        self.raise_exception = raise_exception
        self.stop_reason = stop_reason
        self.extra_blocks = extra_blocks or []
        self.calls: List[Dict[str, Any]] = []
        self.messages = self
        self.last_usage = SimpleNamespace(
            input_tokens=100,
            output_tokens=200,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.raise_exception is not None:
            raise self.raise_exception
        content = list(self.extra_blocks)
        if self.tool_input is not None:
            content.append(SimpleNamespace(
                type="tool_use",
                name="submit_yinsh_position",
                input=self.tool_input,
            ))
        return SimpleNamespace(
            content=content,
            usage=self.last_usage,
            stop_reason=self.stop_reason,
        )


class TestCallClaudeVision:
    def test_extracts_tool_use_input(self):
        from analysis_board.screenshot_import import call_claude_vision
        stub = StubClient(tool_input={
            "pieces": [{"pos": "E5", "piece": "WHITE_RING"}],
            "phase": "MAIN_GAME",
            "side_to_move": "WHITE",
            "scores": {"WHITE": 0, "BLACK": 0},
            "confidence": "high",
            "notes": "",
        })
        result = call_claude_vision(ONE_PX_PNG_B64, "image/png", client=stub)
        assert result["pieces"] == [{"pos": "E5", "piece": "WHITE_RING"}]
        # Usage is attached to the parsed dict for the endpoint to log.
        assert "_usage" in result

    def test_forces_position_tool(self):
        """Without tool_choice forcing, Sonnet preambles before the
        tool call and we hit the no-tool_use path. Pin this so a refactor
        that drops tool_choice gets caught."""
        from analysis_board.screenshot_import import call_claude_vision
        stub = StubClient()
        call_claude_vision(ONE_PX_PNG_B64, "image/png", client=stub)
        sent = stub.calls[0]
        assert sent["tool_choice"] == {
            "type": "tool", "name": "submit_yinsh_position",
        }
        # And the tool definition must actually be present in tools.
        tools = sent["tools"]
        assert any(t.get("name") == "submit_yinsh_position" for t in tools)

    def test_caches_system_prompt(self):
        """The system prompt must be sent with cache_control set, otherwise
        repeat calls don't get the ~10x input-cost discount."""
        from analysis_board.screenshot_import import call_claude_vision
        stub = StubClient()
        call_claude_vision(ONE_PX_PNG_B64, "image/png", client=stub)
        sent_system = stub.calls[0]["system"]
        assert isinstance(sent_system, list)
        assert sent_system[0]["cache_control"] == {"type": "ephemeral"}

    def test_raises_when_model_skips_the_tool(self):
        """If the model refuses or hits max_tokens before tool_use, no
        tool_use block is emitted. Surface a useful error to the caller
        rather than crashing on a missing .input."""
        from analysis_board.screenshot_import import (
            ScreenshotImportError, call_claude_vision,
        )
        stub = StubClient(tool_input=None, stop_reason="max_tokens")
        with pytest.raises(ScreenshotImportError) as exc:
            call_claude_vision(ONE_PX_PNG_B64, "image/png", client=stub)
        assert "tool" in exc.value.user_message.lower()
        # stop_reason should be surfaced in the error message for debug.
        assert "max_tokens" in exc.value.user_message

    def test_ignores_preamble_text_block(self):
        """If the model emits a text block alongside the tool_use (rare
        but legal), we still extract from the tool_use. The earlier
        text-mode implementation broke on this kind of mixed response."""
        from analysis_board.screenshot_import import call_claude_vision
        text_preamble = SimpleNamespace(
            type="text", text="Analyzing the board...",
        )
        stub = StubClient(extra_blocks=[text_preamble])
        result = call_claude_vision(ONE_PX_PNG_B64, "image/png", client=stub)
        assert "pieces" in result

    def test_wraps_sdk_exceptions(self):
        from analysis_board.screenshot_import import (
            ScreenshotImportError, call_claude_vision,
        )
        stub = StubClient(raise_exception=RuntimeError("Anthropic 500"))
        with pytest.raises(ScreenshotImportError) as exc:
            call_claude_vision(ONE_PX_PNG_B64, "image/png", client=stub)
        assert exc.value.status == 502
        assert "Anthropic 500" in exc.value.user_message


# ---------------------------------------------------------------------------
# Flask endpoint (HTTP-level)
# ---------------------------------------------------------------------------

@pytest.fixture()
def flask_client(monkeypatch):
    """Build a Flask test client without booting the model registry.

    `discover_models` walks `models/` and loads `.pt` files — way too
    expensive for unit tests, and ones that need to run in CI without
    Apple Silicon (MPS) on hand. Skip it; the import endpoint doesn't
    need models loaded.
    """
    from analysis_board import server
    monkeypatch.setattr(server, "discover_models", lambda: [])
    server._models = []
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        yield c


def _make_request_body(b64=ONE_PX_PNG_B64, mime="image/png"):
    return {"image_base64": b64, "mime_type": mime}


def _patch_vision_success(monkeypatch, response_dict=None):
    """Replace `server.call_claude_vision` with a stub that returns a
    canned successful parse. Returns the stub for call-count assertions."""
    if response_dict is None:
        response_dict = {
            "pieces": [{"pos": "E5", "piece": "WHITE_RING"}],
            "phase": "MAIN_GAME",
            "side_to_move": "WHITE",
            "scores": {"WHITE": 0, "BLACK": 0},
            "confidence": "high",
            "notes": "",
            "_usage": {
                "input_tokens": 50,
                "output_tokens": 60,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 1500,
            },
        }
    calls = []
    def stub(image_b64, mime_type, **kwargs):
        calls.append({"mime_type": mime_type, "b64_len": len(image_b64)})
        return dict(response_dict)
    from analysis_board import server
    monkeypatch.setattr(server, "call_claude_vision", stub)
    return calls


class TestEndpoint:
    def test_success_returns_validated_payload(self, flask_client, monkeypatch):
        calls = _patch_vision_success(monkeypatch)
        resp = flask_client.post(
            "/api/import_screenshot",
            json=_make_request_body(),
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["ok"] is True
        assert body["confidence"] == "high"
        assert body["position"]["pieces"] == [{"pos": "E5", "piece": "WHITE_RING"}]
        assert body["position"]["phase"] == "MAIN_GAME"
        assert body["position"]["side_to_move"] == "WHITE"
        assert body["position"]["scores"] == {"WHITE": 0, "BLACK": 0}
        # `_usage` is consumed by the server (for logs) but never exposed
        # to the frontend.
        assert "_usage" not in body
        assert len(calls) == 1
        assert calls[0]["mime_type"] == "image/png"

    def test_missing_image_returns_400(self, flask_client):
        resp = flask_client.post(
            "/api/import_screenshot",
            json={"mime_type": "image/png"},
        )
        assert resp.status_code == 400
        body = resp.get_json()
        assert body["ok"] is False
        assert "image_base64" in body["errors"][0]

    def test_unsupported_mime_returns_400(self, flask_client):
        resp = flask_client.post(
            "/api/import_screenshot",
            json={"image_base64": ONE_PX_PNG_B64, "mime_type": "application/pdf"},
        )
        assert resp.status_code == 400
        body = resp.get_json()
        assert "unsupported mime_type" in body["errors"][0]

    def test_oversize_returns_400(self, flask_client):
        from analysis_board.screenshot_import import MAX_IMAGE_BYTES
        big = base64.b64encode(b"\x00" * (MAX_IMAGE_BYTES + 1)).decode("ascii")
        resp = flask_client.post(
            "/api/import_screenshot",
            json={"image_base64": big, "mime_type": "image/png"},
        )
        assert resp.status_code == 400
        body = resp.get_json()
        assert "too large" in body["errors"][0]

    def test_missing_api_key_returns_503(self, flask_client, monkeypatch):
        """The real `call_claude_vision` raises a 503 when neither the
        env var nor an injected client is available. Don't mock; let it
        run its precondition check."""
        from analysis_board import server
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        # Force-reset to the real implementation in case a previous test
        # left a stub in place.
        from analysis_board.screenshot_import import call_claude_vision as real
        monkeypatch.setattr(server, "call_claude_vision", real)
        resp = flask_client.post(
            "/api/import_screenshot",
            json=_make_request_body(),
        )
        assert resp.status_code == 503
        body = resp.get_json()
        assert "ANTHROPIC_API_KEY" in body["errors"][0]

    def test_rate_limit_returns_429_on_11th_call(self, flask_client, monkeypatch):
        _patch_vision_success(monkeypatch)
        # 10 successes, then 11th gets 429. Use a varied IP to confirm
        # the header path works.
        headers = {"X-Forwarded-For": "1.2.3.4"}
        for i in range(10):
            resp = flask_client.post(
                "/api/import_screenshot",
                json=_make_request_body(),
                headers=headers,
            )
            assert resp.status_code == 200, f"call #{i+1} unexpectedly failed"
        # 11th call
        resp = flask_client.post(
            "/api/import_screenshot",
            json=_make_request_body(),
            headers=headers,
        )
        assert resp.status_code == 429
        body = resp.get_json()
        assert "rate limit" in body["errors"][0].lower()
        # Different IP still works — confirms per-IP isolation.
        resp2 = flask_client.post(
            "/api/import_screenshot",
            json=_make_request_body(),
            headers={"X-Forwarded-For": "5.6.7.8"},
        )
        assert resp2.status_code == 200

    def test_cf_connecting_ip_takes_precedence(self, flask_client, monkeypatch):
        """When CF-Connecting-IP is set, rate limiting must key on that
        and not on remote_addr / X-Forwarded-For. Otherwise a Cloudflare-
        fronted deployment rate-limits 'the tunnel' instead of 'the user'."""
        _patch_vision_success(monkeypatch)
        for _ in range(10):
            flask_client.post(
                "/api/import_screenshot",
                json=_make_request_body(),
                headers={"CF-Connecting-IP": "9.9.9.9"},
            )
        # CF-Connecting-IP=9.9.9.9 should be capped now.
        resp = flask_client.post(
            "/api/import_screenshot",
            json=_make_request_body(),
            headers={"CF-Connecting-IP": "9.9.9.9"},
        )
        assert resp.status_code == 429

    def test_failed_parse_does_not_burn_quota(self, flask_client, monkeypatch):
        """If Claude returns junk and the validator rejects it, the user
        shouldn't lose a rate-limit slot."""
        from analysis_board import server
        from analysis_board.screenshot_import import ScreenshotImportError

        def failing(image_b64, mime_type, **kwargs):
            raise ScreenshotImportError("Claude returned non-JSON: garbage")

        monkeypatch.setattr(server, "call_claude_vision", failing)
        # 11 failures in a row — none of them should hit the rate limit.
        for _ in range(11):
            resp = flask_client.post(
                "/api/import_screenshot",
                json=_make_request_body(),
                headers={"X-Forwarded-For": "1.2.3.4"},
            )
            assert resp.status_code == 200
            assert resp.get_json()["ok"] is False
            assert "non-JSON" in resp.get_json()["errors"][0]
