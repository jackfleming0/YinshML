"""Per-run manifest writer.

Captures everything needed to reproduce a training run at launch
(`manifest.json`) and to summarize final results at successful completion
(`manifest_final.json`).

See `CLOUD_TRAINING_PLAN.md` §1.4 for the intended schema.

Failures to write either manifest are logged loudly but must NEVER abort
training — the actual training loop is the source of truth, the manifest
is metadata.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("manifest")


def _run_git(args: list, cwd: Path) -> Optional[str]:
    """Run a git command, return stripped stdout or None on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def collect_git_info(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Return {git_sha, git_branch, git_dirty}.

    All three fields are None if `repo_root` is not a git repo or git is unavailable.
    """
    if repo_root is None:
        repo_root = Path.cwd()

    # Probe whether this is a git repo first — avoids three failed commands if not.
    inside = _run_git(["rev-parse", "--is-inside-work-tree"], repo_root)
    if inside != "true":
        logger.warning(
            "Not inside a git work tree at %s; manifest git fields will be null.",
            repo_root,
        )
        return {"git_sha": None, "git_branch": None, "git_dirty": None}

    sha = _run_git(["rev-parse", "HEAD"], repo_root)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    porcelain = _run_git(["status", "--porcelain"], repo_root)
    # `porcelain is None` means the command failed; empty string means clean.
    dirty = bool(porcelain) if porcelain is not None else None

    if sha is None:
        logger.warning("git rev-parse HEAD failed (repo may have no commits yet).")

    return {"git_sha": sha, "git_branch": branch, "git_dirty": dirty}


def describe_hardware() -> str:
    """Best-effort descriptive hardware string.

    darwin: `platform.processor()` + mac version (e.g. "arm Mac (14.5)").
    linux+cuda: `torch.cuda.get_device_name(0)`.
    linux (no cuda): first `model name` in /proc/cpuinfo.
    fallback: `platform.platform()`.
    """
    system = platform.system().lower()
    try:
        if system == "darwin":
            proc = platform.processor() or "unknown"
            mac_ver = platform.mac_ver()[0] or "unknown"
            # platform.processor() on Apple Silicon returns 'arm'; augment with
            # uname machine (arm64) to make the string more useful.
            machine = platform.machine() or ""
            return f"Apple {machine or proc} macOS {mac_ver}".strip()

        if system == "linux":
            try:
                import torch  # local import to avoid import-time cost

                if torch.cuda.is_available():
                    return torch.cuda.get_device_name(0)
            except Exception:
                pass
            # /proc/cpuinfo fallback
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if line.lower().startswith("model name"):
                            return line.split(":", 1)[1].strip()
            except OSError:
                pass

        return platform.platform()
    except Exception as exc:
        logger.warning("describe_hardware fell back to platform.platform(): %s", exc)
        try:
            return platform.platform()
        except Exception:
            return "unknown"


def _jsonify(value: Any) -> Any:
    """Make `value` JSON-serializable via a best-effort conversion.

    Handles dataclasses, Paths, tuples, sets, and nested dicts/lists.
    Unknown objects fall back to `repr()`.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonify(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(v) for v in value]
    # Fallback: repr so the manifest is always writable even if something exotic
    # (e.g. a numpy dtype) slipped into the config.
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def infer_encoder(config: Any, fallback_channels: Optional[int] = None) -> str:
    """Return "basic" or "enhanced" based on the loaded config.

    Order of preference:
      1. config["encoding"]["type"] if present (YAML convention in this repo).
      2. fallback_channels (15 -> enhanced, else basic) if provided.
      3. "basic" as the ultimate default.
    """
    if isinstance(config, dict):
        enc_cfg = config.get("encoding") or {}
        if isinstance(enc_cfg, dict):
            t = enc_cfg.get("type")
            if isinstance(t, str):
                return "enhanced" if t.strip().lower() == "enhanced" else "basic"
    if fallback_channels is not None:
        return "enhanced" if fallback_channels >= 15 else "basic"
    return "basic"


def build_launch_manifest(
    *,
    config: Any,
    device: str,
    encoder: str,
    total_moves: int,
    repo_root: Optional[Path] = None,
    start_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Assemble the dict that `manifest.json` should contain."""
    if start_time is None:
        start_time = datetime.now(timezone.utc)

    manifest: Dict[str, Any] = {}
    manifest.update(collect_git_info(repo_root=repo_root))
    manifest["config"] = _jsonify(config) if config is not None else None
    manifest["encoder"] = encoder
    manifest["total_moves"] = total_moves
    manifest["device"] = device
    manifest["hardware"] = describe_hardware()
    manifest["start_time_iso"] = start_time.isoformat()
    manifest["cloud_instance_id"] = os.environ.get("CLOUD_INSTANCE_ID") or None
    return manifest


def write_manifest(path: Path, manifest: Dict[str, Any]) -> bool:
    """Atomically-ish write `manifest` to `path` as pretty JSON.

    Returns True on success, False on any failure (failure is logged loudly
    but never raised — manifest writes must not take down training).
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=False, default=repr)
            f.write("\n")
        os.replace(tmp, path)
        return True
    except Exception as exc:
        logger.warning(
            "FAILED to write manifest at %s: %s. Training continues.", path, exc
        )
        return False


def build_final_manifest(
    launch_manifest: Dict[str, Any],
    *,
    end_time: Optional[datetime] = None,
    iterations_completed: int,
    final_anchor_win_rate: Optional[float],
    best_checkpoint_path: Optional[str],
    promotion_count: int,
) -> Dict[str, Any]:
    """Build the dict for `manifest_final.json`.

    Starts from the launch manifest (so fields like git_sha / config / hardware
    are preserved verbatim) and appends completion-time fields.
    """
    if end_time is None:
        end_time = datetime.now(timezone.utc)

    final = dict(launch_manifest)

    start_iso = launch_manifest.get("start_time_iso")
    duration_hours: Optional[float] = None
    if isinstance(start_iso, str):
        try:
            start_dt = datetime.fromisoformat(start_iso)
            duration_hours = (end_time - start_dt).total_seconds() / 3600.0
        except ValueError:
            duration_hours = None

    final["end_time_iso"] = end_time.isoformat()
    final["duration_hours"] = duration_hours
    final["iterations_completed"] = iterations_completed
    final["final_anchor_win_rate"] = final_anchor_win_rate
    final["best_checkpoint_path"] = best_checkpoint_path
    final["promotion_count"] = promotion_count
    return final
