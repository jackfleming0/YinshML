#!/usr/bin/env python3
"""Bounded campaign runner for iterative training experiments.

A campaign executes a finite sequence of training runs and updates config between
runs from either:
1) AutoTuner suggestions (autotune mode), or
2) A fixed experiment plan file (plan mode).

It is explicitly bounded by configurable constraints (max trials, time budget,
patience/no-improvement), so it cannot run forever.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

# Ensure project root on path when executed directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yinsh_ml.analysis.auto_tuner import AutoTuner


logger = logging.getLogger("run_campaign")


def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def read_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    return data or {}


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def iter_leaf_paths(data: Dict[str, Any], prefix: str = "") -> Iterable[Tuple[str, Any]]:
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from iter_leaf_paths(value, path)
        else:
            yield path, value


def set_nested(data: Dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    node = data
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def parse_allowed_params(csv_text: str) -> set[str]:
    return {x.strip() for x in (csv_text or "").split(",") if x.strip()}


def filter_overrides(overrides: Dict[str, Any], allowed_leaf_params: set[str]) -> Dict[str, Any]:
    if not allowed_leaf_params:
        return overrides

    filtered: Dict[str, Any] = {}
    for path, value in iter_leaf_paths(overrides):
        if path in allowed_leaf_params:
            set_nested(filtered, path, value)
    return filtered


def list_timestamp_run_dirs(parent: Path) -> List[Path]:
    if not parent.exists():
        return []
    return sorted([p for p in parent.iterdir() if p.is_dir() and p.name and p.name[0].isdigit()])


def list_iteration_dirs(run_dir: Path) -> List[Path]:
    dirs = [p for p in run_dir.glob("iteration_*") if p.is_dir()]

    def _iter_num(path: Path) -> int:
        try:
            return int(path.name.split("_")[-1])
        except Exception:
            return -1

    return sorted(dirs, key=_iter_num)


def load_metrics_series(run_dir: Path) -> List[Dict[str, Any]]:
    series: List[Dict[str, Any]] = []
    for iteration_dir in list_iteration_dirs(run_dir):
        metrics_path = iteration_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            metrics = json.loads(metrics_path.read_text())
            metrics["_iteration"] = int(iteration_dir.name.split("_")[-1])
            series.append(metrics)
        except Exception:
            continue
    return series


def score_run(run_dir: Path, objective: str, window: int) -> Tuple[Optional[float], Optional[float], int]:
    """Return (score, raw_avg, points_used).

    score is always "higher is better". For loss objectives, score = -raw_avg.
    """
    metrics = load_metrics_series(run_dir)
    if not metrics:
        return None, None, 0

    values: List[float] = []
    for row in metrics:
        value = row.get(objective)
        try:
            values.append(float(value))
        except Exception:
            pass

    if not values:
        return None, None, 0

    points = values[-window:] if window > 0 else values
    raw_avg = mean(points)

    minimize_objectives = {"policy_loss", "value_loss"}
    score = -raw_avg if objective in minimize_objectives else raw_avg
    return score, raw_avg, len(points)


def latest_suggestion_file(run_dir: Path) -> Optional[Path]:
    files = sorted(run_dir.glob("suggestions_iter_*.yaml"))
    return files[-1] if files else None


def load_plan(plan_path: Path) -> List[Dict[str, Any]]:
    data = read_yaml(plan_path)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    experiments = data.get("experiments", []) if isinstance(data, dict) else []
    if not isinstance(experiments, list):
        return []
    return [x for x in experiments if isinstance(x, dict)]


def get_new_run_dir(pre_existing: set[str], trial_runs_dir: Path) -> Optional[Path]:
    post = list_timestamp_run_dirs(trial_runs_dir)
    new_dirs = [p for p in post if p.name not in pre_existing]
    if new_dirs:
        return sorted(new_dirs, key=lambda p: p.stat().st_mtime)[-1]
    return post[-1] if post else None


def get_nested(data: Dict[str, Any], dotted_path: str, default: Any = None) -> Any:
    node: Any = data
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def parse_int_or_none(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except Exception:
        return None


def build_training_cmd(
    config_path: Path,
    trial_runs_dir: Path,
    iterations_per_run: Optional[int],
    resume_run_dir: Optional[Path] = None,
) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        str(ROOT / "scripts" / "run_training.py"),
        "-c",
        str(config_path),
    ]
    if resume_run_dir is not None:
        cmd += ["--resume", str(resume_run_dir)]
    else:
        cmd += ["--save-dir", str(trial_runs_dir)]
    if iterations_per_run is not None:
        cmd += ["--iterations", str(iterations_per_run)]
    return cmd


def apply_oom_backoff(
    cfg: Dict[str, Any],
    *,
    allow_worker_backoff: bool,
    min_workers: int,
    sims_backoff_factor: float,
    min_simulations: int,
    min_late_simulations: int,
    min_mcts_batch_size: int,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Apply one step of memory backoff and return (new_cfg, changes)."""
    next_cfg = deep_merge(cfg, {})
    changes: Dict[str, Dict[str, Any]] = {}

    # Step 1: Reduce batched MCTS footprint first.
    batch_size = parse_int_or_none(get_nested(next_cfg, "self_play.mcts_batch_size"))
    if batch_size is not None and batch_size > min_mcts_batch_size:
        new_batch_size = max(min_mcts_batch_size, batch_size // 2)
        if new_batch_size < batch_size:
            set_nested(next_cfg, "self_play.mcts_batch_size", new_batch_size)
            changes["self_play.mcts_batch_size"] = {"from": batch_size, "to": new_batch_size}
            return next_cfg, changes

    # Step 2: Optional worker backoff (disabled by default to preserve throughput).
    if allow_worker_backoff:
        workers = parse_int_or_none(get_nested(next_cfg, "self_play.num_workers"))
        if workers is not None and workers > min_workers:
            new_workers = max(min_workers, workers - 1)
            if new_workers < workers:
                set_nested(next_cfg, "self_play.num_workers", new_workers)
                changes["self_play.num_workers"] = {"from": workers, "to": new_workers}
                return next_cfg, changes

    # Step 3: Reduce simulations conservatively.
    early_sims = parse_int_or_none(get_nested(next_cfg, "self_play.num_simulations"))
    late_sims = parse_int_or_none(get_nested(next_cfg, "self_play.late_simulations"))

    if early_sims is not None:
        proposed_early = max(min_simulations, int(round(early_sims * sims_backoff_factor)))
        if proposed_early >= early_sims and early_sims > min_simulations:
            proposed_early = early_sims - 1
        if proposed_early < early_sims:
            set_nested(next_cfg, "self_play.num_simulations", proposed_early)
            changes["self_play.num_simulations"] = {"from": early_sims, "to": proposed_early}

    if late_sims is not None:
        early_after = parse_int_or_none(get_nested(next_cfg, "self_play.num_simulations")) or late_sims
        proposed_late = max(min_late_simulations, int(round(late_sims * sims_backoff_factor)))
        proposed_late = min(proposed_late, early_after)
        if proposed_late >= late_sims and late_sims > min_late_simulations:
            proposed_late = max(min_late_simulations, late_sims - 1)
            proposed_late = min(proposed_late, early_after)
        if proposed_late < late_sims:
            set_nested(next_cfg, "self_play.late_simulations", proposed_late)
            changes["self_play.late_simulations"] = {"from": late_sims, "to": proposed_late}

    return next_cfg, changes


def run_training_subprocess(cmd: Sequence[str], log_path: Path, dry_run: bool, append_log: bool = False) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        logger.info("[DRY RUN] Would execute: %s", " ".join(cmd))
        return 0

    mode = "a" if append_log else "w"
    with open(log_path, mode) as logf:
        if append_log:
            logf.write(f"\n\n=== Resume attempt at {now_ts()} ===\n")
        proc = subprocess.Popen(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                print(line, end="")
                logf.write(line)
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait(timeout=10)
            raise

        return proc.wait()


def prune_old_runs(trial_runs_dir: Path, keep: int) -> None:
    if keep <= 0 or not trial_runs_dir.exists():
        return

    runs = sorted(
        list_timestamp_run_dirs(trial_runs_dir),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in runs[keep:]:
        try:
            shutil.rmtree(old)
            logger.info("Pruned old run directory: %s", old)
        except Exception as exc:
            logger.warning("Failed to prune %s: %s", old, exc)


def run_anchor_eval(
    run_dir: Path,
    anchor_run_dir: Path,
    anchor_opponents: List[int],
    games_per_side: int,
    device: str,
) -> Optional[float]:
    """Run optional fixed-opponent anchor evaluation.

    Score is mean win-rate of latest checkpoint vs selected opponent checkpoints.
    """
    try:
        from scripts.cross_era_tournament import load_model, run_match
    except Exception as exc:
        logger.warning("Anchor eval unavailable (import failed): %s", exc)
        return None

    candidate_iters = list_iteration_dirs(run_dir)
    if not candidate_iters:
        return None
    latest_iter = int(candidate_iters[-1].name.split("_")[-1])
    candidate_ckpt = run_dir / f"iteration_{latest_iter}" / f"checkpoint_iteration_{latest_iter}.pt"
    if not candidate_ckpt.exists():
        return None

    try:
        candidate = load_model(str(candidate_ckpt), device)
    except Exception as exc:
        logger.warning("Anchor eval skipped (candidate load failed): %s", exc)
        return None

    win_rates: List[float] = []
    for opp_iter in anchor_opponents:
        opp_ckpt = anchor_run_dir / f"iteration_{opp_iter}" / f"checkpoint_iteration_{opp_iter}.pt"
        if not opp_ckpt.exists():
            logger.warning("Anchor opponent checkpoint missing: %s", opp_ckpt)
            continue

        try:
            opponent = load_model(str(opp_ckpt), device)
            r1, r2 = run_match(candidate, opponent, f"iter_{latest_iter}", f"iter_{opp_iter}", games_per_side)
            # Candidate wins as white in r1 plus as black in r2.
            wins = r1.white_wins + r2.black_wins
            losses = r1.black_wins + r2.white_wins
            draws = r1.draws + r2.draws
            total = wins + losses + draws
            if total > 0:
                win_rates.append(wins / total)
        except Exception as exc:
            logger.warning("Anchor match failed vs iter_%s: %s", opp_iter, exc)

    if not win_rates:
        return None
    return mean(win_rates)


def save_state(path: Path, state: Dict[str, Any]) -> None:
    state["updated_at"] = now_ts()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def load_state(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded training campaign with config adaptation")
    parser.add_argument("--base-config", type=str, default=str(ROOT / "configs" / "training.yaml"))
    parser.add_argument("--runs-dir", type=str, default=str(ROOT / "runs"))
    parser.add_argument("--campaign-dir", type=str, default=str(ROOT / "runs" / "campaigns"))
    parser.add_argument("--resume", type=str, default=None, help="Path to existing campaign_state.json")

    parser.add_argument("--mode", choices=["autotune", "plan"], default="autotune")
    parser.add_argument("--experiment-plan", type=str, default=None, help="YAML plan file for mode=plan")

    parser.add_argument("--max-trials", type=int, default=3)
    parser.add_argument("--max-total-hours", type=float, default=72.0)
    parser.add_argument("--iterations-per-run", type=int, default=None)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-improvement", type=float, default=0.005)

    parser.add_argument(
        "--objective",
        type=str,
        default="tournament_win_rate",
        choices=["tournament_win_rate", "tournament_rating", "policy_loss", "value_loss"],
    )
    parser.add_argument("--metric-window", type=int, default=5)

    parser.add_argument(
        "--allowed-params",
        type=str,
        default=(
            "trainer.discrimination_weight,trainer.value_head_lr_factor,"
            "self_play.final_temp,self_play.games_per_iteration"
        ),
        help="Comma-separated dotted param paths allowed to change in autotune mode",
    )

    parser.add_argument("--anchor-eval", action="store_true", help="Run optional fixed-opponent anchor evaluation")
    parser.add_argument("--anchor-run-dir", type=str, default=None, help="Reference run for anchor opponents")
    parser.add_argument("--anchor-opponents", type=str, default="3,9", help="Comma-separated iteration ids in anchor run")
    parser.add_argument("--anchor-games-per-side", type=int, default=20)
    parser.add_argument("--anchor-device", type=str, default="mps")

    parser.add_argument("--cleanup-keep", type=int, default=0, help="Prune older trial run dirs, keep N newest (0 disables)")
    parser.add_argument("--oom-backoff", action="store_true", help="Retry SIGKILL/OOM failures with reduced memory settings")
    parser.add_argument("--oom-max-retries", type=int, default=3, help="Max retries per trial after SIGKILL/OOM")
    parser.add_argument(
        "--oom-allow-worker-backoff",
        action="store_true",
        help="Allow OOM retries to reduce self_play.num_workers (off by default)",
    )
    parser.add_argument("--oom-min-workers", type=int, default=2, help="Lower bound when worker backoff is enabled")
    parser.add_argument(
        "--oom-sims-backoff-factor",
        type=float,
        default=0.9,
        help="Multiplicative decay for num_simulations/late_simulations during OOM retries",
    )
    parser.add_argument("--oom-min-simulations", type=int, default=128, help="Lower bound for self_play.num_simulations")
    parser.add_argument("--oom-min-late-simulations", type=int, default=80, help="Lower bound for self_play.late_simulations")
    parser.add_argument("--oom-min-mcts-batch-size", type=int, default=8, help="Lower bound for self_play.mcts_batch_size")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if not (0.0 < args.oom_sims_backoff_factor < 1.0):
        raise SystemExit("--oom-sims-backoff-factor must be in the range (0, 1)")

    base_config_path = Path(args.base_config)
    runs_dir = Path(args.runs_dir)
    campaign_base_dir = Path(args.campaign_dir)

    if args.resume:
        state_path = Path(args.resume)
        state = load_state(state_path)
        campaign_root = state_path.parent
        logger.info("Resuming campaign from %s", state_path)
    else:
        campaign_id = time.strftime("campaign_%Y%m%d_%H%M%S")
        campaign_root = campaign_base_dir / campaign_id
        state_path = campaign_root / "campaign_state.json"

        state = {
            "campaign_id": campaign_id,
            "created_at": now_ts(),
            "updated_at": now_ts(),
            "status": "running",
            "base_config": str(base_config_path),
            "runs_dir": str(runs_dir),
            "constraints": {
                "max_trials": args.max_trials,
                "max_total_hours": args.max_total_hours,
                "patience": args.patience,
                "min_improvement": args.min_improvement,
                "objective": args.objective,
                "metric_window": args.metric_window,
                "mode": args.mode,
                "allowed_params": args.allowed_params,
                "oom_backoff": args.oom_backoff,
                "oom_max_retries": args.oom_max_retries,
                "oom_allow_worker_backoff": args.oom_allow_worker_backoff,
                "oom_min_workers": args.oom_min_workers,
                "oom_sims_backoff_factor": args.oom_sims_backoff_factor,
                "oom_min_simulations": args.oom_min_simulations,
                "oom_min_late_simulations": args.oom_min_late_simulations,
                "oom_min_mcts_batch_size": args.oom_min_mcts_batch_size,
            },
            "best_score": None,
            "best_trial": None,
            "no_improve_count": 0,
            "trials": [],
        }
        save_state(state_path, state)

    campaign_root.mkdir(parents=True, exist_ok=True)
    configs_dir = campaign_root / "configs"
    logs_dir = campaign_root / "logs"
    suggestions_dir = campaign_root / "suggestions"

    trial_runs_dir = runs_dir / campaign_root.name
    trial_runs_dir.mkdir(parents=True, exist_ok=True)

    allowed_params = parse_allowed_params(args.allowed_params)

    plan: List[Dict[str, Any]] = []
    if args.mode == "plan":
        if not args.experiment_plan:
            raise SystemExit("--experiment-plan is required when --mode plan")
        plan = load_plan(Path(args.experiment_plan))
        if not plan:
            raise SystemExit("Experiment plan is empty or invalid")

    base_cfg = read_yaml(base_config_path)

    completed_trials = len(state.get("trials", []))
    trial_index = completed_trials

    start_wall = time.time()
    if state.get("created_at"):
        # keep wall budget relative to actual campaign process start for this invocation
        start_wall = time.time()

    current_cfg = base_cfg
    if trial_index > 0:
        # Resume from last written trial config if present.
        last_cfg = configs_dir / f"trial_{trial_index - 1:03d}.yaml"
        if last_cfg.exists():
            current_cfg = read_yaml(last_cfg)

    logger.info("Campaign root: %s", campaign_root)
    logger.info("Trial run output dir: %s", trial_runs_dir)

    try:
        while trial_index < args.max_trials:
            elapsed_h = (time.time() - start_wall) / 3600.0
            if elapsed_h >= args.max_total_hours:
                state["status"] = "stopped_time_budget"
                state["stop_reason"] = f"Reached max_total_hours={args.max_total_hours}"
                break

            if state.get("no_improve_count", 0) >= args.patience:
                state["status"] = "stopped_patience"
                state["stop_reason"] = f"No improvement for {state['no_improve_count']} trial(s)"
                break

            # Build trial config.
            trial_name = f"trial_{trial_index:03d}"
            config_path = configs_dir / f"{trial_name}.yaml"
            trial_mode = args.mode
            overrides_applied: Dict[str, Any] = {}
            suggestion_source: Optional[str] = None

            if trial_index == 0 and not config_path.exists():
                write_yaml(config_path, current_cfg)
            elif args.mode == "plan":
                plan_entry = plan[trial_index] if trial_index < len(plan) else None
                if plan_entry is None:
                    state["status"] = "completed_plan"
                    state["stop_reason"] = "Exhausted experiment plan entries"
                    break

                overrides_applied = plan_entry.get("overrides", {}) if isinstance(plan_entry, dict) else {}
                trial_cfg = deep_merge(base_cfg, overrides_applied)
                current_cfg = trial_cfg
                write_yaml(config_path, trial_cfg)
            else:
                # autotune mode
                if trial_index > 0:
                    prev_trial = state["trials"][-1]
                    prev_run_dir = Path(prev_trial["run_dir"]) if prev_trial.get("run_dir") else None

                    suggestions: Dict[str, Any] = {}
                    if prev_run_dir and prev_run_dir.exists():
                        suggestion_file = latest_suggestion_file(prev_run_dir)
                        if suggestion_file:
                            suggestions = read_yaml(suggestion_file)
                            suggestion_source = str(suggestion_file)

                        if not suggestions:
                            tuner = AutoTuner(prev_run_dir)
                            suggestions = tuner.propose(current_cfg)
                            suggestion_source = "AutoTuner.propose"

                    filtered = filter_overrides(suggestions, allowed_params)
                    overrides_applied = filtered
                    current_cfg = deep_merge(current_cfg, filtered)

                    out_suggestions = suggestions_dir / f"{trial_name}_applied_overrides.yaml"
                    write_yaml(out_suggestions, overrides_applied)

                write_yaml(config_path, current_cfg)

            pre_dirs = {p.name for p in list_timestamp_run_dirs(trial_runs_dir)}
            cmd = build_training_cmd(
                config_path=config_path,
                trial_runs_dir=trial_runs_dir,
                iterations_per_run=args.iterations_per_run,
                resume_run_dir=None,
            )

            trial_record: Dict[str, Any] = {
                "index": trial_index,
                "name": trial_name,
                "mode": trial_mode,
                "config_path": str(config_path),
                "command": cmd,
                "started_at": now_ts(),
                "status": "running",
                "suggestion_source": suggestion_source,
                "overrides_applied": overrides_applied,
                "objective": args.objective,
                "metric_window": args.metric_window,
                "attempts": [],
            }

            state["trials"].append(trial_record)
            save_state(state_path, state)

            logger.info("Starting %s", trial_name)
            logger.info("Config: %s", config_path)
            log_path = logs_dir / f"{trial_name}.log"
            run_dir: Optional[Path] = None
            resume_run_dir: Optional[Path] = None
            max_attempts = 1 + max(0, args.oom_max_retries) if args.oom_backoff else 1
            exit_code = 0
            attempt = 0

            while attempt < max_attempts:
                if attempt > 0:
                    cmd = build_training_cmd(
                        config_path=config_path,
                        trial_runs_dir=trial_runs_dir,
                        iterations_per_run=args.iterations_per_run,
                        resume_run_dir=resume_run_dir,
                    )

                logger.info("Trial %s attempt %d/%d", trial_name, attempt + 1, max_attempts)
                logger.info("Command: %s", " ".join(cmd))
                exit_code = run_training_subprocess(
                    cmd,
                    log_path=log_path,
                    dry_run=args.dry_run,
                    append_log=(attempt > 0),
                )

                if run_dir is None:
                    run_dir = get_new_run_dir(pre_dirs, trial_runs_dir)

                trial_record["attempts"].append(
                    {
                        "attempt": attempt,
                        "command": cmd,
                        "exit_code": int(exit_code),
                        "ended_at": now_ts(),
                        "resume_run_dir": str(resume_run_dir) if resume_run_dir else None,
                    }
                )
                save_state(state_path, state)

                if exit_code == 0:
                    break

                is_sigkill = exit_code == -9
                can_retry = args.oom_backoff and is_sigkill and (attempt + 1) < max_attempts
                if not can_retry:
                    break

                if run_dir is None:
                    logger.warning("Cannot OOM-retry %s: no run directory found to resume from", trial_name)
                    break

                next_cfg, backoff_changes = apply_oom_backoff(
                    current_cfg,
                    allow_worker_backoff=args.oom_allow_worker_backoff,
                    min_workers=args.oom_min_workers,
                    sims_backoff_factor=args.oom_sims_backoff_factor,
                    min_simulations=args.oom_min_simulations,
                    min_late_simulations=args.oom_min_late_simulations,
                    min_mcts_batch_size=args.oom_min_mcts_batch_size,
                )
                if not backoff_changes:
                    logger.warning("OOM backoff exhausted; no further safe reductions available for %s", trial_name)
                    break

                logger.warning("Detected SIGKILL/OOM for %s; applying backoff: %s", trial_name, backoff_changes)
                current_cfg = next_cfg
                write_yaml(config_path, current_cfg)
                resume_run_dir = run_dir
                trial_record.setdefault("oom_backoff_history", []).append(
                    {
                        "attempt": attempt + 1,
                        "changes": backoff_changes,
                        "at": now_ts(),
                    }
                )
                save_state(state_path, state)
                attempt += 1

            trial_record["run_dir"] = str(run_dir) if run_dir else None
            trial_record["log_path"] = str(log_path)
            trial_record["ended_at"] = now_ts()
            trial_record["exit_code"] = int(exit_code)
            trial_record["num_attempts"] = len(trial_record.get("attempts", []))

            if exit_code != 0:
                trial_record["status"] = "failed"
                state["status"] = "failed"
                state["stop_reason"] = f"Training command failed with exit_code={exit_code}"
                save_state(state_path, state)
                break

            trial_record["status"] = "completed"

            if run_dir and run_dir.exists():
                score, raw_value, points = score_run(run_dir, objective=args.objective, window=args.metric_window)
                trial_record["score"] = score
                trial_record["objective_avg"] = raw_value
                trial_record["points_used"] = points

                if args.anchor_eval and args.anchor_run_dir:
                    opponent_ids = [int(x.strip()) for x in args.anchor_opponents.split(",") if x.strip()]
                    anchor_score = run_anchor_eval(
                        run_dir=run_dir,
                        anchor_run_dir=Path(args.anchor_run_dir),
                        anchor_opponents=opponent_ids,
                        games_per_side=args.anchor_games_per_side,
                        device=args.anchor_device,
                    )
                    trial_record["anchor_win_rate"] = anchor_score

                improved = False
                best_score = state.get("best_score")
                if score is not None:
                    if best_score is None or score > (best_score + args.min_improvement):
                        improved = True
                        state["best_score"] = score
                        state["best_trial"] = trial_index
                        state["no_improve_count"] = 0
                    else:
                        state["no_improve_count"] = int(state.get("no_improve_count", 0)) + 1
                else:
                    state["no_improve_count"] = int(state.get("no_improve_count", 0)) + 1

                trial_record["improved"] = improved
            else:
                trial_record["score"] = None
                trial_record["objective_avg"] = None
                trial_record["points_used"] = 0
                state["no_improve_count"] = int(state.get("no_improve_count", 0)) + 1

            save_state(state_path, state)

            if args.cleanup_keep > 0:
                prune_old_runs(trial_runs_dir, keep=args.cleanup_keep)

            trial_index += 1

        if state.get("status") == "running":
            state["status"] = "completed"
            state["stop_reason"] = "Reached max_trials" if trial_index >= args.max_trials else "Campaign finished"

        save_state(state_path, state)

    except KeyboardInterrupt:
        state["status"] = "interrupted"
        state["stop_reason"] = "KeyboardInterrupt"
        save_state(state_path, state)
        raise

    logger.info("Campaign finished with status: %s", state.get("status"))
    logger.info("State written to: %s", state_path)


if __name__ == "__main__":
    main()
