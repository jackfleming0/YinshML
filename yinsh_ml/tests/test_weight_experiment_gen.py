"""Tests for the weight-experiment generator + launcher cores (torch-free)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "experiments"))
import gen_weight_experiment as gen  # noqa: E402
import run_experiment as runner  # noqa: E402


BASE = {
    "version": 1,
    "num_iterations": 50,
    "save_dir": "runs/base",
    "self_play": {"evaluation_mode": "hybrid", "heuristic_weight": 0.3},
    "trainer": {"batch_size": 256},
}


def test_generate_arm_configs_sets_weights_and_save_dirs():
    arms = {"baseline": "default", "refit": "configs/heuristic_weights/refit.json"}
    cfgs = gen.generate_arm_configs(BASE, arms, "exp1", base_save_dir="runs_x")

    assert cfgs["baseline"]["self_play"]["heuristic_weight_config_file"] is None
    assert (cfgs["refit"]["self_play"]["heuristic_weight_config_file"]
            == "configs/heuristic_weights/refit.json")
    # distinct save dirs per arm, namespaced by experiment
    assert cfgs["baseline"]["save_dir"] == "runs_x/exp1/baseline"
    assert cfgs["refit"]["save_dir"] == "runs_x/exp1/refit"
    # base config untouched (deepcopy)
    assert "heuristic_weight_config_file" not in BASE["self_play"]


def test_build_commands_formats_template():
    manifest = {"arms": {
        "a": {"config": "c/a.yaml", "save_dir": "runs/a"},
        "b": {"config": "c/b.yaml", "save_dir": "runs/b"},
    }}
    cmds = runner.build_commands(manifest, runner.DEFAULT_TRAIN_CMD)
    assert "c/a.yaml" in cmds["a"] and "runs/a" in cmds["a"]
    assert "c/b.yaml" in cmds["b"]


def test_run_arms_dry_run_lists_without_executing(capsys):
    manifest = {"arms": {"a": {"config": "x.yaml", "save_dir": "runs/a"}}}
    results = runner.run_arms(manifest, runner.DEFAULT_TRAIN_CMD, 1, dry_run=True)
    assert results[0]["dry_run"] is True


def test_run_one_captures_returncode_and_writes_log(tmp_path):
    res_ok = runner._run_one("ok", 'python -c "print(1)"', str(tmp_path / "ok"))
    res_bad = runner._run_one("bad", 'python -c "import sys; sys.exit(3)"', str(tmp_path / "bad"))
    assert res_ok["returncode"] == 0
    assert res_bad["returncode"] == 3
    assert Path(res_ok["log"]).exists()


def test_run_arms_executes_in_parallel_and_reports_status(tmp_path):
    manifest = {"arms": {
        "a": {"config": "x", "save_dir": str(tmp_path / "a")},
        "b": {"config": "x", "save_dir": str(tmp_path / "b")},
    }}
    # template ignores placeholders; both arms just exit 0
    results = runner.run_arms(manifest, "python -c \"pass\"", max_parallel=2)
    assert {r["arm"] for r in results} == {"a", "b"}
    assert all(r["returncode"] == 0 for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
