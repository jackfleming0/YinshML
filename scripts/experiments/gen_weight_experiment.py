#!/usr/bin/env python
"""Generate a matrix of training configs for a heuristic-weight experiment.

Takes a base training YAML and a set of named "arms", each pointing at a
heuristic weights JSON (or "default" for the hardcoded weights), and writes one
runnable config per arm plus a manifest. Each arm gets its own ``save_dir`` so
the downstream champion tournament can find every arm's checkpoints.

Example:
  python scripts/experiments/gen_weight_experiment.py \
      --base-config configs/ablation_baseline.yaml \
      --exp-name refit_v1 \
      --arm baseline=default \
      --arm refit_logreg=configs/heuristic_weights/refit_logreg.json \
      --arm refit_corr=configs/heuristic_weights/refit_corr.json

Then launch them with scripts/experiments/run_experiment.py.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml


def generate_arm_configs(base_config: dict, arms: dict, exp_name: str,
                         base_save_dir: str = "runs_experiments") -> dict:
    """Return ``{arm_name: config_dict}`` for each arm.

    ``arms`` maps arm name -> weights path string, or one of {"default",
    "none", None} for the hardcoded default weights.
    """
    out = {}
    for name, weights in arms.items():
        cfg = copy.deepcopy(base_config)
        sp = cfg.setdefault("self_play", {})
        if weights in (None, "default", "none"):
            sp["heuristic_weight_config_file"] = None
        else:
            sp["heuristic_weight_config_file"] = str(weights)
        cfg["save_dir"] = f"{base_save_dir}/{exp_name}/{name}"
        out[name] = cfg
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=WEIGHTS",
                    help="arm spec; WEIGHTS is a JSON path or 'default'. Repeatable.")
    ap.add_argument("--out-dir", default=None,
                    help="where to write arm configs (default: configs/experiments/<exp-name>)")
    ap.add_argument("--base-save-dir", default="runs_experiments")
    args = ap.parse_args(argv)

    arms = {}
    for spec in args.arm:
        if "=" not in spec:
            ap.error(f"--arm expects NAME=WEIGHTS, got {spec!r}")
        name, weights = spec.split("=", 1)
        arms[name] = weights

    with open(args.base_config) as fh:
        base_config = yaml.safe_load(fh)

    out_dir = Path(args.out_dir or f"configs/experiments/{args.exp_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = generate_arm_configs(base_config, arms, args.exp_name, args.base_save_dir)
    manifest = {"exp_name": args.exp_name, "base_config": args.base_config, "arms": {}}
    for name, cfg in configs.items():
        path = out_dir / f"{name}.yaml"
        with open(path, "w") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)
        manifest["arms"][name] = {
            "config": str(path),
            "save_dir": cfg["save_dir"],
            "weights": cfg["self_play"].get("heuristic_weight_config_file"),
        }
        print(f"  wrote {path}  (weights={manifest['arms'][name]['weights']})")

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
