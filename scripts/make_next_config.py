import argparse
from pathlib import Path
import yaml


def deep_merge(base: dict, overrides: dict) -> dict:
    out = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def find_latest_run(runs_dir: Path) -> Path:
    runs = [p for p in runs_dir.iterdir() if p.is_dir() and p.name[0].isdigit()]
    if not runs:
        raise SystemExit('No run directories found')
    return sorted(runs)[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description='Create next training config from latest suggestions')
    parser.add_argument('-b', '--base', type=str, required=True, help='Base YAML config (e.g., configs/training.yaml)')
    parser.add_argument('-o', '--output', type=str, default='configs/training_next.yaml', help='Output YAML path')
    parser.add_argument('-r', '--runs', type=str, default='runs', help='Runs directory')
    args = parser.parse_args()

    base_path = Path(args.base)
    base_cfg = yaml.safe_load(base_path.read_text())

    latest_run = find_latest_run(Path(args.runs))
    # Find last suggestion file
    sug_files = sorted(latest_run.glob('suggestions_iter_*.yaml'))
    if not sug_files:
        raise SystemExit('No suggestions files found in latest run')
    last_sug = sug_files[-1]
    overrides = yaml.safe_load(last_sug.read_text()) or {}

    next_cfg = deep_merge(base_cfg, overrides)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.safe_dump(next_cfg, f, sort_keys=False)

    print(f'Wrote next config to {out_path} (based on {last_sug})')


if __name__ == '__main__':
    main()


