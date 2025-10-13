import json
from pathlib import Path
from typing import Dict, Any


class AutoTuner:
    """Heuristic analyzer that proposes next-step config changes based on metrics."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)

    def _load_last_metrics(self) -> Dict[str, Any]:
        iterations = sorted((p for p in self.run_dir.glob('iteration_*') if p.is_dir()), key=lambda p: p.name)
        if not iterations:
            return {}
        last_iter = iterations[-1]
        metrics_path = last_iter / 'metrics.json'
        if not metrics_path.exists():
            return {}
        try:
            return json.loads(metrics_path.read_text())
        except Exception:
            return {}

    def propose(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Return a dict of suggested overrides for the next run."""
        metrics = self._load_last_metrics()
        suggestions: Dict[str, Any] = {}

        # Fallbacks
        training = metrics.get('training', {})
        value_loss = float(training.get('value_loss', 0.0))
        policy_loss = float(training.get('policy_loss', 0.0))
        win_rate = float(metrics.get('win_rate', 0.0)) if 'win_rate' in metrics else 0.0

        # Start from current settings
        sp = dict(base_config.get('self_play', {}))
        tr = dict(base_config.get('trainer', {}))

        # Heuristics
        # 1) If value loss is high and sign match (approx via value loss) is poor, increase value_head_lr a bit
        if value_loss > 0.6:
            tr['value_head_lr_factor'] = min(8.0, float(tr.get('value_head_lr_factor', 5.0)) * 1.2)

        # 2) If policy loss is very high, modestly increase sims to improve targets; also reduce temp
        if policy_loss > 6.0:
            sp['num_simulations'] = int(max( sp.get('num_simulations', 96) * 1.25, sp.get('num_simulations', 96) + 16))
            sp['late_simulations'] = sp.get('late_simulations', sp['num_simulations'])
            sp['final_temp'] = max(0.05, float(sp.get('final_temp', 0.1)) * 0.9)

        # 3) If win rate (vs prior) is low or unchanged and buffer is small, increase games_per_iteration a bit
        if win_rate < 0.52:
            sp['games_per_iteration'] = int(max(2, sp.get('games_per_iteration', 50) * 1.2))

        # 4) Clamp simulation switch if too late
        sp['simulation_switch_ply'] = int(min( max(4, sp.get('simulation_switch_ply', 20)), 30))

        suggestions['self_play'] = sp
        suggestions['trainer'] = tr
        return suggestions

    def write_suggestions(self, base_config_path: Path, out_path: Path) -> Dict[str, Any]:
        cfg = json.loads(base_config_path.read_text()) if base_config_path.suffix == '.json' else {}
        if base_config_path.suffix in ('.yml', '.yaml'):
            import yaml
            cfg = yaml.safe_load(base_config_path.read_text())
        props = self.propose(cfg)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            import yaml
            yaml.safe_dump(props, f)
        return props



