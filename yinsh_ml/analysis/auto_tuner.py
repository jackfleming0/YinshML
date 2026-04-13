import json
from pathlib import Path
from typing import Any, Dict


class AutoTuner:
    """Heuristic analyzer that proposes next-step config changes based on metrics."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    def _load_last_metrics(self) -> Dict[str, Any]:
        iterations = sorted(
            (p for p in self.run_dir.glob('iteration_*') if p.is_dir()),
            key=lambda p: int(p.name.split('_')[-1]) if p.name.split('_')[-1].isdigit() else -1,
        )
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

    def _extract_metric(self, metrics: Dict[str, Any], key: str, fallback: float = 0.0) -> float:
        """Extract metric from known shapes.

        Supports both current top-level metrics.json keys and older nested structures.
        """
        if key in metrics:
            return self._safe_float(metrics.get(key), fallback)

        training = metrics.get('training', {})
        if isinstance(training, dict) and key in training:
            return self._safe_float(training.get(key), fallback)

        evaluation = metrics.get('evaluation', {})
        if isinstance(evaluation, dict):
            tournament = evaluation.get('tournament', {})
            if isinstance(tournament, dict):
                if key == 'tournament_win_rate' and 'win_rate' in tournament:
                    return self._safe_float(tournament.get('win_rate'), fallback)
                if key == 'tournament_rating' and 'rating' in tournament:
                    return self._safe_float(tournament.get('rating'), fallback)

        return fallback

    def propose(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Return a dict of suggested overrides for the next run."""
        metrics = self._load_last_metrics()

        value_loss = self._extract_metric(metrics, 'value_loss', 0.0)
        policy_loss = self._extract_metric(metrics, 'policy_loss', 0.0)
        tournament_win_rate = self._extract_metric(metrics, 'tournament_win_rate', 0.0)

        # Start from current settings
        sp = dict(base_config.get('self_play', {}))
        tr = dict(base_config.get('trainer', {}))

        # Heuristics
        # 1) If value loss is high, increase value-head learning pressure slightly.
        if value_loss > 1.9:
            tr['value_head_lr_factor'] = min(8.0, float(tr.get('value_head_lr_factor', 5.0)) * 1.1)

        # 2) If policy loss is high, modestly increase sims and reduce final temperature.
        if policy_loss > 5.5:
            sims = int(sp.get('num_simulations', 96))
            sp['num_simulations'] = int(max(sims + 16, sims * 1.2))
            sp['late_simulations'] = int(max(sp.get('late_simulations', sims), sp['num_simulations'] * 0.6))
            sp['final_temp'] = max(0.05, float(sp.get('final_temp', 0.1)) * 0.9)

        # 3) If tournament win rate is weak, add more self-play diversity.
        if tournament_win_rate < 0.52:
            gpi = int(sp.get('games_per_iteration', 100))
            sp['games_per_iteration'] = int(max(20, round(gpi * 1.1)))

        # 4) Keep simulation switch in sane range.
        sp['simulation_switch_ply'] = int(min(max(4, int(sp.get('simulation_switch_ply', 20))), 30))

        return {
            'self_play': sp,
            'trainer': tr,
        }

    def write_suggestions(self, base_config_path: Path, out_path: Path) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        if base_config_path.suffix in ('.yml', '.yaml'):
            import yaml

            cfg = yaml.safe_load(base_config_path.read_text()) or {}
        elif base_config_path.suffix == '.json':
            cfg = json.loads(base_config_path.read_text())

        props = self.propose(cfg)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            import yaml

            yaml.safe_dump(props, f, sort_keys=False)
        return props
