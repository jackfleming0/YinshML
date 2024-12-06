class MetricsSaver:
    def __init__(self, base_dir: Path):
        self.metrics_dir = base_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True, parents=True)

        # Create subdirectories
        self.subdirs = {
            'phase': self.metrics_dir / 'phase_metrics',
            'value': self.metrics_dir / 'value_metrics',
            'critical': self.metrics_dir / 'critical_positions',
            'summary': self.metrics_dir / 'summaries'
        }

        for dir_path in self.subdirs.values():
            dir_path.mkdir(exist_ok=True)

    def save_phase_metrics(self, iteration: int, metrics: Dict):
        path = self.subdirs['phase'] / f'phase_metrics_{iteration}.json'
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)

    def save_value_metrics(self, iteration: int, metrics: Dict):
        path = self.subdirs['value'] / f'value_metrics_{iteration}.json'
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)

    def save_critical_positions(self, iteration: int, positions: List[Dict]):
        path = self.subdirs['critical'] / f'critical_pos_{iteration}.json'
        with open(path, 'w') as f:
            json.dump(positions, f, indent=2)

    def save_iteration_summary(self, iteration: int, metrics: Dict):
        path = self.subdirs['summary'] / f'summary_{iteration}.json'
        with open(path, 'w') as f:
            json.dump({
                'iteration': iteration,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            }, f, indent=2)