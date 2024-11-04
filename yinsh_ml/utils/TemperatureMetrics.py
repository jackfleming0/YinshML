class TemperatureMetrics:
    """Tracks the effectiveness of temperature annealing."""

    def __init__(self):
        self.early_game_stats = []  # High temperature phase
        self.mid_game_stats = []  # Annealing phase
        self.late_game_stats = []  # Low temperature phase

    def add_game_result(self, states: List[np.ndarray], policies: List[np.ndarray],
                        outcome: int, annealing_steps: int):
        """Record game statistics based on game phase."""
        game_length = len(states)

        # Split game into phases
        early_moves = states[:annealing_steps // 3]
        mid_moves = states[annealing_steps // 3:annealing_steps]
        late_moves = states[annealing_steps:]

        # Calculate policy sharpness (lower entropy = more focused decisions)
        def get_policy_entropy(policies_subset):
            if not policies_subset:
                return 0.0
            entropy = -np.sum(policies_subset * np.log(policies_subset + 1e-8), axis=1)
            return np.mean(entropy)

        # Record statistics for each phase
        if early_moves:
            self.early_game_stats.append({
                'policy_entropy': get_policy_entropy(policies[:len(early_moves)]),
                'outcome': outcome
            })

        if mid_moves:
            self.mid_game_stats.append({
                'policy_entropy': get_policy_entropy(policies[len(early_moves):len(early_moves) + len(mid_moves)]),
                'outcome': outcome
            })

        if late_moves:
            self.late_game_stats.append({
                'policy_entropy': get_policy_entropy(policies[len(early_moves) + len(mid_moves):]),
                'outcome': outcome
            })

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for each game phase."""

        def phase_stats(stats):
            if not stats:
                return {}

            wins = sum(1 for s in stats if s['outcome'] == 1)
            total = len(stats)
            avg_entropy = np.mean([s['policy_entropy'] for s in stats])

            return {
                'win_rate': wins / total if total > 0 else 0,
                'avg_policy_entropy': avg_entropy,
                'num_games': total
            }

        return {
            'early_game': phase_stats(self.early_game_stats),
            'mid_game': phase_stats(self.mid_game_stats),
            'late_game': phase_stats(self.late_game_stats)
        }