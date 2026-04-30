"""Profile a single MCTS-driven game to measure engine fraction.

Plays one self-play game with a small neural net at small sim count, under
cProfile. Reports time spent in yinsh_ml.game.* (engine, the bitboard target)
vs yinsh_ml.search.* + yinsh_ml.network.* (MCTS + NN, NOT the target).

This is the load-bearing measurement for the bitboard port: the engine
fraction is the Amdahl's-law ceiling on speedup.

Usage:
    python scripts/profile_mcts_selfplay.py [--sims 64] [--games 1] [--top 25]
"""
import argparse
import cProfile
import pstats
import time

import torch

from yinsh_ml.game.game_state import GameState
from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.search.mcts import MCTS, MCTSConfig, EvaluationMode


def play_one_game(mcts: MCTS, max_moves: int = 500) -> int:
    state = GameState()
    moves_played = 0
    while not state.is_terminal() and moves_played < max_moves:
        probs = mcts.search(state, move_number=moves_played)
        valid = state.get_valid_moves()
        if not valid:
            break
        # Greedy pick from probs over valid moves.
        best = None
        best_p = -1.0
        encoder = mcts.state_encoder
        for m in valid:
            try:
                idx = encoder.move_to_index(m)
            except Exception:
                continue
            p = float(probs[idx])
            if p > best_p:
                best_p = p
                best = m
        if best is None:
            best = valid[0]
        if not state.make_move(best):
            break
        moves_played += 1
    return moves_played


def aggregate(stats: pstats.Stats, prefix: str) -> tuple[float, float]:
    """Return (cumtime, tottime) summed over functions whose filename
    contains the given prefix."""
    cum = tot = 0.0
    for func, stat in stats.stats.items():
        filename = func[0]
        if prefix in filename:
            # stat = (cc, nc, tt, ct, callers)
            tot += stat[2]
            cum_func = stat[3]
            # cumtime accounting double-counts when one tracked module calls
            # another tracked module — we only want self time aggregated.
            cum += cum_func
    return cum, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--device", type=str, default="cpu",
                    help="cpu / mps / cuda — cpu makes engine fraction more visible")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    torch.set_num_threads(1)

    cfg = MCTSConfig(
        num_simulations=args.sims,
        late_simulations=args.sims,
        c_puct=1.0,
        dirichlet_alpha=0.25,
        evaluation_mode=EvaluationMode.PURE_NEURAL,
        use_heuristic_evaluation=False,
    )
    network = NetworkWrapper(device=args.device)
    mcts = MCTS(network=network, config=cfg)

    pr = cProfile.Profile()
    pr.enable()
    t0 = time.perf_counter()
    total_moves = 0
    for _ in range(args.games):
        total_moves += play_one_game(mcts)
    elapsed = time.perf_counter() - t0
    pr.disable()

    print(f"\n=== Wall-clock summary ===")
    print(f"Games:        {args.games}")
    print(f"Sims/move:    {args.sims}")
    print(f"Total moves:  {total_moves}")
    print(f"Elapsed:      {elapsed:.2f}s")
    print(f"Moves/sec:    {total_moves / elapsed:.1f}")

    st = pstats.Stats(pr)

    # Self-time (tottime) aggregation by module — gives the speedup ceiling.
    engine_tot = sum(
        stat[2] for func, stat in st.stats.items()
        if "/yinsh_ml/game/" in func[0]
    )
    search_tot = sum(
        stat[2] for func, stat in st.stats.items()
        if "/yinsh_ml/search/" in func[0]
    )
    network_tot = sum(
        stat[2] for func, stat in st.stats.items()
        if "/yinsh_ml/network/" in func[0]
    )
    encoding_tot = sum(
        stat[2] for func, stat in st.stats.items()
        if "/yinsh_ml/utils/encoding" in func[0]
    )
    total_tot = sum(stat[2] for func, stat in st.stats.items())

    print(f"\n=== Self-time by layer (cumulative across all calls) ===")
    print(f"engine  yinsh_ml/game/...:           {engine_tot:6.2f}s "
          f"({100 * engine_tot / total_tot:5.1f}%)  <-- bitboard target")
    print(f"encoder yinsh_ml/utils/encoding...:  {encoding_tot:6.2f}s "
          f"({100 * encoding_tot / total_tot:5.1f}%)")
    print(f"search  yinsh_ml/search/...:         {search_tot:6.2f}s "
          f"({100 * search_tot / total_tot:5.1f}%)")
    print(f"network yinsh_ml/network/...:        {network_tot:6.2f}s "
          f"({100 * network_tot / total_tot:5.1f}%)")
    print(f"profiled total tottime:              {total_tot:6.2f}s")

    print(f"\n=== Top {args.top} by self time ===")
    st.sort_stats("tottime").print_stats(args.top)


if __name__ == "__main__":
    main()
