"""Profile the pure-Python YINSH engine via random playouts.

Isolates engine cost — no neural net, no MCTS, just legal-move enumeration
and make_move. Establishes the baseline for the bitboard port.

Usage:
    python scripts/profile_engine.py [--games N] [--top K]
"""
import argparse
import cProfile
import pstats
import random
import time

from yinsh_ml.game.game_state import GameState, GamePhase


def play_random_game(rng: random.Random, max_moves: int = 500) -> int:
    state = GameState()
    moves_played = 0
    while not state.is_terminal() and moves_played < max_moves:
        moves = state.get_valid_moves()
        if not moves:
            break
        m = rng.choice(moves)
        if not state.make_move(m):
            break
        moves_played += 1
    return moves_played


def run(num_games: int) -> tuple[int, float]:
    rng = random.Random(0xC0FFEE)
    t0 = time.perf_counter()
    total_moves = 0
    for _ in range(num_games):
        total_moves += play_random_game(rng)
    elapsed = time.perf_counter() - t0
    return total_moves, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--no-profile", action="store_true",
                    help="Wall-clock only, skip cProfile (cleaner timing)")
    args = ap.parse_args()

    if args.no_profile:
        moves, elapsed = run(args.games)
    else:
        pr = cProfile.Profile()
        pr.enable()
        moves, elapsed = run(args.games)
        pr.disable()

    print(f"\n=== Wall-clock summary ===")
    print(f"Games:        {args.games}")
    print(f"Total moves:  {moves}")
    print(f"Elapsed:      {elapsed:.2f}s")
    print(f"Games/sec:    {args.games / elapsed:.1f}")
    print(f"Moves/sec:    {moves / elapsed:.0f}")
    print(f"us / move:    {1e6 * elapsed / moves:.1f}")

    if not args.no_profile:
        print(f"\n=== Top {args.top} by cumulative time ===")
        st = pstats.Stats(pr).sort_stats("cumulative")
        st.print_stats(args.top)
        print(f"\n=== Top {args.top} by self time ===")
        st.sort_stats("tottime").print_stats(args.top)


if __name__ == "__main__":
    main()
