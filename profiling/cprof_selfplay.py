"""cProfile a few self-play games to size where worker CPU time goes.
Run with the box interpreter: /venv/main/bin/python profiling/cprof_selfplay.py
pure_neural + 200 sims approximates iter5 self-play (heuristic_weight annealed ~0)."""
import cProfile, pstats, io
import numpy as np
from yinsh_ml.game.game_state import GameState
from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.training.self_play import MCTS
from yinsh_ml.utils.encoding import StateEncoder

MODEL = "runs_symmetry/20260601_105750/iteration_4/checkpoint_iteration_4_ema.pt"
nw = NetworkWrapper(model_path=MODEL)
enc = StateEncoder()


def make_mcts():
    return MCTS(
        network=nw, evaluation_mode="pure_neural", heuristic_evaluator=None,
        heuristic_weight=0.0, num_simulations=64, late_simulations=64,
        simulation_switch_ply=20, c_puct=1.0, dirichlet_alpha=0.3, value_weight=1.0,
        max_depth=150, epsilon_mix_start=0.0, epsilon_mix_end=0.0,
        epsilon_mix_taper_moves=1, initial_temp=1.0, final_temp=0.1,
        annealing_steps=30, temp_clamp_fraction=0.6, enable_subtree_reuse=True,
        fpu_reduction=0.25,
    )


def play(rng):
    m = make_mcts(); s = GameState(); mc = 0
    while not s.is_terminal() and mc < 70:
        valid = s.get_valid_moves()
        if not valid:
            break
        pol = m.search_batch(s, mc, batch_size=64)
        probs = np.zeros(len(valid))
        for i, mv in enumerate(valid):
            idx = enc.move_to_index(mv)
            if 0 <= idx < len(pol):
                probs[i] = pol[idx]
        sel = valid[int(np.argmax(probs))] if probs.sum() > 0 else valid[rng.integers(len(valid))]
        s.make_move(sel); m.advance_root(sel); mc += 1
    return mc


pr = cProfile.Profile(); rng = np.random.default_rng(0)
pr.enable()
total = sum(play(rng) for _ in range(2))
pr.disable()
pr.dump_stats("/tmp/selfplay_profile.prof")
print(f"profiled {total} plies across 4 games\n")

st = pstats.Stats(pr)
buf = io.StringIO(); st.stream = buf
print("===== TOP 25 BY tottime (leaf CPU hotspots) =====")
st.sort_stats("tottime").print_stats(25)
print(buf.getvalue())

for tag in ("board.py", "moves.py", "encoding", "self_play.py", "wrapper.py"):
    b2 = io.StringIO(); st.stream = b2
    st.sort_stats("cumulative").print_stats(tag, 3)
    print(f"----- cumulative share: {tag} -----")
    print(b2.getvalue())
