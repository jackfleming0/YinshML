#!/usr/bin/env python
"""E26 teacher-data generator — high-budget-search distillation corpus.

The E25 verdict reaimed E26 at the POLICY: search manufactures a better policy than
the raw net (MCTS visit counts >> raw prior; KL grows with sims), and the ablation
proved policy is the binding head. So we run iter1 self-play at HIGH sims and save,
per main-game position, the SEARCH-IMPROVED targets that exceed the student:
  - search policy : the MCTS visit-count distribution (stored top-K sparse)
  - search value  : the MCTS root value (`last_root_value`, side-to-move POV)
  - state         : 15ch encoded (iter1's encoding)

Distillation (`e26_distill.py`) then banks these targets into the net. The point is
the target EXCEEDS the student — which does NOT require the value head to improve
(§ the value ceiling is intrinsic); the *policy* gain is the lever.

Storage: policy stored as top-K (idx,prob) — MCTS distributions are peaked, so
K=64 captures ~all visit mass without a 7433-dim dense vector per position
(1M positions dense = 30GB; top-64 = ~0.5GB).

Incremental checkpoint + per-game log + early-stop at --max-positions (lessons from
the local thrash saga). Parallel via --workers (each worker its own net).

Usage (box, high-budget):
  python scripts/gen_distill_corpus.py \
      --model models/iter1_ema_2026-05-27/iter1_ema.pt \
      --out expert_games/e26_teacher_800sim.npz \
      --games 2000 --sims 800 --workers 64 --device cuda \
      --max-positions 2000000 --checkpoint-every 50

Local smoke (tiny):
  python scripts/gen_distill_corpus.py --model models/iter1_ema_2026-05-27/iter1_ema.pt \
      --out /tmp/e26_smoke.npz --games 2 --sims 16 --max-positions 200 --device cpu
"""
import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.types import GamePhase

TOPK = 64


def make_mcts(net, sims, dirichlet_alpha, evaluator=None):
    """High-budget teacher MCTS. dirichlet noise for self-play diversity.

    ``evaluator`` (a ProcessEvaluatorClient) routes batched leaf eval to the
    shared inference server; None = the net runs predict_batch itself.
    """
    from yinsh_ml.training.self_play import MCTS
    return MCTS(
        network=net, evaluation_mode='pure_neural', heuristic_evaluator=None,
        heuristic_weight=0.0, num_simulations=sims, late_simulations=sims,
        simulation_switch_ply=20, c_puct=1.0, dirichlet_alpha=dirichlet_alpha,
        value_weight=1.0, max_depth=300, epsilon_mix_start=0.25 if dirichlet_alpha > 0 else 0.0,
        epsilon_mix_end=0.0, epsilon_mix_taper_moves=20, initial_temp=1.0,
        final_temp=0.1, annealing_steps=20, temp_clamp_fraction=0.6,
        enable_subtree_reuse=True, fpu_reduction=0.25, evaluator=evaluator,
    )


def _topk(policy, k=TOPK):
    """Top-k (idx, prob) of a dense policy vector, padded with idx=-1, prob=0."""
    nz = np.nonzero(policy)[0]
    if len(nz) > k:
        nz = nz[np.argpartition(policy[nz], -k)[-k:]]
    idx = np.full(k, -1, dtype=np.int32)
    prob = np.zeros(k, dtype=np.float32)
    idx[:len(nz)] = nz
    prob[:len(nz)] = policy[nz]
    return idx, prob


def select_move(policy, valid, encoder, temp, rng):
    probs = np.zeros(len(valid), dtype=np.float64)
    for i, mv in enumerate(valid):
        j = encoder.move_to_index(mv)
        if 0 <= j < len(policy):
            probs[i] = float(policy[j])
    if probs.sum() <= 0:
        return valid[rng.integers(0, len(valid))]
    if temp <= 1e-3:
        return valid[int(np.argmax(probs))]
    probs = probs ** (1.0 / temp)
    probs /= probs.sum()
    return valid[rng.choice(len(valid), p=probs)]


def play_and_capture(net, sims, dirichlet_alpha, seed, max_moves=250,
                     batch_size=32, evaluator=None):
    """One high-sim self-play game; capture (state15, topk_idx, topk_prob, value)
    for every MAIN_GAME position. Targets are the SEARCH outputs at this position.

    ``batch_size`` is the in-search leaf-batch flush size (raise it for the
    inference-server path so the server coalesces fat batches); ``evaluator``
    routes leaf eval to the shared GPU server."""
    encoder = net.state_encoder
    mcts = make_mcts(net, sims, dirichlet_alpha, evaluator=evaluator)
    rng = np.random.default_rng(seed)
    state = GameState()
    states, pidx, pprob, vals = [], [], [], []
    mc = 0
    while not state.is_terminal() and mc < max_moves:
        valid = state.get_valid_moves()
        if not valid:
            break
        policy = mcts.search_batch(state, mc, batch_size=batch_size)   # search-improved policy
        if state.phase == GamePhase.MAIN_GAME:
            ti, tp = _topk(np.asarray(policy, np.float32))
            states.append(np.asarray(encoder.encode_state(state), np.float32))
            pidx.append(ti); pprob.append(tp)
            vals.append(np.float32(getattr(mcts, "last_root_value", 0.0)))  # search value, STM POV
        temp = mcts.get_temperature(mc)
        sel = select_move(policy, valid, encoder, temp, rng)
        if not state.make_move(sel):
            break
        mcts.advance_root(sel)
        mc += 1
    if not states:
        z = (np.empty((0, 15, 11, 11), np.float32), np.empty((0, TOPK), np.int32),
             np.empty((0, TOPK), np.float32), np.empty((0,), np.float32))
        return z
    return (np.stack(states), np.stack(pidx), np.stack(pprob), np.asarray(vals, np.float32))


_NET = None


def _init_worker(model_path, device):
    global _NET
    import torch
    # CRITICAL for many CPU workers: cap each process to 1 thread. torch defaults
    # to all-cores intra-op parallelism, so N workers x ~ncores threads oversubscribe
    # catastrophically (e.g. 48 workers on 192 cores -> ~1850 load, box wedged).
    torch.set_num_threads(1)
    from yinsh_ml.network.wrapper import NetworkWrapper
    _NET = NetworkWrapper(model_path=model_path, device=device)
    _NET.network.eval()


def _worker(task):
    sims, da, seed, mm, bs = task
    return play_and_capture(_NET, sims, da, seed, mm, batch_size=bs)


# --- inference-server worker: CPU-only, leaf eval via the shared GPU server ---
def _gen_inference_worker(worker_id, request_queue, response_queue, result_queue,
                          stop_event, payload):
    """Persistent E26 teacher-gen worker for the inference-server path.

    CPU-only (no CUDA context — only the server has one). Builds a CPU
    NetworkWrapper for its encoder + MCTS handle (never runs a forward; the
    ProcessEvaluatorClient routes every batched leaf eval to the shared bf16
    server), plays its assigned game seeds at high sims, and streams each
    game's (states, topk_idx, topk_prob, values) onto ``result_queue``.
    Honors ``stop_event`` between games for parent-driven early-stop.
    """
    import torch
    torch.set_num_threads(1)
    from yinsh_ml.network.wrapper import NetworkWrapper
    from yinsh_ml.network.inference_server import ProcessEvaluatorClient

    net = NetworkWrapper(model_path=payload["model_path"], device="cpu")
    net.network.eval()
    client = ProcessEvaluatorClient(worker_id, request_queue, response_queue, net.state_encoder)

    sims = payload["sims"]
    da = payload["dirichlet_alpha"]
    mm = payload["max_moves"]
    bs = payload["batch_size"]
    try:
        for seed in payload["seeds"]:
            if stop_event.is_set():
                break
            try:
                res = play_and_capture(net, sims, da, seed, mm, batch_size=bs, evaluator=client)
                result_queue.put(res)
            except Exception:
                import traceback
                traceback.print_exc()
                # Empty result so the parent's per-game counter still advances.
                result_queue.put((np.empty((0, 15, 11, 11), np.float32),
                                  np.empty((0, TOPK), np.int32),
                                  np.empty((0, TOPK), np.float32),
                                  np.empty((0,), np.float32)))
    except Exception:
        import traceback
        traceback.print_exc()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--sims", type=int, default=800, help="teacher search budget (high = better targets)")
    ap.add_argument("--dirichlet-alpha", type=float, default=0.3)
    ap.add_argument("--max-moves", type=int, default=250)
    ap.add_argument("--max-positions", type=int, default=2_000_000)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    # --- E20 inference-server path (recommended for many workers on a GPU box) ---
    ap.add_argument("--use-inference-server", action="store_true",
                    help="route all leaf eval to ONE bf16 GPU server (no per-worker CUDA "
                         "context); workers stay CPU. Far faster than --device cuda with "
                         "many workers (which serializes on the GPU command queue).")
    ap.add_argument("--inference-dtype", default="bf16", choices=["fp32", "bf16", "fp16"],
                    help="server forward precision (inference-server path only)")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="in-search leaf-batch flush size; raise it on the server path so "
                         "the server coalesces fat batches (the virtual-loss fix fills them)")
    args = ap.parse_args(argv)

    device = None if args.device == "auto" else args.device
    tasks = [(args.sims, args.dirichlet_alpha, args.seed + i, args.max_moves, args.batch_size)
             for i in range(args.games)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"E26 teacher gen: {args.games} games @ {args.sims} sims, workers={args.workers}, "
          f"device={device or 'auto'}, target={args.max_positions} positions", flush=True)

    S, PI, PP, V = [], [], [], []
    total, t0 = 0, time.time()

    def save(tag):
        if not S:
            return 0
        states = np.concatenate(S); pidx = np.concatenate(PI)
        pprob = np.concatenate(PP); vals = np.concatenate(V)
        if len(states) > args.max_positions:
            r = np.random.default_rng(args.seed).choice(len(states), args.max_positions, replace=False)
            states, pidx, pprob, vals = states[r], pidx[r], pprob[r], vals[r]
        np.savez_compressed(args.out, states=states, policy_idx=pidx,
                            policy_prob=pprob, values=vals)
        print(f"  [checkpoint {tag}] {len(states)} positions -> {args.out}", flush=True)
        return len(states)

    # --- E20 inference-server path: 1 bf16 GPU server + N CPU workers --------
    if args.use_inference_server:
        import queue as _queue
        from yinsh_ml.network.inference_server import InferenceServerPool
        from yinsh_ml.network.wrapper import NetworkWrapper
        ctx = mp.get_context("spawn")

        # Detect the checkpoint's encoding so the server and the workers' encoders agree.
        _probe = NetworkWrapper(model_path=args.model, device="cpu")
        use_enhanced = bool(getattr(_probe, "use_enhanced_encoding", True))
        value_head_type = getattr(_probe, "value_head_type", None)
        del _probe

        nW = max(1, args.workers)
        assignments = [[] for _ in range(nW)]
        for i in range(args.games):
            assignments[i % nW].append(args.seed + i)
        payloads = [dict(model_path=args.model, sims=args.sims,
                         dirichlet_alpha=args.dirichlet_alpha, max_moves=args.max_moves,
                         batch_size=args.batch_size, seeds=assignments[w]) for w in range(nW)]
        server_cfg = dict(use_enhanced_encoding=use_enhanced, value_head_type=value_head_type,
                          max_batch=min(1024, max(args.batch_size * nW, 64)),
                          max_wait_ms=1.0, inference_dtype=args.inference_dtype)
        print(f"  inference-server path: 1 {args.inference_dtype} GPU server + {nW} CPU workers, "
              f"batch_size={args.batch_size}", flush=True)

        with InferenceServerPool(args.model, server_cfg, nW, _gen_inference_worker,
                                 payloads, mp_context=ctx) as pool:
            received = 0
            while received < args.games:
                try:
                    st, ti, tp, vl = pool.result_queue.get(timeout=5.0)
                except _queue.Empty:
                    if not pool.server.is_alive() or not pool.workers_alive():
                        print("  inference workers/server exited early; stopping", flush=True)
                        break
                    continue
                received += 1
                if len(st):
                    S.append(st); PI.append(ti); PP.append(tp); V.append(vl); total += len(st)
                print(f"  game {received}/{args.games}: +{len(st)} (total {total}/{args.max_positions}) "
                      f"| {(time.time()-t0)/60:.1f}m", flush=True)
                if received % args.checkpoint_every == 0:
                    save(f"g{received}")
                if total >= args.max_positions:
                    print(f"  reached target at game {received}; stopping", flush=True)
                    pool.stop_event.set()
                    break
        n = save("final")
        print(f"done: {n} positions, {(time.time()-t0)/60:.1f}m total", flush=True)
        return 0

    if args.workers > 1:
        pool = mp.Pool(args.workers, initializer=_init_worker, initargs=(args.model, device))
        it = pool.imap_unordered(_worker, tasks)
    else:
        _init_worker(args.model, device)
        it = (_worker(t) for t in tasks)
        pool = None
    try:
        for gi, (st, ti, tp, vl) in enumerate(it, 1):
            if len(st):
                S.append(st); PI.append(ti); PP.append(tp); V.append(vl); total += len(st)
            print(f"  game {gi}/{args.games}: +{len(st)} (total {total}/{args.max_positions}) "
                  f"| {(time.time()-t0)/60:.1f}m", flush=True)
            if gi % args.checkpoint_every == 0:
                save(f"g{gi}")
            if total >= args.max_positions:
                print(f"  reached target at game {gi}; stopping", flush=True)
                break
    finally:
        if pool is not None:
            pool.terminate(); pool.join()
    n = save("final")
    print(f"done: {n} positions, {(time.time()-t0)/60:.1f}m total", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
