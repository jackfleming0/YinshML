"""Process-based shared inference server for parallel MCTS self-play.

Why this exists
---------------
The 2026-05-05 4090 sweep (`GPU_SCALING_RESULTS.md`) found that N independent
worker *processes*, each calling `predict_batch` on its own NetworkWrapper,
serialize on the GPU command queue instead of coalescing — `sm_avg` sits at
10-15% while `sm_p95` hits 73%, and throughput *regresses* past 4 workers.

The first fix attempt (`BatchedEvaluator`, the threaded path in `self_play.py`)
coalesced inference into one model but ran the MCTS tree work in *threads*. It
was slower, not faster: the C++ engine's pybind methods (`get_valid_moves`,
`apply_move`, clone) do **not** release the GIL — only the `Bench*` functions in
`game_cpp/src/bindings.cpp` do — so the threads serialized on the GIL anyway.

This module is the process-based alternative the design doc named as the
fallback ("processes that share a model via torch.multiprocessing"):

- **Workers stay processes.** No GIL contention between them at all.
- **One model on the GPU.** A single dedicated server process owns the only
  CUDA context, so there's no command-queue contention to begin with.
- **Coalesce ragged per-worker batches.** Each worker flushes its per-game
  leaf batch (size `mcts_batch_size`) to a shared request queue; the server
  drains across *all* workers up to `max_batch` and runs one forward pass.
- **Encode worker-side, ship arrays.** Workers encode states to compact
  fixed-size float arrays (6 or 15 × 11 × 11) before sending — so the GPU
  forward sees pre-stacked tensors and `CppGameState` (which doesn't pickle)
  never crosses the process boundary. The server calls
  `NetworkWrapper.predict_batch_encoded`.

Transport is plain `multiprocessing` queues carrying numpy arrays. State
arrays are small (~2.9KB basic / ~7.2KB enhanced) and the message rate is
per-flush (not per-leaf), so pickle overhead is modest; if a profile later
shows IPC on the critical path, swap the transport for shared memory without
touching the call sites.

`ProcessEvaluatorClient.evaluate_batch` has the same `(logits, values)`
signature as `BatchedEvaluator.evaluate_batch` and `NetworkWrapper.predict_batch`,
so the call site in `MCTS._evaluate_and_backup_batch` is unchanged — the worker
just constructs the client and passes it as `evaluator=`.
"""

from __future__ import annotations

import logging
import queue
import time
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger("InferenceServer")


# Sentinel placed on a worker's response queue when the server fails to run a
# batch, so the client raises instead of blocking forever on a reply that will
# never come.
_ERROR = "ERROR"


def run_inference_server(
    model_path: str,
    server_cfg: dict,
    request_queue,
    response_queues: List,
    stop_event,
    ready_event,
) -> None:
    """Server-process entrypoint: own one model on the GPU, coalesce requests.

    Runs until ``stop_event`` is set. Drains ``request_queue`` (tuples of
    ``(worker_id, seq, encoded_array)``), greedily coalesces across workers up
    to ``server_cfg['max_batch']`` states or ``max_wait_ms``, runs a single
    forward pass, and scatters each request's slice back onto
    ``response_queues[worker_id]`` as ``(seq, logits_slice, values_slice)``.

    Args:
        model_path: Path to the model state_dict the parent wrote for workers.
        server_cfg: dict with ``use_enhanced_encoding``, ``value_head_type``,
            ``max_batch``, ``max_wait_ms`` (and optional ``idle_poll_ms``).
        request_queue: shared mp.Queue workers push encoded batches onto.
        response_queues: list indexed by worker_id; each worker's reply channel.
        stop_event: mp.Event — set by the parent once all games are collected.
        ready_event: mp.Event — set by this function once the model is loaded
            and on the GPU, so the parent can gate worker launch on readiness.
    """
    # Import here, not at module top: this runs in a freshly spawned process,
    # and keeping the heavy import inside the entrypoint avoids paying it in
    # every process that merely imports this module (e.g. the workers).
    from .wrapper import NetworkWrapper

    _configure_process_logger()

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network = NetworkWrapper(
            device=device,
            use_enhanced_encoding=server_cfg.get("use_enhanced_encoding", False),
            value_head_type=server_cfg.get("value_head_type", None),
        )
        network.load_model(model_path)
        network.network.eval()
        logger.info(
            "Inference server up on %s (enhanced=%s, max_batch=%d, max_wait=%.1fms)",
            device,
            server_cfg.get("use_enhanced_encoding", False),
            int(server_cfg["max_batch"]),
            float(server_cfg["max_wait_ms"]),
        )
    except Exception:
        logger.exception("Inference server failed to start; signalling ready so the parent unblocks and tears down")
        # Set ready so the parent's wait() returns; it will then find the
        # server dead / workers erroring and abort cleanly rather than hang.
        ready_event.set()
        return

    max_batch = int(server_cfg["max_batch"])
    max_wait_s = float(server_cfg["max_wait_ms"]) / 1000.0
    idle_poll_s = float(server_cfg.get("idle_poll_ms", 50.0)) / 1000.0

    ready_event.set()

    total_requests = 0
    total_states = 0
    total_forward_calls = 0

    while not stop_event.is_set():
        batch = _collect_batch(request_queue, max_batch, max_wait_s, idle_poll_s, stop_event)
        if not batch:
            continue

        arrays = [arr for (_wid, _seq, arr) in batch]
        big = np.concatenate(arrays, axis=0)
        total_requests += len(batch)
        total_states += big.shape[0]
        total_forward_calls += 1

        try:
            logits_t, values_t = network.predict_batch_encoded(big)
            logits_np = logits_t.detach().cpu().numpy()
            values_np = values_t.detach().cpu().numpy()
        except Exception:
            logger.exception("predict_batch_encoded failed on a batch of %d; erroring %d requests",
                             big.shape[0], len(batch))
            for (wid, seq, _arr) in batch:
                response_queues[wid].put((seq, _ERROR, _ERROR))
            continue

        offset = 0
        for (wid, seq, arr) in batch:
            n = arr.shape[0]
            response_queues[wid].put(
                (seq, logits_np[offset:offset + n], values_np[offset:offset + n])
            )
            offset += n

    avg_batch = (total_states / total_forward_calls) if total_forward_calls else 0.0
    logger.info(
        "Inference server shutting down — %d forward calls, %d states, "
        "%d requests, mean coalesced batch=%.1f",
        total_forward_calls, total_states, total_requests, avg_batch,
    )


def _collect_batch(request_queue, max_batch, max_wait_s, idle_poll_s, stop_event):
    """Block on the first request, then greedily coalesce up to ``max_batch``.

    Mirrors `BatchedEvaluator._collect_batch` but counts *states* (each request
    carries a whole per-worker flush of n states), and bounds the initial wait
    so the loop periodically re-checks ``stop_event``. Returns [] on idle so the
    caller can re-check the stop flag and exit cleanly.
    """
    try:
        first = request_queue.get(timeout=idle_poll_s)
    except queue.Empty:
        return []

    batch = [first]
    n_states = first[2].shape[0]
    deadline = time.monotonic() + max_wait_s
    while n_states < max_batch:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            req = request_queue.get(timeout=remaining)
        except queue.Empty:
            break
        batch.append(req)
        n_states += req[2].shape[0]
    return batch


class ProcessEvaluatorClient:
    """Worker-side proxy with the `evaluate_batch` shape MCTS expects.

    Lives in a self-play worker process. `evaluate_batch(states)` encodes the
    states to a stacked array, ships it to the server over ``request_queue``,
    and blocks on this worker's ``response_queue`` for the matching reply. Each
    worker has at most one in-flight request (the call is synchronous), so the
    ``seq`` check is a cheap safety assertion, not real demux.
    """

    def __init__(self, worker_id: int, request_queue, response_queue, encoder):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.encoder = encoder
        self._seq = 0
        # Lightweight diagnostics (E20 bottleneck attribution). `ipc_wait_s` is
        # time blocked on the server reply (IPC + server batching + GPU forward);
        # `encode_s` is CPU encode time; `calls`/`states` for averages. The
        # worker logs these at shutdown so we can tell IPC/server-bound (high
        # ipc_wait fraction of wall) from worker-CPU-bound (low) without a
        # profiler — see docs/experiments/e20_throughput_build.md Open Levers.
        self.ipc_wait_s = 0.0
        self.encode_s = 0.0
        self.calls = 0
        self.states = 0

    def evaluate(self, state):
        logits, values = self.evaluate_batch([state])
        return logits[0], values[0]

    def evaluate_batch(self, states: List):
        if not states:
            raise ValueError("evaluate_batch called with empty states list")

        _t0 = time.monotonic()
        arr = np.stack(
            [self.encoder.encode_state(s).astype(np.float32) for s in states]
        )
        self._seq += 1
        seq = self._seq
        self.request_queue.put((self.worker_id, seq, arr))
        _t1 = time.monotonic()

        rseq, logits_np, values_np = self.response_queue.get()
        self.ipc_wait_s += time.monotonic() - _t1
        self.encode_s += _t1 - _t0
        self.calls += 1
        self.states += len(states)
        # Single in-flight request per worker; a mismatch means a stale reply
        # leaked through (shouldn't happen) — drain until we see ours.
        while rseq != seq:
            rseq, logits_np, values_np = self.response_queue.get()

        if isinstance(logits_np, str) and logits_np == _ERROR:
            raise RuntimeError(
                f"Inference server failed to evaluate worker {self.worker_id} "
                f"request {seq} (see server log)."
            )

        return torch.from_numpy(logits_np), torch.from_numpy(values_np)


def _configure_process_logger() -> None:
    """Give the spawned process a console handler (spawn doesn't inherit them)."""
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(logging.INFO)
