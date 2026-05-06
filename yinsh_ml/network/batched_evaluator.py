"""Shared batched-inference evaluator for parallel MCTS workers.

See `BATCHED_EVALUATOR_DESIGN.md` for the why; this module is the what.

The 2026-05-05 4090 sweep showed `sm_p95=73%, sm_avg=15%` at 4 workers
and *worse* throughput at 8/16 workers — N independent worker processes
each calling `predict_batch` on their own `NetworkWrapper` create
GPU-side command-queue contention rather than coalescing. This evaluator
replaces that pattern with one model on the GPU and one inference loop
that drains a shared queue, batching across all workers.

Structural choices:

- **Threads, not processes.** Bitboard work moved most of MCTS's hot
  path out of pure Python (so the GIL is released during the heavy
  compute), making threads viable. No IPC, no model duplication, no
  spawn overhead. If profiling later shows GIL contention is the wall,
  fall back to processes that share a model via `torch.multiprocessing`.
- **One model on the GPU.** The current per-worker architecture has *N*.
  At 4090 24GB and a small ResNet (~150MB), one copy is plenty.
- **Per-request `Future`s, not a shared reply channel.** Each caller
  blocks on its own future; routing back is O(1) by reference instead
  of O(N) scan.

Public API mirrors `NetworkWrapper.predict_batch` so the call-site swap
in `self_play.MCTS._evaluate_and_backup_batch` is one line.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Future
from typing import List, Tuple

import torch

logger = logging.getLogger(__name__)


class _Request:
    """Single inference request: input state + reply slot."""

    __slots__ = ("state", "reply")

    def __init__(self, state, reply: Future):
        self.state = state
        self.reply = reply


class BatchedEvaluator:
    """One inference loop, many MCTS workers.

    Workers call `evaluate(state)` or `evaluate_batch(states)`; an internal
    drain thread pulls requests off a shared queue, calls
    `network.predict_batch(...)` once per drained batch, and resolves the
    per-request `Future`s with their slice of the result.

    Use as a context manager (`with BatchedEvaluator(network) as ev: ...`)
    so `shutdown()` always runs — pending futures get their exception set
    rather than blocking workers forever.

    Args:
        network: NetworkWrapper-shaped object exposing `predict_batch`.
        max_batch: Hard upper bound on a single `predict_batch` call.
            Once the drain thread has collected this many requests it
            flushes immediately. Default 128.
        max_wait_ms: After receiving the first request of a batch, wait
            up to this long for more before flushing. Smaller values =
            lower per-request latency but smaller batches; larger values
            = higher GPU utilization at the cost of latency. Default 1ms.
        idle_poll_ms: When the queue is empty, how long the drain thread
            blocks waiting for the first request. Bounds shutdown latency
            from above. Tests may want this lower; production fine at 50ms.
    """

    def __init__(
        self,
        network,
        *,
        max_batch: int = 128,
        max_wait_ms: float = 1.0,
        idle_poll_ms: float = 50.0,
    ):
        self.network = network
        self.max_batch = max_batch
        self._max_wait_s = max_wait_ms / 1000.0
        self._idle_poll_s = idle_poll_ms / 1000.0

        self._req_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._drain_thread = threading.Thread(
            target=self._drain_loop,
            name="BatchedEvaluator-drain",
            daemon=True,
        )
        # Cheap counters for diagnostic output. Not lock-protected; reads
        # are racy but only used for logging, not control flow.
        self.total_requests = 0
        self.total_predict_batch_calls = 0

        self._drain_thread.start()

    # ----- Public API ---------------------------------------------------

    def evaluate(self, state) -> Tuple[torch.Tensor, torch.Tensor]:
        """Submit one state, block on its result.

        Returns (logits_row, value_scalar) shaped like a single row of
        `predict_batch`'s outputs: logits is shape (total_moves,), value
        is shape (1,).
        """
        logits, values = self.evaluate_batch([state])
        return logits[0], values[0]

    def evaluate_batch(
        self, states: List
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Submit N states, block on their combined result.

        Returns (logits_batch, values_batch) shaped exactly like
        `NetworkWrapper.predict_batch`: logits is (N, total_moves), values
        is (N, 1). Inputs are submitted as N independent requests so
        they can coalesce with requests from other workers — the caller
        does not get a "batch lock" on the GPU.
        """
        if not states:
            raise ValueError("evaluate_batch called with empty states list")

        futures: List[Future] = []
        for state in states:
            reply: Future = Future()
            self._req_queue.put(_Request(state, reply))
            futures.append(reply)

        # Block until each future resolves. set_exception propagates here.
        rows = [f.result() for f in futures]
        logits = torch.stack([row[0] for row in rows])
        values = torch.stack([row[1] for row in rows])
        return logits, values

    def shutdown(self) -> None:
        """Stop the drain thread, fail any still-pending futures.

        Safe to call multiple times. Drains for up to `idle_poll_ms +
        max_wait_ms` to let the in-flight batch (if any) complete.
        """
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        self._drain_thread.join(timeout=5.0)
        # Anything left in the queue is orphaned — fail loudly so
        # callers don't deadlock waiting on a result that will never come.
        drained = 0
        while True:
            try:
                req = self._req_queue.get_nowait()
            except queue.Empty:
                break
            req.reply.set_exception(
                RuntimeError("BatchedEvaluator shut down before request was processed")
            )
            drained += 1
        if drained:
            logger.warning("BatchedEvaluator shutdown: %d orphaned requests failed", drained)

    def __enter__(self) -> "BatchedEvaluator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    # ----- Internal -----------------------------------------------------

    def _drain_loop(self) -> None:
        """Pull requests, batch them, run inference, resolve futures."""
        while not self._stop_event.is_set():
            batch = self._collect_batch()
            if not batch:
                continue
            self.total_requests += len(batch)
            self.total_predict_batch_calls += 1
            states = [r.state for r in batch]
            try:
                logits_batch, values_batch = self.network.predict_batch(states)
            except Exception as exc:  # pragma: no cover — defensive
                logger.exception("predict_batch failed; failing %d futures", len(batch))
                for r in batch:
                    r.reply.set_exception(exc)
                continue
            # Distribute: row i goes to request i. Each future gets a
            # *view* into the batch tensors; callers can stack/index as
            # they wish. We use `.detach()` to be safe — the network is
            # in eval/no_grad mode but autograd state can leak via
            # tensor pool reuse.
            for i, r in enumerate(batch):
                logits_row = logits_batch[i].detach()
                value_row = values_batch[i].detach()
                r.reply.set_result((logits_row, value_row))

    def _collect_batch(self) -> List[_Request]:
        """Block on first request, then greedily drain up to max_batch.

        Returns [] if no request arrives before stop is signalled, so the
        outer loop can re-check `_stop_event` and exit cleanly.
        """
        # Block on the first request, but bound the wait so shutdown is
        # responsive and we periodically re-check `_stop_event`.
        try:
            first = self._req_queue.get(timeout=self._idle_poll_s)
        except queue.Empty:
            return []

        batch: List[_Request] = [first]
        deadline = time.monotonic() + self._max_wait_s
        while len(batch) < self.max_batch:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                req = self._req_queue.get(timeout=remaining)
            except queue.Empty:
                break
            batch.append(req)
        return batch
