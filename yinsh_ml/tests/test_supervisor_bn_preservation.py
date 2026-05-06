"""Regression test: TrainingSupervisor.clear_pytorch_memory must NOT delete
BatchNorm running statistics from the network.

Pre-fix bug: clear_pytorch_memory iterated every module's `_buffers` and
set each tensor buffer to None to "free memory." That deregistered BN's
running_mean / running_var / num_batches_tracked, so the next saved
checkpoint had no BN stats. On reload, BN fell back to mean=0 / var=1 and
the entire conv stack produced scrambled features — the cloud_run_v1
50/0/0 policy-collapse failure mode.

If this test fails, do NOT "fix" it by mutating BN buffers. Find another
way to free whatever memory you were trying to free.
"""

from __future__ import annotations

import torch

from yinsh_ml.network.wrapper import NetworkWrapper


def _bn_keys(state_dict: dict) -> set[str]:
    return {
        k for k in state_dict
        if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k
    }


def test_clear_pytorch_memory_preserves_bn_running_stats():
    """clear_pytorch_memory must keep BN running stats intact and saveable."""
    # Lazy import to keep test module importable without supervisor's heavy deps
    # being pulled in at collection time.
    from yinsh_ml.training.supervisor import TrainingSupervisor  # noqa: F401

    nw = NetworkWrapper(device="cpu")

    # Populate BN running stats with a few forward passes in train mode.
    nw.network.train()
    x = torch.randn(4, 6, 11, 11)
    for _ in range(3):
        nw.network(x)

    pre_keys = set(nw.network.state_dict().keys())
    pre_bn = _bn_keys(pre_keys)
    assert len(pre_bn) > 0, "fixture failed: forward pass should populate BN stats"

    # Reach into the supervisor's cleanup. We do not need the full supervisor —
    # we run only the buffer-clearing block that previously contained the bug.
    # Construct a tiny stand-in that has the same `network.network` attribute.
    class _Stub:
        pass

    stub = _Stub()
    stub.network = nw

    # Inline the post-fix loss-clearing logic. If a future refactor re-introduces
    # buffer mutation here, this test fails because BN keys go missing.
    if hasattr(stub.network, "network"):
        # Post-fix: this block must NOT touch model._buffers. It may clear
        # transient lists / caches but must not deregister BN running stats.
        pass  # no-op stand-in; the real method is exercised in the smoke test below

    # Now exercise the actual cleanup method. Rebuild a minimal supervisor-like
    # object with just enough for clear_pytorch_memory to run.
    class _MiniSupervisor:
        def __init__(self, network_wrapper):
            self.network = network_wrapper
            # Minimal trainer surface.
            class _Trainer:
                def __init__(self):
                    self.policy_losses = [1.0, 2.0]
                    self.value_losses = [1.0, 2.0]
                    # Optimizer surface needed by clear_pytorch_memory.
                    self.policy_optimizer = torch.optim.Adam(network_wrapper.network.parameters(), lr=1e-3)
                    self.value_optimizer = torch.optim.SGD(network_wrapper.network.parameters(), lr=1e-3)
                    self.experience = self  # self-reference satisfies hasattr check
            self.trainer = _Trainer()
            import logging
            self.logger = logging.getLogger("test.MiniSupervisor")

        # Bind the real method.
        clear_pytorch_memory = TrainingSupervisor.clear_pytorch_memory

    mini = _MiniSupervisor(nw)
    mini.clear_pytorch_memory()

    post_keys = set(nw.network.state_dict().keys())
    post_bn = _bn_keys(post_keys)
    missing_bn = pre_bn - post_bn

    assert missing_bn == set(), (
        f"clear_pytorch_memory dropped {len(missing_bn)} BN running-stat keys: "
        f"{sorted(missing_bn)[:3]}... — this is the cloud_run_v1 50/0/0 bug. "
        "DO NOT null out module._buffers in the cleanup path."
    )

    # Belt-and-suspenders: the saved+reloaded round trip must preserve BN keys
    # too (this is what actually broke checkpoint inference).
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "bn_check.pt")
        nw.save_model(path)
        loaded = torch.load(path, map_location="cpu")
        loaded_bn = _bn_keys(loaded)
        assert loaded_bn == pre_bn, (
            f"saved checkpoint missing BN keys after cleanup: "
            f"{sorted(pre_bn - loaded_bn)[:3]}..."
        )
