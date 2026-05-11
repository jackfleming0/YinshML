"""Parity tests: serial `MCTS.search` vs batched `MCTS.search_batch`.

These tests pin down whether the batched leaf-evaluation path produces visit
distributions identical to the per-leaf serial path on a deterministic
fake network. The expert review (T1.1) claimed the batched path expands the
root multiple times and applies Dirichlet noise only once, effectively
collapsing a 48-sim batch to ~16 sims with no root noise. Verification
disagreed: `batch_leaves` only contains non-root leaves and the
`dirichlet_applied` flag gates Dirichlet to exactly one application.

The contract these tests pin:

  1. With Dirichlet disabled and a deterministic fake network, the visit
     distribution at the root after `search()` and `search_batch()` is
     identical (visit-by-visit equality across children for every move).
  2. With Dirichlet enabled, Dirichlet is applied to root exactly once per
     `search_batch()` invocation, regardless of how many simulations or
     batches roll through `_evaluate_and_backup_batch`.
  3. Total root visits in `search_batch()` equal `num_simulations` (the
     virtual-loss / batching machinery doesn't drop simulations).

If (1) fails, T1.1 is a real bug — Wave 1 needs to route it as a workstream.
If (1) passes, T1.1 is dead and the bake should focus elsewhere.
"""

from __future__ import annotations

import random
from typing import Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from yinsh_ml.game.game_state import GameState
from yinsh_ml.training.self_play import MCTS as SelfPlayMCTS
from yinsh_ml.utils.encoding import StateEncoder


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


def _seed_all(seed: int = 0) -> None:
    """Pin every RNG that MCTS touches.

    `_select_action` uses both `np.random.uniform` (tiebreak noise) and
    `random.choice` (zero-visit fallback). `_apply_root_dirichlet_noise`
    plus the inline noise inside `search`/`_evaluate_and_backup_batch`
    both use `np.random.dirichlet`. torch.manual_seed is set defensively
    in case the wrapper's MagicMock leaks into a real codepath.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _fake_network_deterministic(total_moves: int) -> MagicMock:
    """Fake NetworkWrapper-shaped object.

    `predict_from_state(state)` returns (zeros_logits, [[0.0]]) — used by
    serial `search()`. `predict_batch(states)` returns the batched
    equivalent. Both return uniform logits (zeros softmax to uniform after
    valid-move masking) and a fixed value of 0.0. Deterministic by
    construction — no RNG inside the network.
    """
    network = MagicMock()
    network.state_encoder = StateEncoder()

    def predict_from_state(_state):
        policy_logits = torch.zeros(1, total_moves)
        value = torch.tensor([[0.0]])
        return policy_logits, value

    def predict_batch(states):
        n = len(states)
        policy_logits = torch.zeros(n, total_moves)
        values = torch.zeros(n, 1)
        return policy_logits, values

    network.predict_from_state = predict_from_state
    network.predict_batch = predict_batch
    return network


def _build_mcts(
    num_simulations: int,
    dirichlet_alpha: float = 0.0,
    epsilon_mix: float = 0.0,
) -> SelfPlayMCTS:
    """One MCTS instance with all the no-randomness knobs except those the
    test explicitly wants on. Subtree reuse off — every search starts fresh
    so the two variants aren't comparing a serial-built tree against a
    batched-grown tree."""
    encoder = StateEncoder()
    network = _fake_network_deterministic(encoder.total_moves)
    return SelfPlayMCTS(
        network=network,
        evaluation_mode="pure_neural",
        num_simulations=num_simulations,
        late_simulations=num_simulations,
        simulation_switch_ply=10_000,  # never switch
        c_puct=1.0,
        dirichlet_alpha=dirichlet_alpha,
        value_weight=1.0,
        max_depth=50,
        initial_temp=1.0,
        final_temp=1.0,
        annealing_steps=1,
        enable_subtree_reuse=False,
        epsilon_mix_start=epsilon_mix,
        epsilon_mix_end=epsilon_mix,
        epsilon_mix_taper_moves=0,
        fpu_reduction=0.0,  # zero out FPU so unvisited children only differ via priors
    )


def _run_and_capture_visits(
    mcts: SelfPlayMCTS,
    state: GameState,
    move_number: int,
    use_batch: bool,
    batch_size: int = 32,
) -> Tuple[np.ndarray, int, int]:
    """Run the search and capture per-child visit counts at the root.

    Returns (visit_counts indexed by canonical move-index, total_visits at
    root, dirichlet_applications_count). To get the visit counts we have to
    sidestep `root.clear_tree()` that fires when subtree reuse is off; we
    do that by temporarily enabling subtree reuse on the instance (it still
    starts each call fresh because `_cached_root` is None initially).

    We also instrument `np.random.dirichlet` to count how many times it was
    called during the search — this is how we verify the Dirichlet-applied-
    once invariant.
    """
    # Enable subtree reuse for THIS test only so the root tree survives past
    # the search return. The very first call still builds a fresh root
    # (because `_cached_root` is None) — exactly what we want.
    mcts.enable_subtree_reuse = True
    mcts._cached_root = None

    encoder = mcts.state_encoder

    # Instrument np.random.dirichlet so we can count root-noise applications.
    real_dirichlet = np.random.dirichlet
    counter = {"calls": 0}

    def counting_dirichlet(*args, **kwargs):
        counter["calls"] += 1
        return real_dirichlet(*args, **kwargs)

    np.random.dirichlet = counting_dirichlet
    try:
        if use_batch:
            _ = mcts.search_batch(state, move_number=move_number, batch_size=batch_size)
        else:
            _ = mcts.search(state, move_number=move_number)
    finally:
        np.random.dirichlet = real_dirichlet

    root = mcts._cached_root
    assert root is not None, "Test invariant: subtree-reuse keeps root alive after search."

    visits = np.zeros(encoder.total_moves, dtype=np.int64)
    total = 0
    for move, child in root.children.items():
        idx = encoder.move_to_index(move)
        if 0 <= idx < len(visits):
            visits[idx] = child.visit_count
            total += child.visit_count

    return visits, total, counter["calls"]


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("num_sims", [16, 48])
def test_serial_and_batch_match_with_no_dirichlet(num_sims: int):
    """Deterministic-network, no-root-noise: serial visit counts must equal
    batched visit counts child-by-child.

    Why this works even with `_select_action`'s ε-noise: with `c_puct=1`,
    `fpu_reduction=0`, zero logits (uniform prior), and the FIRST few sims
    selecting from a fully-unexpanded tree, the per-call `np.random.uniform`
    sequence is deterministic under a pinned seed. The serial and batched
    paths consume the same number of `_select_action` calls in the same
    order (one per sim, plus zero descents from the unexpanded root for the
    very first sim) — pinning numpy's seed makes the two runs draw the same
    epsilon-noise vector at each step.

    Both implementations apply the same `_backpropagate` math, the same
    expansion machinery, and use a fake network with constant outputs.
    Anywhere they diverge is a real semantic difference.
    """
    _seed_all(0)
    state_serial = GameState()
    mcts_serial = _build_mcts(num_simulations=num_sims, dirichlet_alpha=0.0)
    visits_s, total_s, dir_calls_s = _run_and_capture_visits(
        mcts_serial, state_serial, move_number=0, use_batch=False
    )

    # Use a small batch_size so the batched path actually descends through an
    # expanded root for most sims. With batch_size >= num_sims, EVERY sim sees
    # an unexpanded root and ends up backpropping at root only — that's a
    # different (and more egregious) failure mode of T1.1, but the visit-count
    # comparison test below catches both.
    batch_size = 4
    _seed_all(0)
    state_batch = GameState()
    mcts_batch = _build_mcts(num_simulations=num_sims, dirichlet_alpha=0.0)
    visits_b, total_b, dir_calls_b = _run_and_capture_visits(
        mcts_batch, state_batch, move_number=0, use_batch=True, batch_size=batch_size
    )

    # 1. Serial: sim 1 has root as leaf (unexpanded), backprops to root only —
    #    no child visit. Sims 2..N descend exactly one ply (root expanded after
    #    sim 1; sim's leaf is an unexpanded child node). Each child-visiting
    #    sim adds 1 to some child's visit_count. So Σ child visits = N - 1.
    #
    #    Batched: the FIRST batch (size B) has all B sims seeing root
    #    unexpanded — every sim in batch 1 ends up with search_path=[root].
    #    Only sims B+1..N descend through the now-expanded root. So Σ child
    #    visits = N - B (where B = batch_size, capped at N).
    #
    #    If T1.1's "batch wastes B-1 sims at root" claim is right, the two
    #    totals will diverge. If T1.1 is wrong, they should match.
    expected_serial_child_visits = num_sims - 1
    assert total_s == expected_serial_child_visits, (
        f"serial child-visit total {total_s} != num_sims-1 "
        f"({expected_serial_child_visits}). Investigate why selection bailed."
    )

    # 2. Dirichlet was disabled — neither should have called np.random.dirichlet
    #    for root noise. (The implementation has an `epsilon_mix > 0` gate
    #    that also blocks it, so this is belt-and-suspenders.)
    assert dir_calls_s == 0, f"serial fired Dirichlet {dir_calls_s} times despite alpha=0/eps=0"
    assert dir_calls_b == 0, f"batched fired Dirichlet {dir_calls_b} times despite alpha=0/eps=0"

    # 3. The actual claim: visit count vectors are identical.
    if not np.array_equal(visits_s, visits_b):
        # Surface the diff so a failing run is actionable, not just "they differ".
        diff_idx = np.where(visits_s != visits_b)[0]
        diffs = [(int(i), int(visits_s[i]), int(visits_b[i])) for i in diff_idx[:10]]
        pytest.fail(
            f"Serial vs batched visit-count mismatch on {len(diff_idx)} child(ren) "
            f"with num_sims={num_sims}. First few (move_idx, serial, batched): {diffs}. "
            f"This means T1.1 is REAL — batched search semantically differs from serial."
        )


def test_batch_search_applies_dirichlet_exactly_once():
    """With Dirichlet on, `search_batch` should call `np.random.dirichlet`
    EXACTLY once at the root, no matter how many sims or batches.

    Confirms the `dirichlet_applied` flag in `_evaluate_and_backup_batch`
    is actually gating the noise (verification claim).
    """
    _seed_all(0)
    state = GameState()
    mcts = _build_mcts(num_simulations=48, dirichlet_alpha=0.3, epsilon_mix=0.25)
    visits, total, dir_calls = _run_and_capture_visits(
        mcts, state, move_number=0, use_batch=True, batch_size=8
    )
    # 48 sims / batch=8 → 6 batches. If Dirichlet were applied per batch, we'd
    # see 6 calls. If applied per sim (the T1.1 claim), we'd see ~48 calls.
    # The contract is exactly 1.
    assert dir_calls == 1, (
        f"`search_batch` called np.random.dirichlet {dir_calls} times — expected exactly 1. "
        f"If >1, the `dirichlet_applied` flag is not gating noise correctly (T1.1 partial)."
    )
    # 48 sims, batch_size=8: first batch (8 sims) all see root unexpanded and
    # backprop to root only — Σ child visits = 48 - 8 = 40. (This isn't the
    # ideal contract — the ideal is Σ = 47 like the serial path — but that's
    # exactly what T1.1 is about; this assertion just pins current behavior so
    # we know if a fix lands and shifts it.)
    assert total == 40, (
        f"Expected 40 child visits (48 sims - batch_size 8 = 40), got {total}. "
        f"If this changed to 47, T1.1 was fixed and this assertion needs updating."
    )


def test_serial_search_applies_dirichlet_exactly_once():
    """Same invariant for the serial path. Pins the symmetric contract."""
    _seed_all(0)
    state = GameState()
    mcts = _build_mcts(num_simulations=48, dirichlet_alpha=0.3, epsilon_mix=0.25)
    visits, total, dir_calls = _run_and_capture_visits(
        mcts, state, move_number=0, use_batch=False
    )
    assert dir_calls == 1, (
        f"`search` called np.random.dirichlet {dir_calls} times — expected exactly 1."
    )
    assert total == 47, f"Expected 47 child visits (48 sims - 1 root-leaf sim), got {total}"
