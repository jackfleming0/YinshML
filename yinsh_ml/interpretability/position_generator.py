"""Lightweight in-process position generator for SAE probe (Track B §8).

The §8 SAE training needs (encoded_state, eventual_outcome) pairs from a
trained checkpoint. The existing `SelfPlayRunner` is game-level and tied to
the trainer's data path; for the probe we just need positions, not policy
targets, and we want them on disk fast without spinning up the full
training infrastructure.

This module does the minimum needed:
  * Load a checkpoint as a NetworkWrapper.
  * Play self-play games using greedy/sampled network policy (no MCTS — we
    want raw network behavior, plus MCTS at long sims would dominate
    wall-clock).
  * Yield each position alongside the eventual game outcome from that
    player's perspective.

The "no MCTS" choice is deliberate: the SAE probes what the *network* has
learned. MCTS-augmented decisions would change the *distribution* of
positions visited but not the network's representation of any one position.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..game.game_state import GameState
from ..game.types import Player
from ..network.wrapper import NetworkWrapper
from ..utils.encoding import StateEncoder

logger = logging.getLogger(__name__)


@dataclass
class PositionRecord:
    """One position from a self-play game."""
    encoded_state: np.ndarray   # shape (C, 11, 11)
    move_number: int
    player_to_move: int         # +1 = white, -1 = black (Player enum value)
    network_value: float        # network's predicted value at this position
    # Filled in after the game ends:
    eventual_outcome: float = 0.0   # +1 = current-player won, -1 = lost, 0 = draw


def play_single_game(
    network: NetworkWrapper,
    encoder: StateEncoder,
    max_moves: int = 300,
    temperature: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> List[PositionRecord]:
    """Play one self-play game, return per-position records with outcomes
    backfilled from the final game state."""
    if rng is None:
        rng = np.random.default_rng()

    state = GameState()
    records: List[PositionRecord] = []
    move_count = 0

    while not state.is_terminal() and move_count < max_moves:
        valid_moves = state.get_valid_moves()
        if not valid_moves:
            break

        encoded = encoder.encode_state(state).astype(np.float32)
        # Network forward (no grad, no MCTS).
        with torch.no_grad():
            policy_logits, value_t = network.predict_from_state(state)
        net_value = float(value_t.item())
        # Mask invalid moves before sampling.
        valid_indices = [encoder.move_to_index(m) for m in valid_moves]
        policy_probs = F.softmax(policy_logits.squeeze(0), dim=-1).cpu().numpy()
        masked = np.zeros_like(policy_probs)
        masked[valid_indices] = policy_probs[valid_indices]
        if masked.sum() <= 0:
            # Network put zero mass on every legal move — fall back to uniform.
            masked[valid_indices] = 1.0
        # Apply temperature, then sample.
        if temperature > 0:
            masked = np.power(masked, 1.0 / temperature)
        masked /= masked.sum()
        chosen_idx = int(rng.choice(len(masked), p=masked))
        # Map idx back to a Move via the valid_moves search (encoder lacks
        # an inverse for the masked-vector index in general).
        # The valid_moves list is the source of truth — find which entry's
        # move_to_index matches chosen_idx.
        chosen_move = None
        for m in valid_moves:
            if encoder.move_to_index(m) == chosen_idx:
                chosen_move = m
                break
        if chosen_move is None:
            # Safety net: if the chosen index isn't decodable (shouldn't
            # happen with masking above), fall back to argmax over valid moves.
            best_local = max(range(len(valid_moves)), key=lambda i: masked[valid_indices[i]])
            chosen_move = valid_moves[best_local]

        records.append(PositionRecord(
            encoded_state=encoded,
            move_number=move_count,
            player_to_move=int(state.current_player.value),
            network_value=net_value,
        ))
        state.make_move(chosen_move)
        move_count += 1

    # Backfill eventual outcomes from the perspective of the player who
    # was to move at each position.
    score_diff = state.white_score - state.black_score
    outcome_white = float(np.clip(score_diff / 3.0, -1.0, 1.0))
    for rec in records:
        if rec.player_to_move == Player.WHITE.value:
            rec.eventual_outcome = outcome_white
        else:
            rec.eventual_outcome = -outcome_white

    return records


def generate_positions(
    checkpoint_path: str | Path,
    num_positions: int,
    device: str = 'cpu',
    use_enhanced_encoding: bool = False,
    temperature: float = 0.5,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Drive `play_single_game` until `num_positions` records have been
    collected. Returns (encoded_states, network_values, eventual_outcomes,
    move_numbers) as parallel arrays.

    Args:
        checkpoint_path: Path to a NetworkWrapper-loadable checkpoint.
        num_positions: Stop once this many positions have been collected.
        device: Torch device for the network.
        use_enhanced_encoding: Match the encoding used to train the checkpoint.
        temperature: Move-sampling temperature.
        seed: RNG seed for reproducibility.

    Returns:
        encoded_states: (N, C, 11, 11) float32
        network_values: (N,) float32  — network's predicted value at each pos
        eventual_outcomes: (N,) float32  — actual outcome from player's POV
        move_numbers: (N,) int32
    """
    network = NetworkWrapper(device=torch.device(device), use_enhanced_encoding=use_enhanced_encoding)
    network.load_model(str(checkpoint_path))
    network.network.eval()
    encoder = network.state_encoder

    rng = np.random.default_rng(seed)
    encoded_list = []
    netv_list = []
    outcome_list = []
    move_list = []
    games_played = 0
    while sum(map(len, encoded_list)) < num_positions:
        records = play_single_game(network, encoder, temperature=temperature, rng=rng)
        if not records:
            continue
        encoded_list.append(np.stack([r.encoded_state for r in records]))
        netv_list.append(np.array([r.network_value for r in records], dtype=np.float32))
        outcome_list.append(np.array([r.eventual_outcome for r in records], dtype=np.float32))
        move_list.append(np.array([r.move_number for r in records], dtype=np.int32))
        games_played += 1
        if games_played % 10 == 0:
            collected = sum(map(len, encoded_list))
            logger.info(f"[positions] {games_played} games, {collected}/{num_positions} positions")

    encoded = np.concatenate(encoded_list, axis=0)[:num_positions]
    netv = np.concatenate(netv_list, axis=0)[:num_positions]
    outcomes = np.concatenate(outcome_list, axis=0)[:num_positions]
    moves = np.concatenate(move_list, axis=0)[:num_positions]
    return encoded, netv, outcomes, moves
