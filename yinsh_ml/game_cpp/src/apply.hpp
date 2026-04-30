// Move application + phase transitions for the bitboard engine.
//
// Mirrors yinsh_ml/game/game_state.py::GameState.make_move and
// _update_game_phase, encoded against the bitboard State. ApplyMove is
// trust-the-caller: validation lives in get_valid_moves / is_valid_move
// in movegen.hpp; the inner MCTS / self-play loop relies on never
// passing invalid moves here, mirroring how the Python engine is
// driven in practice.
//
// Key behaviours preserved bit-for-bit:
//   * Marker placement at the source cell when a ring moves.
//   * Marker flip (color swap) for every marker on the move path.
//   * Atomic _handle_marker_removal: validate-all then remove-all.
//   * RING_REMOVAL → ROW_COMPLETION re-entry if removing the ring
//     uncovers (does not affect markers, but the previous removal may
//     have left another row): the Python engine re-runs find_marker
//     after each ring-removal in case ROW_COMPLETION should fire again
//     on the new score state. We mirror that.
//   * Player switching only on PLACE_RING and on MOVE_RING that stays
//     in MAIN_GAME. ROW_COMPLETION + RING_REMOVAL keep the same player.

#pragma once

#include "bitboard.hpp"
#include "moves.hpp"
#include "state.hpp"
#include "tables.hpp"

namespace yinsh::cpp {

// Path between two cells along a hex axis (exclusive of source and
// destination). Caller has already validated the move is along a hex
// axis. Returns up to 9 intermediate cell indices terminated by -1.
struct Path {
    std::array<std::int8_t, 11> cells;
    int length = 0;
};

inline Path PathBetween(int source, int destination) noexcept {
    Path p;
    if (source == destination) return p;

    // Find which of the 6 directions points from source to destination,
    // and walk it. Reuses the precomputed walk table — destination is
    // guaranteed to be one of the cells in the walk for some direction.
    for (std::size_t d = 0; d < 6; ++d) {
        const auto& walk = kRayTable.walk[source][d];
        for (int i = 0; i < static_cast<int>(walk.size()); ++i) {
            const std::int8_t next = walk[i];
            if (next < 0) break;
            if (next == destination) {
                // i is the index of the destination in walk; cells
                // 0..i-1 are the intermediates.
                for (int k = 0; k < i; ++k) {
                    p.cells[k] = walk[k];
                }
                p.length = i;
                return p;
            }
        }
    }
    return p;  // not found — caller will see length=0 and handle
}

inline void HandlePlaceRing(State& s, const Move& m) noexcept {
    const Bitboard bit = kOne << m.source;
    if (m.player_is_black) {
        s.black_rings |= bit;
        ++s.black_rings_placed;
    } else {
        s.white_rings |= bit;
        ++s.white_rings_placed;
    }
}

inline void HandleMoveRing(State& s, const Move& m) noexcept {
    const Bitboard src_bit = kOne << m.source;
    const Bitboard dst_bit = kOne << m.destination;

    // Move the ring to destination, drop a same-color marker at source.
    if (m.player_is_black) {
        s.black_rings &= ~src_bit;
        s.black_rings |= dst_bit;
        s.black_markers |= src_bit;
    } else {
        s.white_rings &= ~src_bit;
        s.white_rings |= dst_bit;
        s.white_markers |= src_bit;
    }

    // Flip every marker on the path: white_markers ↔ black_markers
    // for the cells along the walk between source and destination.
    const Path path = PathBetween(m.source, m.destination);
    Bitboard flip_mask = 0;
    for (int i = 0; i < path.length; ++i) {
        flip_mask |= (kOne << path.cells[i]);
    }
    // For cells on the path that have white markers → become black, and
    // vice versa. Cells with no marker (rings can't be on the path —
    // they would have blocked the move) stay untouched.
    const Bitboard white_on_path = s.white_markers & flip_mask;
    const Bitboard black_on_path = s.black_markers & flip_mask;
    s.white_markers = (s.white_markers & ~white_on_path) | black_on_path;
    s.black_markers = (s.black_markers & ~black_on_path) | white_on_path;
}

inline bool HandleMarkerRemoval(State& s, const Move& m) noexcept {
    Bitboard remove_mask = 0;
    for (int i = 0; i < 5; ++i) {
        const std::int8_t cell = m.markers[i];
        if (cell < 0) return false;
        remove_mask |= (kOne << cell);
    }

    // Validate: every cell must hold a same-color marker.
    const Bitboard expected = m.player_is_black ? s.black_markers : s.white_markers;
    if ((remove_mask & expected) != remove_mask) {
        return false;  // not all 5 are same-color markers — malformed move
    }

    // Atomic remove: only mutates if validation passed.
    if (m.player_is_black) {
        s.black_markers &= ~remove_mask;
    } else {
        s.white_markers &= ~remove_mask;
    }
    return true;
}

inline void HandleRingRemoval(State& s, const Move& m) noexcept {
    const Bitboard bit = kOne << m.source;
    if (m.player_is_black) {
        s.black_rings &= ~bit;
        ++s.black_score;
    } else {
        s.white_rings &= ~bit;
        ++s.white_score;
    }
}

inline bool HasRowsForColor(Bitboard markers) noexcept {
    return !FindMarkerRows(markers).empty();
}

// Phase + player update logic, mirroring _update_game_phase. Called
// after the move-handling helpers have already mutated `s`.
//
// `move_type` and `move_player_is_black` describe the move that was
// just applied. We need them because the post-move player switch and
// the move_maker bookkeeping both depend on the move type.
inline void UpdatePhaseAndSwitchPlayer(State& s,
                                       std::int8_t move_type,
                                       bool move_player_is_black) noexcept {
    // Game-end takes precedence over any phase machinery. Clear the
    // move_maker so a pool-reused State doesn't carry stale bookkeeping.
    if (s.white_score >= 3 || s.black_score >= 3) {
        s.phase = kPhaseGameOver;
        s.move_maker_is_black = -1;
        return;
    }

    if (s.phase == kPhaseRingPlacement) {
        if (s.white_rings_placed == kRingsPerPlayer
                && s.black_rings_placed == kRingsPerPlayer) {
            s.phase = kPhaseMainGame;
        }
        // Always switch player after PLACE_RING.
        s.current_player_is_black = !s.current_player_is_black;
        return;
    }

    const bool white_rows = HasRowsForColor(s.white_markers);
    const bool black_rows = HasRowsForColor(s.black_markers);
    const bool any_rows = white_rows || black_rows;

    auto set_row_completion_player = [&]() {
        // If both colors have rows AND we have a move_maker, move_maker
        // gets priority. Otherwise prefer the color that has rows.
        if (s.move_maker_is_black >= 0 && white_rows && black_rows) {
            s.current_player_is_black = (s.move_maker_is_black != 0);
        } else if (white_rows) {
            s.current_player_is_black = false;
        } else if (black_rows) {
            s.current_player_is_black = true;
        }
    };

    if (s.phase == kPhaseMainGame && any_rows) {
        s.phase = kPhaseRowCompletion;
        if (s.move_maker_is_black < 0) {
            s.move_maker_is_black = move_player_is_black ? 1 : 0;
        }
        set_row_completion_player();
        return;
    }

    if (s.phase == kPhaseMainGame) {
        // Plain MAIN_GAME → MAIN_GAME after a MOVE_RING; switch player.
        s.current_player_is_black = !s.current_player_is_black;
        return;
    }

    if (s.phase == kPhaseRowCompletion) {
        s.phase = kPhaseRingRemoval;
        // Same player keeps going.
        return;
    }

    if (s.phase == kPhaseRingRemoval) {
        if (any_rows) {
            s.phase = kPhaseRowCompletion;
            set_row_completion_player();
            return;
        }
        // No more rows — return to MAIN_GAME with opponent of move_maker.
        s.phase = kPhaseMainGame;
        if (s.move_maker_is_black >= 0) {
            s.current_player_is_black = (s.move_maker_is_black == 0);
            s.move_maker_is_black = -1;
        }
        return;
    }
}

inline State ApplyMove(const State& src, const Move& m) noexcept {
    State n = src;
    switch (m.type) {
        case kMovePlaceRing:     HandlePlaceRing(n, m); break;
        case kMoveMoveRing:      HandleMoveRing(n, m); break;
        case kMoveRemoveMarkers: HandleMarkerRemoval(n, m); break;
        case kMoveRemoveRing:    HandleRingRemoval(n, m); break;
        default: return n;  // unknown move type — return unchanged
    }
    UpdatePhaseAndSwitchPlayer(n, m.type, m.player_is_black);
    return n;
}

}  // namespace yinsh::cpp
