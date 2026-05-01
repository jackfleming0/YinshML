// Move generation for the bitboard engine.
//
// One entry point per phase plus a dispatcher GetValidMoves(state).
// Output order is deterministic and matches the Python reference's
// natural order so set-equality comparisons (the parity-test contract)
// hold without sorting on either side.

#pragma once

#include <vector>

#include "apply.hpp"
#include "bitboard.hpp"
#include "moves.hpp"
#include "state.hpp"
#include "tables.hpp"

namespace yinsh::cpp {

// Iterate set bits from low to high.
inline int LowestSetBit(Bitboard b) noexcept {
    if (Lo64(b)) return __builtin_ctzll(Lo64(b));
    return 64 + __builtin_ctzll(Hi64(b));
}

// Generate all PLACE_RING moves: enumerate empty legal cells.
inline void GenRingPlacement(const State& s, std::vector<Move>& out) {
    Bitboard empties = kBoardMask & ~State::AllOccupied(s);
    while (empties) {
        const int cell = LowestSetBit(empties);
        empties &= ~(kOne << cell);
        Move m;
        m.type = kMovePlaceRing;
        m.player_is_black = s.current_player_is_black;
        m.source = static_cast<std::int8_t>(cell);
        out.push_back(m);
    }
}

// Generate all MOVE_RING moves: for each ring of the current player,
// every cell in valid_ring_destinations is one move.
inline void GenRingMovement(const State& s, std::vector<Move>& out) {
    Bitboard rings = s.current_player_is_black ? s.black_rings : s.white_rings;
    const Bitboard all_rings = State::AllRings(s);
    const Bitboard all_markers = State::AllMarkers(s);

    while (rings) {
        const int src = LowestSetBit(rings);
        rings &= ~(kOne << src);
        Bitboard dests = ValidRingDestinations(src, all_rings, all_markers);
        while (dests) {
            const int dst = LowestSetBit(dests);
            dests &= ~(kOne << dst);
            Move m;
            m.type = kMoveMoveRing;
            m.player_is_black = s.current_player_is_black;
            m.source = static_cast<std::int8_t>(src);
            m.destination = static_cast<std::int8_t>(dst);
            out.push_back(m);
        }
    }
}

// Generate all REMOVE_MARKERS moves: every length-5 window over every
// maximal same-color run of length ≥5. Real YINSH allows 6/7-runs, in
// which case multiple windows may slide along the same run; the
// `seen` mask deduplicates windows reachable from overlapping
// detection paths (none should occur with our run-detection scheme,
// but we keep the dedupe for safety against engine evolution).
inline void GenMarkerRemoval(const State& s, std::vector<Move>& out) {
    const Bitboard markers = s.current_player_is_black ? s.black_markers
                                                       : s.white_markers;
    const auto runs = FindMarkerRows(markers);

    // Dedupe by sorted-tuple of cell indices, encoded as a __uint128_t
    // mask. A 5-window is at most 5 bits set — uniquely identifies
    // the window.
    Bitboard seen = 0;  // unused as a mask of windows; placeholder
    (void)seen;
    std::vector<Bitboard> seen_windows;
    seen_windows.reserve(runs.size() * 3);

    for (const auto& run : runs) {
        // Reconstruct the cells of the run in walk order. The board
        // geometry permits runs up to 10 cells (full vertical of column
        // E, full diagonal A2→J11) — sized to 11 with one cell of
        // headroom in case the geometry ever loosens. The CLAUDE.md
        // note about "6/7-marker row support" describes the cases that
        // motivated the move-gen change, not an upper bound.
        std::int8_t cells[11];
        cells[0] = run.start_cell;
        const auto& walk = kRayTable.walk[run.start_cell][run.axis_idx];
        for (int i = 0; i < run.length - 1; ++i) {
            cells[i + 1] = walk[i];
        }

        for (int i = 0; i + 5 <= run.length; ++i) {
            // Sort window's 5 cells ascending so REMOVE_MARKERS moves
            // are canonical, matching Python's
            // _get_marker_removal_moves which sorts windows by
            // (column, row) before deduping.
            std::int8_t win[5];
            for (int k = 0; k < 5; ++k) win[k] = cells[i + k];
            // Insertion sort, n=5 → trivial.
            for (int a = 1; a < 5; ++a) {
                std::int8_t x = win[a];
                int b = a;
                while (b > 0 && win[b - 1] > x) {
                    win[b] = win[b - 1];
                    --b;
                }
                win[b] = x;
            }

            // Build the window mask + dedupe.
            Bitboard mask = 0;
            for (int k = 0; k < 5; ++k) mask |= (kOne << win[k]);
            bool duplicate = false;
            for (Bitboard prev : seen_windows) {
                if (prev == mask) { duplicate = true; break; }
            }
            if (duplicate) continue;
            seen_windows.push_back(mask);

            Move m;
            m.type = kMoveRemoveMarkers;
            m.player_is_black = s.current_player_is_black;
            for (int k = 0; k < 5; ++k) m.markers[k] = win[k];
            out.push_back(m);
        }
    }
}

// Generate all REMOVE_RING moves: every ring of the current player.
inline void GenRingRemoval(const State& s, std::vector<Move>& out) {
    Bitboard rings = s.current_player_is_black ? s.black_rings : s.white_rings;
    while (rings) {
        const int src = LowestSetBit(rings);
        rings &= ~(kOne << src);
        Move m;
        m.type = kMoveRemoveRing;
        m.player_is_black = s.current_player_is_black;
        m.source = static_cast<std::int8_t>(src);
        out.push_back(m);
    }
}

inline std::vector<Move> GetValidMoves(const State& s) {
    std::vector<Move> moves;
    moves.reserve(64);
    switch (s.phase) {
        case kPhaseRingPlacement:  GenRingPlacement(s, moves); break;
        case kPhaseMainGame:       GenRingMovement(s, moves); break;
        case kPhaseRowCompletion:  GenMarkerRemoval(s, moves); break;
        case kPhaseRingRemoval:    GenRingRemoval(s, moves); break;
        case kPhaseGameOver:       break;
        default:                   break;
    }
    return moves;
}

inline bool IsTerminal(const State& s) noexcept {
    return s.phase == kPhaseGameOver;
}

// -1 = no winner (in progress, or stalemate must be checked separately
// since stalemate detection requires move enumeration). 0 = WHITE,
// 1 = BLACK.
inline int Winner(const State& s) noexcept {
    if (s.white_score >= 3) return 0;
    if (s.black_score >= 3) return 1;
    return -1;
}

}  // namespace yinsh::cpp
