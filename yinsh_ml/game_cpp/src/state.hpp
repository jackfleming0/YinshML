// YINSH game state for the bitboard engine.
//
// Trivially-copyable POD by design. Cloning is a struct-copy (~64 bytes
// of memcpy) — the same operation that as a Python deepcopy of the
// reference engine eats 95% of MCTS self-play wall time. Any future
// move generation / apply-move sits on top of this struct as
// stateless free functions returning a new State.
//
// The 4 bitboards encode piece occupancy. Together they should be
// disjoint and cover only kBoardMask cells; off-board bits stay zero.
//
// Player encoding matches Python's Player enum value via a 0/1 bool
// field (`current_player_is_black`). Phase is a small int matching
// GamePhase.value.

#pragma once

#include <cstdint>
#include <cstring>

#include "bitboard.hpp"

namespace yinsh::cpp {

inline constexpr int kPhaseRingPlacement = 0;
inline constexpr int kPhaseMainGame = 1;
inline constexpr int kPhaseRowCompletion = 2;
inline constexpr int kPhaseRingRemoval = 3;
inline constexpr int kPhaseGameOver = 4;

inline constexpr int kRingsPerPlayer = 5;

struct State {
    Bitboard white_rings = 0;
    Bitboard black_rings = 0;
    Bitboard white_markers = 0;
    Bitboard black_markers = 0;

    std::uint8_t phase = kPhaseRingPlacement;
    bool current_player_is_black = false;

    std::uint8_t white_score = 0;
    std::uint8_t black_score = 0;
    std::uint8_t white_rings_placed = 0;
    std::uint8_t black_rings_placed = 0;

    // Row-completion bookkeeping. -1 == "not active" (matches Python's
    // None). Stored as small ints for trivial copyability.
    std::int8_t move_maker_is_black = -1;  // 0 / 1 / -1

    // Anything in this struct that's not yet used (e.g. richer
    // history) will land alongside the move-gen port. Keeping it tiny
    // until then preserves the deepcopy win.

    static constexpr Bitboard AllRings(const State& s) noexcept {
        return s.white_rings | s.black_rings;
    }
    static constexpr Bitboard AllMarkers(const State& s) noexcept {
        return s.white_markers | s.black_markers;
    }
    static constexpr Bitboard AllOccupied(const State& s) noexcept {
        return AllRings(s) | AllMarkers(s);
    }
};

static_assert(std::is_trivially_copyable<State>::value,
              "State must be trivially copyable for the clone() win");

// Copy a State efficiently. With trivial copyability this compiles to
// a memcpy; the function exists so callers don't accidentally write
// `*new_s = *s` and lose the inline-able call site.
inline void CloneState(const State& src, State& dst) noexcept {
    std::memcpy(&dst, &src, sizeof(State));
}

}  // namespace yinsh::cpp
