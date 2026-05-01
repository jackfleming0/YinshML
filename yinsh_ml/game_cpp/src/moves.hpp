// Move representation for the bitboard engine.
//
// One small POD struct covers all four YINSH move types — placing a
// ring, moving a ring, removing 5 markers from a completed row, and
// removing a ring after a marker-removal. Cell indices index into the
// 121-bit board layout from bitboard.hpp.

#pragma once

#include <array>
#include <cstdint>

#include "bitboard.hpp"

namespace yinsh::cpp {

enum MoveType : std::int8_t {
    kMovePlaceRing = 0,
    kMoveMoveRing = 1,
    kMoveRemoveMarkers = 2,
    kMoveRemoveRing = 3,
};

struct Move {
    std::int8_t type = -1;
    bool player_is_black = false;
    std::int8_t source = -1;       // cell index for PLACE_RING / MOVE_RING / REMOVE_RING
    std::int8_t destination = -1;  // cell index for MOVE_RING
    // 5 cell indices for REMOVE_MARKERS, sorted ascending. -1 padding
    // for other move types so equality and hashing remain deterministic.
    std::array<std::int8_t, 5> markers = {-1, -1, -1, -1, -1};

    constexpr bool operator==(const Move& o) const noexcept {
        return type == o.type
            && player_is_black == o.player_is_black
            && source == o.source
            && destination == o.destination
            && markers == o.markers;
    }
};

}  // namespace yinsh::cpp
