// Compile-time-generated ray tables for the YINSH bitboard engine.
//
// For each of the 121 cells and each of the 6 hex directions, store a
// bitboard of cells reachable along that ray (excluding the starting
// cell, intersected with the legal-board mask). Move generation, line
// scanning and row detection all reduce to one or two table lookups.
//
// All tables are constexpr and live in the read-only segment — zero
// runtime init, zero allocation, no startup cost.

#pragma once

#include <array>
#include <cstdint>

#include "bitboard.hpp"

namespace yinsh::cpp {

// Hex directions in canonical order. Indices 0..2 are the forward axes
// used by row detection; the full 6 are used by ring-move generation
// and ray scans. The bit-stride for each direction is precomputed below
// so directional walks reduce to integer addition on the cell index.
struct Direction { int dcol; int drow; int stride; };

inline constexpr std::array<Direction, 6> kHexDirections = {{
    { 0,  1,  +1},   // 0 vertical +
    { 0, -1,  -1},   // 1 vertical -
    { 1,  0, +11},   // 2 horizontal +
    {-1,  0, -11},   // 3 horizontal -
    { 1,  1, +12},   // 4 diagonal +  (matching-sign)
    {-1, -1, -12},   // 5 diagonal -
}};

// Forward-only axes (the three "lower-end" directions), for unique
// row enumeration along each axis. Mirrors constants.DIRECTIONS.
inline constexpr std::array<Direction, 3> kForwardAxes = {{
    { 0, 1,  +1},
    { 1, 0, +11},
    { 1, 1, +12},
}};

constexpr Bitboard RayFrom(int col_idx, int row, Direction d) noexcept {
    Bitboard ray = 0;
    int c = col_idx + d.dcol;
    int r = row + d.drow;
    while (IsValidCell(c, r)) {
        ray |= (kOne << CellIndex(c, r));
        c += d.dcol;
        r += d.drow;
    }
    return ray;
}

struct RayTable {
    // [cell_index][dir_index] — bitboard of all legal cells reachable
    // along the ray, starting just past the cell. Off-board cells get
    // empty entries (they're never used).
    std::array<std::array<Bitboard, 6>, kCellCount> rays{};
    // Same data but ordered as a sequence of cell indices in walk
    // order, terminated with -1. Lets the move generator iterate in
    // geometric order without re-decoding the bitboard for reverse
    // directions. Max ray length on an 11x11 board is 10 cells.
    std::array<std::array<std::array<int8_t, 11>, 6>, kCellCount> walk{};
};

constexpr RayTable MakeRayTable() noexcept {
    RayTable t{};
    for (int col = 0; col < kCols; ++col) {
        for (int row = 1; row <= kRows; ++row) {
            if (!IsValidCell(col, row)) continue;
            const int idx = CellIndex(col, row);
            for (std::size_t d = 0; d < kHexDirections.size(); ++d) {
                t.rays[idx][d] = RayFrom(col, row, kHexDirections[d]);

                // Walk-order: step in the direction, stop at first
                // off-board cell. Sentinel -1 marks the end.
                int c = col + kHexDirections[d].dcol;
                int r = row + kHexDirections[d].drow;
                std::size_t k = 0;
                while (IsValidCell(c, r) && k < t.walk[idx][d].size()) {
                    t.walk[idx][d][k++] = static_cast<int8_t>(CellIndex(c, r));
                    c += kHexDirections[d].dcol;
                    r += kHexDirections[d].drow;
                }
                for (; k < t.walk[idx][d].size(); ++k) {
                    t.walk[idx][d][k] = -1;
                }
            }
        }
    }
    return t;
}

inline constexpr RayTable kRayTable = MakeRayTable();

// One maximal run of same-color markers along a single hex axis.
// `start_cell` is the lower-end of the run (no same-color marker at
// `predecessor(start_cell, axis)`). `axis_idx` is the forward-axis
// index in kHexDirections (always 0, 2, or 4). `length` is the number
// of consecutive markers, 5 ≤ length ≤ 7 for runs reported by
// FindMarkerRows.
struct MarkerRun {
    int8_t start_cell;
    int8_t axis_idx;
    int8_t length;
};

// Find all maximal runs of `markers` along the 3 forward hex axes
// whose length ≥ 5. By only walking from each run's lower end (i.e.
// cells whose reverse-axis predecessor is not a same-color marker),
// each run is enumerated exactly once — no dedup pass needed.
//
// Mirrors Board.find_marker_rows from the Python engine, including
// support for 5/6/7-length runs (real YINSH allows extending a 5-row).
inline std::vector<MarkerRun> FindMarkerRows(Bitboard markers) noexcept {
    std::vector<MarkerRun> runs;
    runs.reserve(8);

    // Forward-axis indices into kHexDirections: vertical+, horizontal+,
    // matching-sign diagonal+. Reverse axis is always forward+1.
    static constexpr int kForwardAxisIndices[3] = {0, 2, 4};

    Bitboard remaining = markers;
    while (remaining) {
        int cell;
        if (Lo64(remaining)) {
            cell = __builtin_ctzll(Lo64(remaining));
        } else {
            cell = 64 + __builtin_ctzll(Hi64(remaining));
        }
        const Bitboard cell_bit = kOne << cell;
        remaining &= ~cell_bit;

        for (int fa = 0; fa < 3; ++fa) {
            const int forward_idx = kForwardAxisIndices[fa];
            const int reverse_idx = forward_idx + 1;

            // Predecessor along the reverse axis: if it's a same-color
            // marker, this cell is mid-run — skip.
            const int8_t prev_cell = kRayTable.walk[cell][reverse_idx][0];
            if (prev_cell >= 0 && (markers & (kOne << prev_cell))) {
                continue;
            }

            // Walk forward counting consecutive same-color markers.
            int length = 1;  // includes start cell
            const auto& walk = kRayTable.walk[cell][forward_idx];
            for (int8_t next : walk) {
                if (next < 0) break;
                if (!(markers & (kOne << next))) break;
                ++length;
            }

            if (length >= 5) {
                runs.push_back({static_cast<int8_t>(cell),
                                static_cast<int8_t>(forward_idx),
                                static_cast<int8_t>(length)});
            }
        }
    }

    return runs;
}

// YINSH ring-move semantics:
//   - From `source_cell`, walk in each of the 6 directions.
//   - Stop the walk on a ring (rings block).
//   - Empty cells before any marker are valid destinations.
//   - After crossing one or more contiguous markers, the FIRST empty
//     cell is a valid destination AND ends the walk for that direction.
//
// Returns a bitboard of valid destinations.
inline Bitboard ValidRingDestinations(int source_cell,
                                      Bitboard rings,
                                      Bitboard markers) noexcept {
    Bitboard out = 0;
    for (std::size_t d = 0; d < 6; ++d) {
        const auto& walk = kRayTable.walk[source_cell][d];
        bool jumped = false;
        for (int8_t next : walk) {
            if (next < 0) break;
            const Bitboard nb = kOne << next;
            if (rings & nb) {
                break;
            }
            if (markers & nb) {
                jumped = true;
                continue;
            }
            // empty cell
            out |= nb;
            if (jumped) break;
        }
    }
    return out;
}

}  // namespace yinsh::cpp
