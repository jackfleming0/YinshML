// YINSH bitboard primitives.
//
// Layout: the 11x11 (column, row) grid maps to a 121-bit field stored in
// a single __uint128_t. Bit index = col_idx * 11 + (row - 1), where
// col_idx is 0..10 for columns A..K and row is 1..11. This keeps the
// three forward hex axes as constant-stride shifts:
//
//   vertical   (0, +1)  -> +1
//   horizontal (+1, 0)  -> +11
//   diagonal   (+1, +1) -> +12   (matching-sign diagonal — the only
//                                  legal hex axis besides the two above)
//
// Only 85 of the 121 cells are valid YINSH positions; the rest sit
// outside the hex board. We carry an `kBoardMask` bitboard so every
// shift result can be masked back into the legal set in one op.

#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace yinsh::cpp {

using Bitboard = unsigned __int128;

inline constexpr int kCols = 11;
inline constexpr int kRows = 11;
inline constexpr int kCellCount = kCols * kRows;  // 121
inline constexpr int kValidCellCount = 85;

// Forward-only hex axes. Each of the 6 directions in
// constants.HEX_DIRECTIONS is one of these axes negated; we encode bit
// shifts as signed ints so a single template handles both directions.
inline constexpr int kAxisVertical = 1;    // (0, +1)
inline constexpr int kAxisHorizontal = 11; // (+1, 0)
inline constexpr int kAxisDiagonal = 12;   // (+1, +1)

inline constexpr Bitboard kOne = static_cast<Bitboard>(1);

// Per-column valid-row ranges, mirroring constants.VALID_POSITIONS in
// the Python engine. Inclusive on both ends.
struct ColumnRange { int min_row; int max_row; };
inline constexpr std::array<ColumnRange, kCols> kColumnRanges = {{
    {2, 5},   // A
    {1, 7},   // B
    {1, 8},   // C
    {1, 9},   // D
    {1, 10},  // E
    {2, 10},  // F
    {2, 11},  // G
    {3, 11},  // H
    {4, 11},  // I
    {5, 11},  // J
    {7, 10},  // K
}};

// Bit index for a (col_idx, row) pair. row is 1-based to match the
// game's algebraic notation. No bounds checking — caller's job.
constexpr int CellIndex(int col_idx, int row) noexcept {
    return col_idx * kRows + (row - 1);
}

constexpr bool IsValidCell(int col_idx, int row) noexcept {
    if (col_idx < 0 || col_idx >= kCols) return false;
    if (row < 1 || row > kRows) return false;
    const auto& r = kColumnRanges[col_idx];
    return row >= r.min_row && row <= r.max_row;
}

// Compile-time-built mask of all 85 legal YINSH positions. Built from
// kColumnRanges so it is provably consistent with the Python engine's
// VALID_POSITIONS.
constexpr Bitboard MakeBoardMask() noexcept {
    Bitboard m = 0;
    for (int col = 0; col < kCols; ++col) {
        const auto& r = kColumnRanges[col];
        for (int row = r.min_row; row <= r.max_row; ++row) {
            m |= (kOne << CellIndex(col, row));
        }
    }
    return m;
}

inline constexpr Bitboard kBoardMask = MakeBoardMask();

// __uint128_t isn't directly representable as a CPython int, but it
// round-trips cleanly through two 64-bit halves. These helpers keep
// the binding layer honest about endianness (low-half first).
inline constexpr std::uint64_t Lo64(Bitboard b) noexcept {
    return static_cast<std::uint64_t>(b);
}
inline constexpr std::uint64_t Hi64(Bitboard b) noexcept {
    return static_cast<std::uint64_t>(b >> 64);
}
inline constexpr Bitboard FromHalves(std::uint64_t lo, std::uint64_t hi) noexcept {
    return (static_cast<Bitboard>(hi) << 64) | static_cast<Bitboard>(lo);
}

inline int Popcount(Bitboard b) noexcept {
    return __builtin_popcountll(Lo64(b)) + __builtin_popcountll(Hi64(b));
}

}  // namespace yinsh::cpp
