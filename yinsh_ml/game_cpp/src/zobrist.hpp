// Zobrist hashing primitives for the bitboard engine.
//
// The C++ side is intentionally dumb: it takes a precomputed table from
// Python at construction time and does fast XOR ops over bitboards.
// All keys (per-(cell,piece) values, side-to-move key, per-phase keys)
// come from yinsh_ml.game.zobrist.ZobristInitializer, which means a
// C++ hash is bit-exact identical to a Python hash for the same state
// — no need to reproduce Python's blake2b key derivation.
//
// Piece index convention (matches Python's DEFAULT_PIECE_ORDER):
//   0 = EMPTY
//   1 = WHITE_RING
//   2 = BLACK_RING
//   3 = WHITE_MARKER
//   4 = BLACK_MARKER

#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>

#include "bitboard.hpp"

namespace yinsh::cpp {

inline constexpr int kPieceCount = 5;
inline constexpr int kEmptyPiece = 0;
inline constexpr int kWhiteRing = 1;
inline constexpr int kBlackRing = 2;
inline constexpr int kWhiteMarker = 3;
inline constexpr int kBlackMarker = 4;

// Index into the flat (cell_count × piece_count) Zobrist table.
constexpr int ZobristIndex(int cell, int piece) noexcept {
    return cell * kPieceCount + piece;
}

class Zobrist {
public:
    // `piece_keys` must be length kCellCount × kPieceCount in row-major
    // (cell × piece) order. Off-board cells should hold zeros (never
    // indexed). `phase_keys` length should match the number of game
    // phases on the Python side; we don't enforce a specific count
    // here so the binding can pass exactly the keys present.
    Zobrist(const std::uint64_t* piece_keys,
            std::uint64_t side_to_move_key,
            const std::uint64_t* phase_keys,
            int phase_key_count)
        : side_to_move_key_(side_to_move_key),
          phase_key_count_(phase_key_count) {
        for (int i = 0; i < kCellCount * kPieceCount; ++i) {
            piece_keys_[i] = piece_keys[i];
        }
        if (phase_key_count > kMaxPhases) {
            throw std::invalid_argument("too many phase keys");
        }
        for (int i = 0; i < phase_key_count; ++i) {
            phase_keys_[i] = phase_keys[i];
        }
        empty_hash_ = ComputeEmptyHash();
    }

    std::uint64_t empty_hash() const noexcept { return empty_hash_; }
    std::uint64_t side_to_move_key() const noexcept { return side_to_move_key_; }
    std::uint64_t phase_key(int phase) const {
        if (phase < 0 || phase >= phase_key_count_) {
            throw std::out_of_range("phase index out of range");
        }
        return phase_keys_[phase];
    }

    // Full board hash from the four piece-occupancy bitboards. Mirrors
    // ZobristHasher.hash_board: starts at empty_hash_, then for every
    // occupied cell XORs out the EMPTY contribution and XORs in the
    // piece contribution.
    std::uint64_t HashBoard(Bitboard white_rings,
                            Bitboard black_rings,
                            Bitboard white_markers,
                            Bitboard black_markers) const noexcept {
        std::uint64_t h = empty_hash_;
        h = XorPieces(h, white_rings, kWhiteRing);
        h = XorPieces(h, black_rings, kBlackRing);
        h = XorPieces(h, white_markers, kWhiteMarker);
        h = XorPieces(h, black_markers, kBlackMarker);
        return h;
    }

    // Full state hash: board + side-to-move (XOR if BLACK to move) +
    // phase. `current_player_is_black` is a bool (the Python side
    // toggles only on BLACK).
    std::uint64_t HashState(Bitboard white_rings,
                            Bitboard black_rings,
                            Bitboard white_markers,
                            Bitboard black_markers,
                            bool current_player_is_black,
                            int phase_idx) const {
        std::uint64_t h = HashBoard(white_rings, black_rings,
                                    white_markers, black_markers);
        if (current_player_is_black) h ^= side_to_move_key_;
        h ^= phase_key(phase_idx);
        return h;
    }

    // Toggle a single (cell, piece) contribution into a running hash.
    std::uint64_t Toggle(std::uint64_t h, int cell, int piece) const noexcept {
        return h ^ piece_keys_[ZobristIndex(cell, piece)];
    }

    // Replace one piece with another at the same cell.
    std::uint64_t UpdatePosition(std::uint64_t h, int cell,
                                 int old_piece, int new_piece) const noexcept {
        return h
            ^ piece_keys_[ZobristIndex(cell, old_piece)]
            ^ piece_keys_[ZobristIndex(cell, new_piece)];
    }

    std::uint64_t FlipSide(std::uint64_t h) const noexcept {
        return h ^ side_to_move_key_;
    }

private:
    static constexpr int kMaxPhases = 8;  // overprovision; Python has 5

    std::uint64_t XorPieces(std::uint64_t h, Bitboard occupied,
                            int piece_idx) const noexcept {
        // For each occupied cell: XOR out EMPTY, XOR in piece.
        Bitboard rest = occupied;
        while (rest) {
            int cell;
            if (Lo64(rest)) {
                cell = __builtin_ctzll(Lo64(rest));
            } else {
                cell = 64 + __builtin_ctzll(Hi64(rest));
            }
            rest &= ~(kOne << cell);
            h ^= piece_keys_[ZobristIndex(cell, kEmptyPiece)];
            h ^= piece_keys_[ZobristIndex(cell, piece_idx)];
        }
        return h;
    }

    std::uint64_t ComputeEmptyHash() const noexcept {
        // Python's empty_hash XORs in the EMPTY key for every VALID
        // position. We only have 85 valid cells; off-board cells have
        // a zero key (passed as zero from Python) so they're harmless,
        // but we still iterate all kCellCount cells for symmetry.
        std::uint64_t h = 0;
        for (int cell = 0; cell < kCellCount; ++cell) {
            h ^= piece_keys_[ZobristIndex(cell, kEmptyPiece)];
        }
        return h;
    }

    std::array<std::uint64_t, kCellCount * kPieceCount> piece_keys_{};
    std::array<std::uint64_t, kMaxPhases> phase_keys_{};
    std::uint64_t side_to_move_key_;
    int phase_key_count_;
    std::uint64_t empty_hash_ = 0;
};

}  // namespace yinsh::cpp
