// Pybind11 entry point for the YINSH C++ engine.
//
// First slice surface:
//   * board layout probes (cell_index, is_valid_cell, board_mask)
//   * ray-table-driven `valid_ring_destinations`, the analogue of
//     Board.valid_move_positions, which alongside find_marker_rows
//     was the single biggest hot-path cost in the Python engine
//     profile (24%/11% self time).
//   * a self-bench hook so we can land a perf datapoint without
//     paying Python-level call overhead per iteration.
//
// No GameState / move objects yet — those land alongside move
// generation, where parity-test scaffolding will need them.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <random>
#include <vector>

#include <memory>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "apply.hpp"
#include "bitboard.hpp"
#include "movegen.hpp"
#include "moves.hpp"
#include "state.hpp"
#include "tables.hpp"
#include "zobrist.hpp"

namespace py = pybind11;

namespace {

using namespace yinsh::cpp;

py::tuple BoardMaskHalves() {
    return py::make_tuple(Lo64(kBoardMask), Hi64(kBoardMask));
}

int CellIndexPy(int col_idx, int row) {
    if (!IsValidCell(col_idx, row)) {
        throw py::value_error("invalid (col_idx, row) for YINSH board");
    }
    return CellIndex(col_idx, row);
}

bool IsValidCellPy(int col_idx, int row) {
    return IsValidCell(col_idx, row);
}

int PopcountHalves(std::uint64_t lo, std::uint64_t hi) {
    return Popcount(FromHalves(lo, hi));
}

// Returns one cell-index list per maximal run, with cells in walk order
// from the run's lower end. Empty list = no runs of length ≥5. The
// caller (Python wrapper / parity test) re-keys these into the Position
// objects the rest of the engine expects.
std::vector<std::vector<int>> FindMarkerRowsPy(std::uint64_t markers_lo,
                                               std::uint64_t markers_hi) {
    const Bitboard markers = FromHalves(markers_lo, markers_hi);
    const auto runs = FindMarkerRows(markers);

    std::vector<std::vector<int>> out;
    out.reserve(runs.size());
    for (const auto& run : runs) {
        std::vector<int> cells;
        cells.reserve(run.length);
        cells.push_back(run.start_cell);
        const auto& walk = kRayTable.walk[run.start_cell][run.axis_idx];
        for (int i = 0; i < run.length - 1; ++i) {
            cells.push_back(walk[i]);
        }
        out.push_back(std::move(cells));
    }
    return out;
}

// Bench: rotate through prebuilt marker bitboards so the loop body
// can't be folded.
double BenchFindMarkerRows(const std::vector<std::uint64_t>& markers_lo,
                           const std::vector<std::uint64_t>& markers_hi,
                           int iters) {
    const std::size_t n = markers_lo.size();
    if (n == 0 || markers_hi.size() != n) {
        throw py::value_error("input vectors must be non-empty and equal length");
    }
    py::gil_scoped_release release;
    std::size_t sink = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        const std::size_t k = static_cast<std::size_t>(i) % n;
        const auto runs = FindMarkerRows(FromHalves(markers_lo[k], markers_hi[k]));
        sink ^= runs.size();
    }
    auto t1 = std::chrono::steady_clock::now();
    if (sink == ~std::size_t{0}) return -1.0;  // optimizer fence
    return std::chrono::duration<double>(t1 - t0).count();
}

// Returns (lo, hi) of the destinations bitboard. Caller (Python) assembles.
py::tuple ValidRingDestinationsPy(int source_cell,
                                  std::uint64_t rings_lo,
                                  std::uint64_t rings_hi,
                                  std::uint64_t markers_lo,
                                  std::uint64_t markers_hi) {
    if (source_cell < 0 || source_cell >= kCellCount) {
        throw py::value_error("source_cell out of range");
    }
    const Bitboard rings = FromHalves(rings_lo, rings_hi);
    const Bitboard markers = FromHalves(markers_lo, markers_hi);
    const Bitboard dests = ValidRingDestinations(source_cell, rings, markers);
    return py::make_tuple(Lo64(dests), Hi64(dests));
}

// Microbenchmark: call ValidRingDestinations `iters` times rotating
// through a list of (source_cell, rings, markers) triples so the
// compiler can't fold the loop body to a constant. Returns total
// seconds. Releases the GIL.
//
// The triples are passed flat as four parallel sequences from Python:
// `sources` (cells), and three uint64 lists for rings_lo/hi and
// markers_lo/hi. Length must match.
double BenchValidRingDestinationsVaried(
    const std::vector<int>& sources,
    const std::vector<std::uint64_t>& rings_lo,
    const std::vector<std::uint64_t>& rings_hi,
    const std::vector<std::uint64_t>& markers_lo,
    const std::vector<std::uint64_t>& markers_hi,
    int iters) {
    const std::size_t n = sources.size();
    if (n == 0) {
        throw py::value_error("need at least one input triple");
    }
    if (rings_lo.size() != n || rings_hi.size() != n ||
        markers_lo.size() != n || markers_hi.size() != n) {
        throw py::value_error("input vectors must have matching length");
    }
    for (int s : sources) {
        if (s < 0 || s >= kCellCount) {
            throw py::value_error("source_cell out of range");
        }
    }

    py::gil_scoped_release release;
    Bitboard sink = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        const std::size_t k = static_cast<std::size_t>(i) % n;
        sink ^= ValidRingDestinations(
            sources[k],
            FromHalves(rings_lo[k], rings_hi[k]),
            FromHalves(markers_lo[k], markers_hi[k]));
    }
    auto t1 = std::chrono::steady_clock::now();
    if (sink == ~Bitboard{0}) {
        return -1.0;  // keeps `sink` observable to the optimizer
    }
    return std::chrono::duration<double>(t1 - t0).count();
}

}  // namespace

// --- State binding ---------------------------------------------------
//
// Surfaces the POD State struct as a Python class. The whole point of
// this slice is to prove the clone() win — a struct-copy that
// substitutes for `copy.deepcopy(GameState)`. Move-gen will land on
// top of this in a follow-up.

class PyState {
public:
    PyState() = default;
    explicit PyState(const State& s) : s_(s) {}

    // --- bitboard getters/setters (split halves for Python ints) ---
    py::tuple white_rings() const { return py::make_tuple(Lo64(s_.white_rings), Hi64(s_.white_rings)); }
    py::tuple black_rings() const { return py::make_tuple(Lo64(s_.black_rings), Hi64(s_.black_rings)); }
    py::tuple white_markers() const { return py::make_tuple(Lo64(s_.white_markers), Hi64(s_.white_markers)); }
    py::tuple black_markers() const { return py::make_tuple(Lo64(s_.black_markers), Hi64(s_.black_markers)); }

    void set_white_rings(std::uint64_t lo, std::uint64_t hi) { s_.white_rings = FromHalves(lo, hi); }
    void set_black_rings(std::uint64_t lo, std::uint64_t hi) { s_.black_rings = FromHalves(lo, hi); }
    void set_white_markers(std::uint64_t lo, std::uint64_t hi) { s_.white_markers = FromHalves(lo, hi); }
    void set_black_markers(std::uint64_t lo, std::uint64_t hi) { s_.black_markers = FromHalves(lo, hi); }

    int phase() const { return s_.phase; }
    void set_phase(int p) { s_.phase = static_cast<std::uint8_t>(p); }
    bool current_player_is_black() const { return s_.current_player_is_black; }
    void set_current_player_is_black(bool b) { s_.current_player_is_black = b; }
    int white_score() const { return s_.white_score; }
    void set_white_score(int v) { s_.white_score = static_cast<std::uint8_t>(v); }
    int black_score() const { return s_.black_score; }
    void set_black_score(int v) { s_.black_score = static_cast<std::uint8_t>(v); }
    int white_rings_placed() const { return s_.white_rings_placed; }
    void set_white_rings_placed(int v) { s_.white_rings_placed = static_cast<std::uint8_t>(v); }
    int black_rings_placed() const { return s_.black_rings_placed; }
    void set_black_rings_placed(int v) { s_.black_rings_placed = static_cast<std::uint8_t>(v); }
    int move_maker_is_black() const { return s_.move_maker_is_black; }
    void set_move_maker_is_black(int v) { s_.move_maker_is_black = static_cast<std::int8_t>(v); }

    PyState clone() const {
        PyState copy;
        CloneState(s_, copy.s_);
        return copy;
    }

    bool equals(const PyState& other) const noexcept {
        return std::memcmp(&s_, &other.s_, sizeof(State)) == 0;
    }

    const State& raw() const noexcept { return s_; }
    State& raw_mut() noexcept { return s_; }

private:
    State s_;
};

// --- Move binding ---------------------------------------------------

class PyMove {
public:
    PyMove() = default;
    explicit PyMove(const Move& m) : m_(m) {}

    static PyMove place_ring(bool player_is_black, int source) {
        Move m;
        m.type = kMovePlaceRing;
        m.player_is_black = player_is_black;
        m.source = static_cast<std::int8_t>(source);
        return PyMove{m};
    }
    static PyMove move_ring(bool player_is_black, int source, int destination) {
        Move m;
        m.type = kMoveMoveRing;
        m.player_is_black = player_is_black;
        m.source = static_cast<std::int8_t>(source);
        m.destination = static_cast<std::int8_t>(destination);
        return PyMove{m};
    }
    static PyMove remove_markers(bool player_is_black,
                                 const std::vector<int>& cells) {
        if (cells.size() != 5) {
            throw py::value_error("remove_markers requires exactly 5 cells");
        }
        Move m;
        m.type = kMoveRemoveMarkers;
        m.player_is_black = player_is_black;
        // Sort ascending so a Python dataclass-derived move and a
        // get_valid_moves-derived move with the same set of cells
        // round-trip identically.
        std::array<int, 5> sorted_cells{cells[0], cells[1], cells[2],
                                        cells[3], cells[4]};
        std::sort(sorted_cells.begin(), sorted_cells.end());
        for (int k = 0; k < 5; ++k) {
            m.markers[k] = static_cast<std::int8_t>(sorted_cells[k]);
        }
        return PyMove{m};
    }
    static PyMove remove_ring(bool player_is_black, int source) {
        Move m;
        m.type = kMoveRemoveRing;
        m.player_is_black = player_is_black;
        m.source = static_cast<std::int8_t>(source);
        return PyMove{m};
    }

    int type() const { return m_.type; }
    bool player_is_black() const { return m_.player_is_black; }
    int source() const { return m_.source; }
    int destination() const { return m_.destination; }
    std::vector<int> markers() const {
        std::vector<int> out;
        for (auto c : m_.markers) {
            if (c < 0) break;
            out.push_back(c);
        }
        return out;
    }

    bool equals(const PyMove& o) const noexcept { return m_ == o.m_; }

    const Move& raw() const noexcept { return m_; }

private:
    Move m_;
};

PyState ApplyMovePy(const PyState& s, const PyMove& m) {
    return PyState{ApplyMove(s.raw(), m.raw())};
}

std::vector<PyMove> GetValidMovesPy(const PyState& s) {
    auto moves = GetValidMoves(s.raw());
    std::vector<PyMove> out;
    out.reserve(moves.size());
    for (const auto& m : moves) out.emplace_back(m);
    return out;
}

bool IsTerminalPy(const PyState& s) { return IsTerminal(s.raw()); }
int WinnerPy(const PyState& s) { return Winner(s.raw()); }

// Bench: drive `n_games` full random playouts entirely in C++. Returns
// (total_seconds, total_moves) so the caller can compute moves/sec and
// compare against the Python random-playout harness in
// scripts/profile_engine.py. Releases the GIL: the full game loop is
// in C++.
py::tuple BenchRandomPlayouts(int n_games, int max_moves, std::uint64_t seed) {
    std::mt19937_64 rng(seed);

    py::gil_scoped_release release;
    int total_moves = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int g = 0; g < n_games; ++g) {
        State s;
        for (int ply = 0; ply < max_moves; ++ply) {
            if (IsTerminal(s)) break;
            const auto moves = GetValidMoves(s);
            if (moves.empty()) break;  // stalemate
            std::uniform_int_distribution<std::size_t> pick(0, moves.size() - 1);
            s = ApplyMove(s, moves[pick(rng)]);
            ++total_moves;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    py::gil_scoped_acquire acquire;
    return py::make_tuple(secs, total_moves);
}

// Bench: clone a State `iters` times. Releases the GIL so the number
// reflects pure C++ memcpy throughput. The asm volatile barrier after
// each clone tells the compiler `dst` was potentially modified through
// memory it can't see, which forces the memcpy to actually happen
// instead of being hoisted out as constant-fold.
double BenchCloneState(const PyState& src, int iters) {
    py::gil_scoped_release release;
    State dst;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        CloneState(src.raw(), dst);
        asm volatile("" : "+m"(dst));  // optimization fence
    }
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

// --- Zobrist binding -------------------------------------------------
//
// The Python ZobristInitializer is the source of truth for the random
// keys. We just take a flattened table and a few scalars and stand up
// a C++ Zobrist that produces bit-exact-identical hashes.

class PyZobrist {
public:
    PyZobrist(py::array_t<std::uint64_t> piece_keys,
              std::uint64_t side_to_move_key,
              py::array_t<std::uint64_t> phase_keys) {
        auto pk = piece_keys.unchecked<1>();
        if (pk.shape(0) != kCellCount * kPieceCount) {
            throw py::value_error(
                "piece_keys must be flat (cell × piece), length 605");
        }
        std::array<std::uint64_t, kCellCount * kPieceCount> piece_arr{};
        for (py::ssize_t i = 0; i < pk.shape(0); ++i) piece_arr[i] = pk(i);

        auto ph = phase_keys.unchecked<1>();
        std::array<std::uint64_t, 8> phase_arr{};
        const int n_phases = static_cast<int>(ph.shape(0));
        if (n_phases > 8) throw py::value_error("too many phase keys");
        for (int i = 0; i < n_phases; ++i) phase_arr[i] = ph(i);

        impl_ = std::make_unique<Zobrist>(
            piece_arr.data(), side_to_move_key,
            phase_arr.data(), n_phases);
    }

    std::uint64_t empty_hash() const { return impl_->empty_hash(); }
    std::uint64_t side_to_move_key() const { return impl_->side_to_move_key(); }
    std::uint64_t phase_key(int phase) const { return impl_->phase_key(phase); }

    std::uint64_t hash_board(std::uint64_t wr_lo, std::uint64_t wr_hi,
                             std::uint64_t br_lo, std::uint64_t br_hi,
                             std::uint64_t wm_lo, std::uint64_t wm_hi,
                             std::uint64_t bm_lo, std::uint64_t bm_hi) const {
        return impl_->HashBoard(
            FromHalves(wr_lo, wr_hi), FromHalves(br_lo, br_hi),
            FromHalves(wm_lo, wm_hi), FromHalves(bm_lo, bm_hi));
    }

    std::uint64_t hash_state(std::uint64_t wr_lo, std::uint64_t wr_hi,
                             std::uint64_t br_lo, std::uint64_t br_hi,
                             std::uint64_t wm_lo, std::uint64_t wm_hi,
                             std::uint64_t bm_lo, std::uint64_t bm_hi,
                             bool current_player_is_black,
                             int phase_idx) const {
        return impl_->HashState(
            FromHalves(wr_lo, wr_hi), FromHalves(br_lo, br_hi),
            FromHalves(wm_lo, wm_hi), FromHalves(bm_lo, bm_hi),
            current_player_is_black, phase_idx);
    }

    std::uint64_t toggle(std::uint64_t h, int cell, int piece) const {
        return impl_->Toggle(h, cell, piece);
    }

    std::uint64_t update_position(std::uint64_t h, int cell,
                                  int old_piece, int new_piece) const {
        return impl_->UpdatePosition(h, cell, old_piece, new_piece);
    }

    std::uint64_t flip_side(std::uint64_t h) const {
        return impl_->FlipSide(h);
    }

private:
    std::unique_ptr<Zobrist> impl_;
};

PYBIND11_MODULE(_engine, m) {
    m.doc() = "YINSH C++ bitboard engine (work in progress)";

    m.attr("CELL_COUNT") = kCellCount;
    m.attr("VALID_CELL_COUNT") = kValidCellCount;
    m.attr("AXIS_VERTICAL") = kAxisVertical;
    m.attr("AXIS_HORIZONTAL") = kAxisHorizontal;
    m.attr("AXIS_DIAGONAL") = kAxisDiagonal;

    m.def("cell_index", &CellIndexPy,
          "Bitboard bit index for (col_idx, row).",
          py::arg("col_idx"), py::arg("row"));
    m.def("is_valid_cell", &IsValidCellPy,
          "True iff (col_idx, row) is a legal YINSH position.",
          py::arg("col_idx"), py::arg("row"));
    m.def("board_mask_halves", &BoardMaskHalves,
          "121-bit board mask as (lo64, hi64).");
    m.def("popcount_halves", &PopcountHalves,
          py::arg("lo"), py::arg("hi"));

    m.def("valid_ring_destinations",
          &ValidRingDestinationsPy,
          "Compute valid ring-move destinations for a ring at "
          "`source_cell`, given current ring/marker occupancy. Returns "
          "the destinations bitboard as (lo, hi).",
          py::arg("source_cell"),
          py::arg("rings_lo"), py::arg("rings_hi"),
          py::arg("markers_lo"), py::arg("markers_hi"));

    m.def("find_marker_rows",
          &FindMarkerRowsPy,
          "Find all maximal same-color marker runs of length ≥5 along "
          "the 3 forward hex axes. Returns a list of cell-index lists, "
          "each in walk order from the run's lower end.",
          py::arg("markers_lo"), py::arg("markers_hi"));

    m.def("bench_find_marker_rows",
          &BenchFindMarkerRows,
          "Run `iters` find_marker_rows calls rotating through markers "
          "bitboards. Returns total seconds.",
          py::arg("markers_lo"), py::arg("markers_hi"), py::arg("iters"));

    py::class_<PyState>(m, "State")
        .def(py::init<>())
        .def("clone", &PyState::clone,
             "Return a deep copy of this State. With trivially-copyable "
             "internals this is a single struct memcpy, ~64 bytes.")
        .def("equals", &PyState::equals)
        .def_property("white_rings",
                      &PyState::white_rings,
                      [](PyState& s, py::tuple t) {
                          s.set_white_rings(t[0].cast<std::uint64_t>(),
                                            t[1].cast<std::uint64_t>());
                      })
        .def_property("black_rings",
                      &PyState::black_rings,
                      [](PyState& s, py::tuple t) {
                          s.set_black_rings(t[0].cast<std::uint64_t>(),
                                            t[1].cast<std::uint64_t>());
                      })
        .def_property("white_markers",
                      &PyState::white_markers,
                      [](PyState& s, py::tuple t) {
                          s.set_white_markers(t[0].cast<std::uint64_t>(),
                                              t[1].cast<std::uint64_t>());
                      })
        .def_property("black_markers",
                      &PyState::black_markers,
                      [](PyState& s, py::tuple t) {
                          s.set_black_markers(t[0].cast<std::uint64_t>(),
                                              t[1].cast<std::uint64_t>());
                      })
        .def_property("phase", &PyState::phase, &PyState::set_phase)
        .def_property("current_player_is_black",
                      &PyState::current_player_is_black,
                      &PyState::set_current_player_is_black)
        .def_property("white_score",
                      &PyState::white_score, &PyState::set_white_score)
        .def_property("black_score",
                      &PyState::black_score, &PyState::set_black_score)
        .def_property("white_rings_placed",
                      &PyState::white_rings_placed,
                      &PyState::set_white_rings_placed)
        .def_property("black_rings_placed",
                      &PyState::black_rings_placed,
                      &PyState::set_black_rings_placed)
        .def_property("move_maker_is_black",
                      &PyState::move_maker_is_black,
                      &PyState::set_move_maker_is_black);

    py::class_<PyMove>(m, "Move")
        .def_static("place_ring", &PyMove::place_ring,
                    py::arg("player_is_black"), py::arg("source"))
        .def_static("move_ring", &PyMove::move_ring,
                    py::arg("player_is_black"),
                    py::arg("source"), py::arg("destination"))
        .def_static("remove_markers", &PyMove::remove_markers,
                    py::arg("player_is_black"), py::arg("cells"))
        .def_static("remove_ring", &PyMove::remove_ring,
                    py::arg("player_is_black"), py::arg("source"))
        .def_property_readonly("type", &PyMove::type)
        .def_property_readonly("player_is_black", &PyMove::player_is_black)
        .def_property_readonly("source", &PyMove::source)
        .def_property_readonly("destination", &PyMove::destination)
        .def_property_readonly("markers", &PyMove::markers)
        .def("equals", &PyMove::equals);

    m.attr("MOVE_PLACE_RING") = static_cast<int>(kMovePlaceRing);
    m.attr("MOVE_MOVE_RING") = static_cast<int>(kMoveMoveRing);
    m.attr("MOVE_REMOVE_MARKERS") = static_cast<int>(kMoveRemoveMarkers);
    m.attr("MOVE_REMOVE_RING") = static_cast<int>(kMoveRemoveRing);

    m.attr("PHASE_RING_PLACEMENT") = static_cast<int>(kPhaseRingPlacement);
    m.attr("PHASE_MAIN_GAME") = static_cast<int>(kPhaseMainGame);
    m.attr("PHASE_ROW_COMPLETION") = static_cast<int>(kPhaseRowCompletion);
    m.attr("PHASE_RING_REMOVAL") = static_cast<int>(kPhaseRingRemoval);
    m.attr("PHASE_GAME_OVER") = static_cast<int>(kPhaseGameOver);

    m.def("apply_move", &ApplyMovePy,
          "Apply a Move to a State, returning a new State.",
          py::arg("state"), py::arg("move"));
    m.def("get_valid_moves", &GetValidMovesPy,
          "Return all legal moves for a State.",
          py::arg("state"));
    m.def("is_terminal", &IsTerminalPy, py::arg("state"));
    m.def("winner", &WinnerPy,
          "Return 0 (WHITE), 1 (BLACK), or -1 if no score-based winner. "
          "Stalemate detection lives separately.",
          py::arg("state"));

    m.def("bench_random_playouts", &BenchRandomPlayouts,
          "Play `n_games` random YINSH games entirely in C++. Returns "
          "(total_seconds, total_moves).",
          py::arg("n_games"), py::arg("max_moves") = 500,
          py::arg("seed") = 0xC0FFEEull);

    m.def("bench_clone_state", &BenchCloneState,
          "Run `iters` State clones; return total seconds.",
          py::arg("state"), py::arg("iters"));

    py::class_<PyZobrist>(m, "Zobrist")
        .def(py::init<py::array_t<std::uint64_t>,
                      std::uint64_t,
                      py::array_t<std::uint64_t>>(),
             py::arg("piece_keys"),
             py::arg("side_to_move_key"),
             py::arg("phase_keys"))
        .def_property_readonly("empty_hash", &PyZobrist::empty_hash)
        .def_property_readonly("side_to_move_key", &PyZobrist::side_to_move_key)
        .def("phase_key", &PyZobrist::phase_key, py::arg("phase"))
        .def("hash_board", &PyZobrist::hash_board,
             py::arg("wr_lo"), py::arg("wr_hi"),
             py::arg("br_lo"), py::arg("br_hi"),
             py::arg("wm_lo"), py::arg("wm_hi"),
             py::arg("bm_lo"), py::arg("bm_hi"))
        .def("hash_state", &PyZobrist::hash_state,
             py::arg("wr_lo"), py::arg("wr_hi"),
             py::arg("br_lo"), py::arg("br_hi"),
             py::arg("wm_lo"), py::arg("wm_hi"),
             py::arg("bm_lo"), py::arg("bm_hi"),
             py::arg("current_player_is_black"),
             py::arg("phase_idx"))
        .def("toggle", &PyZobrist::toggle,
             py::arg("h"), py::arg("cell"), py::arg("piece"))
        .def("update_position", &PyZobrist::update_position,
             py::arg("h"), py::arg("cell"),
             py::arg("old_piece"), py::arg("new_piece"))
        .def("flip_side", &PyZobrist::flip_side, py::arg("h"));

    m.def("bench_valid_ring_destinations_varied",
          &BenchValidRingDestinationsVaried,
          "Run `iters` ValidRingDestinations calls rotating through a "
          "list of (source, rings, markers) triples. Returns total seconds.",
          py::arg("sources"),
          py::arg("rings_lo"), py::arg("rings_hi"),
          py::arg("markers_lo"), py::arg("markers_hi"),
          py::arg("iters"));
}
