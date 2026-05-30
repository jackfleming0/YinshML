// yngine_driver — line-based stdin/stdout protocol around temhelk/yngine.
//
// The Python bridge in `yinsh_ml.yngine.bridge` spawns this binary, sends
// moves from our side, and asks yngine to pick its own moves at a given
// simulation or time budget. yngine itself has no CLI; this driver is the
// missing "V2b protocol bridge" referenced in VOLUME_PRETRAIN_RESULTS.md.
//
// Protocol (line-oriented, "\n" terminator both directions):
//
//   <- ready                       (on startup, after init)
//   -> new                         reset MCTS + board
//   <- ok
//   -> apply P x y                 place ring at yngine coords (0-10)
//   -> apply M fx fy tx ty dir     ring move, dir in 0..5 (SE NE N NW SW S)
//   -> apply R fx fy dir           remove 5-marker row starting at (fx, fy)
//   -> apply X x y                 remove ring
//   -> apply S                     pass (rare: no legal moves)
//   <- ok | err <reason>
//   -> go sims <N>                 search N iterations, return chosen move
//   -> go time <secs>              search for <secs> seconds, return move
//      (optional 4th token "threads <T>", default 1)
//   <- move <wire-format>          same wire format as `apply` payload
//   -> state                       debug: print next-action / turn / result
//   <- state next=<a> turn=<W|B> result=<...>
//   -> quit
//   <- bye
//
// Coordinates are yngine native (x, y) on a 0-10 square indexing scheme.
// See yngine/bitboard.cpp:Bitboard::coords_to_index. The Python side maps
// to/from our algebraic notation (col=chr('A'+x), row=11-y) — same mapping
// as scripts/yngine_corpus_to_npz.py.
//
// yngine's MCTS prints DEBUG lines unconditionally to std::cout. We redirect
// the C-level stdout to /dev/null on startup and reroute std::cout through
// it, then use a separate FILE* (duped from the original fd 1) for protocol
// replies. That way Python's pipe only sees protocol traffic.

#include <yngine/board_state.hpp>
#include <yngine/bitboard.hpp>
#include <yngine/common.hpp>
#include <yngine/mcts.hpp>
#include <yngine/moves.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unistd.h>

using namespace Yngine;

namespace {

FILE* g_proto = nullptr;   // protocol sink (the original stdout, duped)

void say(const std::string& s) {
    std::fputs(s.c_str(), g_proto);
    std::fputc('\n', g_proto);
    std::fflush(g_proto);
}

const char* direction_name(Direction d) {
    switch (d) {
        case Direction::SE: return "0";
        case Direction::NE: return "1";
        case Direction::N:  return "2";
        case Direction::NW: return "3";
        case Direction::SW: return "4";
        case Direction::S:  return "5";
    }
    return "?";
}

bool parse_direction(int v, Direction& out) {
    if (v < 0 || v > 5) return false;
    out = static_cast<Direction>(v);
    return true;
}

// yngine (x, y) → uint8 index. Validates the cell is on the board.
bool coords_to_index(int x, int y, uint8_t& out) {
    if (x < 0 || x > 10 || y < 0 || y > 10) return false;
    if (!Bitboard::are_coords_in_game(static_cast<uint8_t>(x), static_cast<uint8_t>(y))) {
        return false;
    }
    out = Bitboard::coords_to_index(static_cast<uint8_t>(x), static_cast<uint8_t>(y));
    return true;
}

std::string move_to_wire(const Move& mv) {
    std::ostringstream oss;
    std::visit(variant_overloaded{
        [&](const PlaceRingMove& m) {
            const auto [x, y] = Bitboard::index_to_coords(m.index);
            oss << "P " << int(x) << " " << int(y);
        },
        [&](const RingMove& m) {
            const auto [fx, fy] = Bitboard::index_to_coords(m.from);
            const auto [tx, ty] = Bitboard::index_to_coords(m.to);
            oss << "M " << int(fx) << " " << int(fy) << " "
                        << int(tx) << " " << int(ty) << " "
                        << direction_name(m.direction);
        },
        [&](const RemoveRowMove& m) {
            const auto [fx, fy] = Bitboard::index_to_coords(m.from);
            oss << "R " << int(fx) << " " << int(fy) << " "
                        << direction_name(m.direction);
        },
        [&](const RemoveRingMove& m) {
            const auto [x, y] = Bitboard::index_to_coords(m.index);
            oss << "X " << int(x) << " " << int(y);
        },
        [&](const PassMove&) {
            oss << "S";
        },
    }, mv);
    return oss.str();
}

// Parse the payload of an `apply` line (kind char + rest). Validation is
// deferred to apply_command which checks the move against generate_moves.
std::optional<Move> parse_move(std::istringstream& iss) {
    std::string kind;
    iss >> kind;
    if (kind.empty()) return std::nullopt;
    if (kind == "P") {
        int x, y;
        if (!(iss >> x >> y)) return std::nullopt;
        uint8_t idx;
        if (!coords_to_index(x, y, idx)) return std::nullopt;
        return Move{PlaceRingMove{idx}};
    }
    if (kind == "M") {
        int fx, fy, tx, ty, d;
        if (!(iss >> fx >> fy >> tx >> ty >> d)) return std::nullopt;
        uint8_t fi, ti;
        Direction dir;
        if (!coords_to_index(fx, fy, fi)) return std::nullopt;
        if (!coords_to_index(tx, ty, ti)) return std::nullopt;
        if (!parse_direction(d, dir)) return std::nullopt;
        return Move{RingMove{fi, ti, dir}};
    }
    if (kind == "R") {
        int fx, fy, d;
        if (!(iss >> fx >> fy >> d)) return std::nullopt;
        uint8_t fi;
        Direction dir;
        if (!coords_to_index(fx, fy, fi)) return std::nullopt;
        if (!parse_direction(d, dir)) return std::nullopt;
        return Move{RemoveRowMove{fi, dir}};
    }
    if (kind == "X") {
        int x, y;
        if (!(iss >> x >> y)) return std::nullopt;
        uint8_t idx;
        if (!coords_to_index(x, y, idx)) return std::nullopt;
        return Move{RemoveRingMove{idx}};
    }
    if (kind == "S") {
        return Move{PassMove{}};
    }
    return std::nullopt;
}

// True if a move equal-by-content exists in the legal list at the current
// state. RemoveRowMove::operator== handles direction reversal, so callers
// may send either canonical form of a row removal.
bool move_is_legal(const BoardState& bs, const Move& mv) {
    MoveList legal;
    bs.generate_moves(legal);
    for (std::size_t i = 0; i < legal.get_size(); ++i) {
        if (legal[i] == mv) return true;
    }
    return false;
}

void handle_apply(MCTS& mcts, std::istringstream& iss) {
    auto parsed = parse_move(iss);
    if (!parsed) { say("err parse"); return; }
    const BoardState bs = mcts.get_board();
    if (!move_is_legal(bs, *parsed)) { say("err illegal"); return; }
    mcts.apply_move(*parsed);
    say("ok");
}

void handle_state(const MCTS& mcts) {
    const BoardState bs = mcts.get_board();
    const char* na = "?";
    switch (bs.get_next_action()) {
        case NextAction::RingPlacement: na = "place"; break;
        case NextAction::RingMovement:  na = "move";  break;
        case NextAction::RowRemoval:    na = "row";   break;
        case NextAction::RingRemoval:   na = "ring";  break;
        case NextAction::Done:          na = "done";  break;
    }
    const char* turn = (bs.whose_move() == Color::White) ? "W" : "B";
    const char* res = "ongoing";
    if (bs.get_next_action() == NextAction::Done) {
        switch (bs.game_result()) {
            case GameResult::Draw:     res = "draw";  break;
            case GameResult::WhiteWon: res = "white"; break;
            case GameResult::BlackWon: res = "black"; break;
        }
    }
    std::ostringstream oss;
    oss << "state next=" << na << " turn=" << turn << " result=" << res;
    say(oss.str());
}

void handle_go(MCTS& mcts, std::istringstream& iss) {
    std::string mode;
    iss >> mode;
    int threads = 1;
    MCTS::SearchLimit limit;
    if (mode == "sims") {
        int n;
        if (!(iss >> n) || n <= 0) { say("err sims"); return; }
        limit = n;
    } else if (mode == "time") {
        float secs;
        if (!(iss >> secs) || !(secs > 0.0f)) { say("err time"); return; }
        limit = secs;
    } else {
        say("err go-mode");
        return;
    }
    std::string tok;
    if (iss >> tok && tok == "threads") {
        iss >> threads;
        if (threads < 1) threads = 1;
    }
    // If the position is terminal there's nothing to search. yngine's search
    // path UBs in that case (generates 0 moves then dereferences null).
    if (mcts.get_board().get_next_action() == NextAction::Done) {
        say("err terminal");
        return;
    }
    try {
        Move chosen = mcts.search(limit, threads).get();
        say("move " + move_to_wire(chosen));
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "err search " << e.what();
        say(oss.str());
    }
}

}  // namespace

int main(int /*argc*/, char** /*argv*/) {
    // Steal fd 1 for the protocol channel; redirect std::cout / printf to
    // /dev/null so yngine's debug spam (mcts.cpp:237, 304, 406) never reaches
    // the Python pipe.
    int real_fd = ::dup(STDOUT_FILENO);
    if (real_fd < 0) { std::perror("dup"); return 1; }
    if (!std::freopen("/dev/null", "w", stdout)) { std::perror("freopen"); return 1; }
    g_proto = ::fdopen(real_fd, "w");
    if (!g_proto) { std::perror("fdopen"); return 1; }
    // Line-buffer the protocol channel; Python reads line-by-line.
    ::setvbuf(g_proto, nullptr, _IOLBF, 0);
    // std::cout shares fd 1 by default. After freopen, std::cout writes
    // already land in /dev/null. We still rebuild its sync to be safe.
    std::cout.rdbuf(nullptr);
    std::cout.setstate(std::ios::badbit);

    // 128 MB MCTS pool. yngine's ArenaAllocator mmaps the full pool up
    // front, so the original 512 MB default pinned ~5 GB across 10 sequential
    // eval games (a fresh process per game) and triggered the macOS OOM
    // killer mid-search. 128 MB still fits an MCTS-10K search (the cloud
    // V2a fingerprint run peaked ~50 MB at MCTS-10K) and is well under the
    // per-process pressure threshold on a 16 GB box.
    constexpr std::size_t MEM_LIMIT = 128ull * 1024ull * 1024ull;
    auto make_mcts = [&]() {
        auto m = std::make_unique<MCTS>(MEM_LIMIT);
        // Workaround for an upstream yngine bug: ~MCTS unconditionally calls
        // search_thread.join(), but search_thread is default-constructed (not
        // joinable) until search() runs once. We discard a 1-iteration search
        // to set search_thread to a finished-but-joinable state so the dtor
        // doesn't throw. Cheap (~µs on the empty placement board).
        try { (void)m->search(MCTS::SearchLimit{1}, 1).get(); }
        catch (...) { /* ignore */ }
        return m;
    };
    std::unique_ptr<MCTS> mcts = make_mcts();
    say("ready");

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string cmd;
        if (!(iss >> cmd)) continue;

        if (cmd == "new") {
            // Recreate the MCTS instance: the pool allocator can't be reset
            // in-place without exposing internals.
            mcts.reset();
            mcts = make_mcts();
            say("ok");
        } else if (cmd == "apply") {
            handle_apply(*mcts, iss);
        } else if (cmd == "go") {
            handle_go(*mcts, iss);
        } else if (cmd == "state") {
            handle_state(*mcts);
        } else if (cmd == "ping") {
            say("pong");
        } else if (cmd == "quit" || cmd == "exit") {
            say("bye");
            break;
        } else {
            say("err cmd");
        }
    }

    return 0;
}
