// yinsh-driver — line-based stdin/stdout protocol around sharkdp/yinsh's
// negamax AI (crate `yinsh_ai`). Deliberately speaks the *same* wire protocol
// as `third_party/yngine_driver` so the existing Python bridge & move codec
// (`yinsh_ml.yngine.move_codec`) are reused verbatim. The YinshML side spawns
// this binary, feeds it our moves, and asks it to pick sharkdp's own moves.
//
// Protocol (line-oriented, "\n" terminator both directions):
//
//   <- ready                       (on startup)
//   -> new                         reset board to the initial position
//   <- ok
//   -> apply P x y                 place ring
//   -> apply M fx fy tx ty dir     ring move (marker placed at from, ring slides to)
//   -> apply R fx fy dir           remove a 5-marker run (fx,fy = one end, dir = line dir)
//   -> apply X x y                 remove ring
//   -> apply S                     pass (no legal move — rare)
//   <- ok | err <reason>
//   -> go [depth N]                search and return sharkdp's chosen move
//   <- move <wire-format>          same wire format as the `apply` payload
//   -> depth N                     set the default search depth (persists)
//   -> state                       debug: print turn mode + active player
//   <- state mode=<…> turn=<A|B> points=<a>:<b>
//   -> quit
//   <- bye
//
// Coordinate systems
// ------------------
// The wire uses yngine's (x, y) on a 0..10 square index. sharkdp uses hex
// Coord{x,y} on -5..5. The exact bijection (verified against both engines'
// 85-point boards) is:
//
//     x_wire = x_shark + 5      x_shark = x_wire - 5
//     y_wire = 5 - y_shark      y_shark = 5 - y_wire
//
// It is a hex symmetry, so it preserves lines, ring slides and 5-in-a-row
// runs — exactly what cross-engine play needs. Direction codes map per
// DIR_SHARK_TO_WIRE below (derived from the same bijection: a shark delta
// (dx,dy) becomes a wire delta (dx,-dy)).

use std::io::{self, BufRead, Write};

use yinsh::{Coord, GameState, Move, Player, TurnMode};
use yinsh_ai::{SimpleHeuristic, YinshAi, YinshAiPlayer, possible_moves};

// ---- coordinate conversion -------------------------------------------------

fn shark_to_wire(c: Coord) -> (i32, i32) {
    (c.x as i32 + 5, 5 - c.y as i32)
}

fn wire_to_shark(x: i32, y: i32) -> Coord {
    Coord::new((x - 5) as i8, (5 - y) as i8)
}

/// shark unit step (dx, dy) -> wire direction code 0..5. Panics on a non-unit
/// or non-hex step (a bug, not a protocol error).
fn shark_step_to_wire_dir(dx: i8, dy: i8) -> i32 {
    match (dx, dy) {
        (1, 0) => 0,   // S  -> SE
        (0, -1) => 1,  // SE -> NE
        (-1, -1) => 2, // NE -> N
        (-1, 0) => 3,  // N  -> NW
        (0, 1) => 4,   // NW -> SW
        (1, 1) => 5,   // SW -> S
        _ => panic!("non-hex shark step ({dx}, {dy})"),
    }
}

/// wire direction code 0..5 -> shark unit step (dx, dy).
fn wire_dir_to_shark_step(dir: i32) -> (i8, i8) {
    match dir {
        0 => (1, 0),
        1 => (0, -1),
        2 => (-1, -1),
        3 => (-1, 0),
        4 => (0, 1),
        5 => (1, 1),
        _ => panic!("bad wire direction {dir}"),
    }
}

fn unit_step(from: Coord, to: Coord) -> (i8, i8) {
    let dx = (to.x - from.x).clamp(-1, 1);
    let dy = (to.y - from.y).clamp(-1, 1);
    (dx, dy)
}

// ---- wire encoding of a sharkdp Move ---------------------------------------

fn wire_place_ring(c: Coord) -> String {
    let (x, y) = shark_to_wire(c);
    format!("P {x} {y}")
}

fn wire_remove_ring(c: Coord) -> String {
    let (x, y) = shark_to_wire(c);
    format!("X {x} {y}")
}

fn wire_move_ring(from: Coord, to: Coord) -> String {
    let (fx, fy) = shark_to_wire(from);
    let (tx, ty) = shark_to_wire(to);
    let (dx, dy) = unit_step(from, to);
    let dir = shark_step_to_wire_dir(dx, dy);
    format!("M {fx} {fy} {tx} {ty} {dir}")
}

/// Encode a run removal. `state` must still hold the run (call before applying).
fn wire_remove_run(state: &GameState, seed: Coord) -> Result<String, String> {
    let coords = state
        .board
        .run_coords_from(seed)
        .ok_or_else(|| format!("seed {seed:?} is not part of a run"))?;
    // The run is 5 colinear markers. Find an endpoint and the line direction.
    // An endpoint is a coord whose predecessor (coord - step) is not in the run.
    for &cand in &coords {
        for dir in 0..6 {
            let (sx, sy) = wire_dir_to_shark_step(dir);
            let step = Coord::new(sx, sy);
            let prev = cand - step;
            let walk: Vec<Coord> = (0..5).map(|i| cand + step * (i as i8)).collect();
            let mut all_present = true;
            for w in &walk {
                if !coords.contains(w) {
                    all_present = false;
                    break;
                }
            }
            if all_present && !coords.contains(&prev) {
                let (fx, fy) = shark_to_wire(cand);
                return Ok(format!("R {fx} {fy} {dir}"));
            }
        }
    }
    Err(format!("could not orient run from seed {seed:?}: {coords:?}"))
}

// ---- turn-mode helpers -----------------------------------------------------

fn is_wait_mode(m: &TurnMode) -> bool {
    matches!(
        m,
        TurnMode::WaitForRingMovement(_)
            | TurnMode::WaitForRunRemoval(_)
            | TurnMode::WaitForRingRemoval(_)
            | TurnMode::WaitForMarkerPlacement
    )
}

/// sharkdp splits a logical turn with structural `Wait` plies (one legal move).
/// Drain them so the state sits in an actionable mode before we read/write it.
fn drain_waits(state: &mut GameState) {
    while is_wait_mode(&state.turn_mode) {
        state.perform_move(&Move::Wait);
    }
}

fn mode_name(m: &TurnMode) -> &'static str {
    match m {
        TurnMode::RingPlacement => "ring_placement",
        TurnMode::MarkerPlacement => "marker_placement",
        TurnMode::WaitForRingMovement(_) => "wait_ring_movement",
        TurnMode::RingMovement(_) => "ring_movement",
        TurnMode::RunRemoval(_) => "run_removal",
        TurnMode::WaitForRunRemoval(_) => "wait_run_removal",
        TurnMode::RingRemoval(_) => "ring_removal",
        TurnMode::WaitForRingRemoval(_) => "wait_ring_removal",
        TurnMode::WaitForMarkerPlacement => "wait_marker_placement",
    }
}

// ---- apply an incoming wire move to the real state -------------------------

fn apply_wire(state: &mut GameState, line: &str) -> Result<(), String> {
    let mut it = line.split_whitespace();
    let kind = it.next().ok_or("empty move")?;

    drain_waits(state);
    let player = state.active_player;

    match kind {
        "P" => {
            let c = parse_coord(&mut it)?;
            if state.board.free_coords().all(|f| f != c) {
                return Err(format!("P {c:?} is not a free field"));
            }
            if !matches!(state.turn_mode, TurnMode::RingPlacement) {
                return Err(format!("P in mode {}", mode_name(&state.turn_mode)));
            }
            state.perform_move(&Move::PlaceRing(c));
        }
        "M" => {
            let from = parse_coord(&mut it)?;
            let to = parse_coord(&mut it)?;
            // dir token is redundant (we have both endpoints) — consume & ignore.
            let _ = it.next();
            if !matches!(state.turn_mode, TurnMode::MarkerPlacement) {
                return Err(format!("M in mode {}", mode_name(&state.turn_mode)));
            }
            if state.board.marker_moves(player).all(|c| c != from) {
                return Err(format!("M cannot start a marker/ring move at {from:?}"));
            }
            state.perform_move(&Move::PlaceMarker(from));
            drain_waits(state); // -> RingMovement(from)
            if !state.board.ring_moves(from).contains(&to) {
                return Err(format!("M illegal ring slide {from:?} -> {to:?}"));
            }
            state.perform_move(&Move::MoveRing(from, to));
        }
        "R" => {
            let end = parse_coord(&mut it)?;
            let dir: i32 = it
                .next()
                .ok_or("R missing dir")?
                .parse()
                .map_err(|_| "R bad dir")?;
            if !matches!(state.turn_mode, TurnMode::RunRemoval(_)) {
                return Err(format!("R in mode {}", mode_name(&state.turn_mode)));
            }
            let (sx, sy) = wire_dir_to_shark_step(dir);
            let step = Coord::new(sx, sy);
            let want: Vec<Coord> = (0..5).map(|i| end + step * (i as i8)).collect();
            let seed = find_seed_for_run(state, player, &want)
                .ok_or_else(|| format!("R no matching run for {want:?}"))?;
            state.perform_move(&Move::RemoveRun(seed));
        }
        "X" => {
            let c = parse_coord(&mut it)?;
            if !matches!(state.turn_mode, TurnMode::RingRemoval(_)) {
                return Err(format!("X in mode {}", mode_name(&state.turn_mode)));
            }
            if state.board.ring_coords(player).all(|r| r != c) {
                return Err(format!("X {c:?} is not a ring of the active player"));
            }
            state.perform_move(&Move::RemoveRing(c));
        }
        "S" => {
            // Pass — sharkdp has no real pass; nothing to do.
        }
        other => return Err(format!("unknown move kind {other:?}")),
    }
    Ok(())
}

fn parse_coord<'a>(it: &mut impl Iterator<Item = &'a str>) -> Result<Coord, String> {
    let x: i32 = it.next().ok_or("missing x")?.parse().map_err(|_| "bad x")?;
    let y: i32 = it.next().ok_or("missing y")?.parse().map_err(|_| "bad y")?;
    Ok(wire_to_shark(x, y))
}

/// Map a 5-coord run (in shark coords) back to a canonical run seed.
fn find_seed_for_run(state: &GameState, player: Player, want: &[Coord]) -> Option<Coord> {
    let mut want_sorted = want.to_vec();
    want_sorted.sort_by_key(|c| (c.x, c.y));
    for seed in state.board.run_seeds(player) {
        if let Some(mut coords) = state.board.run_coords_from(seed) {
            coords.sort_by_key(|c| (c.x, c.y));
            if coords == want_sorted {
                return Some(seed);
            }
        }
    }
    None
}

// ---- compute sharkdp's own move (no mutation of the real state) ------------

/// Pick a move for the current (actionable) position. Returns `None` only when
/// there is genuinely no legal move. sharkdp's `Negamax::choose_move` unwraps to
/// nothing once `winner()` is set (the point is scored at `RemoveRun`, before the
/// follow-up ring removal), so on an already-decided position we just take the
/// first legal move to let the final removal flow through instead of panicking.
fn pick(ai: &YinshAi<SimpleHeuristic>, s: &GameState) -> Option<Move> {
    if possible_moves(s).next().is_none() {
        return None;
    }
    if s.winner().is_some() {
        return possible_moves(s).next();
    }
    Some(ai.choose_move(s))
}

fn compute_move(state: &GameState, depth: usize) -> Option<String> {
    let ai = YinshAi::new(SimpleHeuristic::default(), depth);
    let mut s = state.clone();
    loop {
        drain_waits(&mut s);
        let mv = match pick(&ai, &s) {
            Some(m) => m,
            None => return Some("S".to_string()),
        };
        match mv {
            Move::Wait => s.perform_move(&Move::Wait),
            Move::PlaceRing(c) => return Some(wire_place_ring(c)),
            Move::RemoveRing(c) => return Some(wire_remove_ring(c)),
            Move::RemoveRun(seed) => return wire_remove_run(&s, seed).ok(),
            Move::MoveRing(a, b) => return Some(wire_move_ring(a, b)),
            Move::PlaceMarker(from) => {
                // A logical ring move = PlaceMarker + (Wait) + MoveRing. Combine
                // both sharkdp plies into one wire `M`.
                s.perform_move(&Move::PlaceMarker(from));
                loop {
                    drain_waits(&mut s);
                    let m2 = match pick(&ai, &s) {
                        Some(m) => m,
                        None => {
                            eprintln!(
                                "no moves after PlaceMarker({from:?}) in mode {}",
                                mode_name(&s.turn_mode)
                            );
                            return Some("S".to_string());
                        }
                    };
                    match m2 {
                        Move::Wait => s.perform_move(&Move::Wait),
                        Move::MoveRing(a, b) => return Some(wire_move_ring(a, b)),
                        other => {
                            eprintln!("unexpected move after PlaceMarker: {other:?}");
                            return None;
                        }
                    }
                }
            }
        }
    }
}

// ---- main loop -------------------------------------------------------------

fn main() {
    let mut default_depth: usize = std::env::var("SHARKDP_DEPTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    let mut state = GameState::initial();

    let say = |out: &mut io::StdoutLock, s: &str| {
        let _ = writeln!(out, "{s}");
        let _ = out.flush();
    };

    say(&mut out, "ready");

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut head = line.split_whitespace();
        let cmd = head.next().unwrap_or("");
        match cmd {
            "new" => {
                state = GameState::initial();
                say(&mut out, "ok");
            }
            "apply" => {
                let rest = line.strip_prefix("apply").unwrap_or("").trim();
                match apply_wire(&mut state, rest) {
                    Ok(()) => say(&mut out, "ok"),
                    Err(e) => say(&mut out, &format!("err {e}")),
                }
            }
            "go" => {
                let mut depth = default_depth;
                // optional "depth N"
                let toks: Vec<&str> = head.collect();
                if toks.len() >= 2 && toks[0] == "depth" {
                    if let Ok(d) = toks[1].parse::<usize>() {
                        depth = d;
                    }
                }
                match compute_move(&state, depth) {
                    Some(w) => say(&mut out, &format!("move {w}")),
                    None => say(&mut out, "err search failed"),
                }
            }
            "depth" => {
                if let Some(d) = head.next().and_then(|s| s.parse::<usize>().ok()) {
                    default_depth = d;
                    say(&mut out, "ok");
                } else {
                    say(&mut out, "err bad depth");
                }
            }
            "state" => {
                let turn = match state.active_player {
                    Player::A => "A",
                    Player::B => "B",
                };
                say(
                    &mut out,
                    &format!(
                        "state mode={} turn={} points={}:{}",
                        mode_name(&state.turn_mode),
                        turn,
                        state.points_a,
                        state.points_b
                    ),
                );
            }
            "quit" => {
                say(&mut out, "bye");
                break;
            }
            other => say(&mut out, &format!("err unknown command {other:?}")),
        }
    }
}
