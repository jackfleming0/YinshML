---
description: English summary of the YINSH board game rules for quick reference
globs: YINSH_RULES.md
alwaysApply: true
---

# YINSH Game Rules

This document summarizes the core rules of the **YINSH** board game so you can understand the game logic used throughout the YinshML codebase. It assumes the standard 11×11 hexagon-shaped board and the traditional ring/marker interaction.

## Game Setup

- **Board geometry**: A 11×11 hexagonal board has non-rectangular columns (A–K) with 99 playable intersections. Each column has a restricted range of rows (for example, column `D` contains rows 1 through 9). Only positions defined in [`yinsh_ml/game/constants.py`](mdc:yinsh_ml/game/constants.py) are legal.
- **Pieces**:
  - **Rings**: Each player has five rings (`R` for White, `r` for Black). Rings are the only pieces that move.
  - **Markers**: When a ring moves, it leaves a marker (`M`/`m`) on each square it traverses. These markers are used to capture lines.
- **Objective**: The first player to remove three rows of five markers wins. In blitz-style games (used in some experiments), capturing a single row is enough to win.

## Turn Phases

1. **Ring Placement** (also called the starting phase):
   - Players alternate placing rings on empty board intersections until both have placed all five.
   - Rings can only be placed on valid positions; no markers exist yet.
2. **Main Game (Movement Phase)**:
   - On a turn, a player picks one of their rings and moves it along one of the six hex directions (`HEX_DIRECTIONS` in the constants file).
   - Rings slide in a straight line over empty cells, and each traversed cell receives that player's marker. The ring cannot jump over other rings.
   - Moves are limited in distance: a ring may travel at most five spaces in one move (see `MAX_RING_MOVE_DISTANCE`). The ring always leaves markers behind.
3. **Row Capture**:
   - As soon as five consecutive markers of the same color line up along a straight direction, the owning player removes those markers and one of their rings from the board. The removed ring can later be re-used to increase remaining pieces as long as additional rings remain on the board.
   - Removing a row may open up new capture opportunities on the same turn, but you must continue moving other rings until capture opportunities are exhausted.
4. **Victory Condition**:
   - The first player to capture **three** rows wins (`POINTS_TO_WIN = 3`), unless the game configuration is set to blitz (`POINTS_TO_WIN_BLITZ = 1`).

## Movement and Capture Details

- **Valid Directions**: Moves follow the board’s axial system: `(0,1)`, `(1,0)`, `(1,1)`, `(−1,1)` (defined as `DIRECTIONS`). These correspond to the vertical and two diagonal lines of the hex board.
- **Marker Tracking**: The board maintains markers for all moves. Capturing five markers removes them permanently and resets the associated ring. Some implementations (like `yinsh_ml/heuristics`) also consider runs of up to seven markers (`MAX_MARKER_SEQUENCE = 7`).
- **Ring Constraints**: Rings may only move across empty cells. The destination must be empty, and the ring cannot stop in the middle of another marker sequence unless the path is clear.

## Strategy Implication for YinshML

- **Markers encode board control**: The self-play agents and evaluations treat markers as positional advantages, so capturing and avoiding opponent rows is key.
- **Ring efficiency matters**: Since only rings move, preserving all five rings while generating capture opportunities is the central trade-off reflected in the neural network inputs and heuristics.
- **Row removal resets tempo**: Each captured row removes markers and a ring, which changes both the playable area and the number of available ring moves.

For a deeper dive, the YinshML codebase mirrors these rules across `yinsh_ml/game/board.py`, `game_state.py`, and the `rules` referenced in `yins_ml/search/README.md`.
