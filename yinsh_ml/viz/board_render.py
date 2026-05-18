"""Hex-board renderer for YINSH game states.

Produces a matplotlib Figure suitable for direct rendering in Streamlit,
Jupyter, or save-to-file workflows. Independent of TensorBoard.

Coordinate system
-----------------
YINSH's three hex axes are vertical (0,+1), horizontal (+1,0), and the
matching-sign diagonal (+1,+1). To make all three render as straight
60°-separated screen lines, we use a monotonic skew transform:

    screen_x = col_idx * (sqrt(3) / 2)
    screen_y = (row - 1) - col_idx * 0.5

so that the +1-col, +1-row diagonal is screen-up-right, the +1-col
horizontal is screen-down-right, and the +1-row vertical is screen-up.

Note: ``yinsh_ml/tracking/yinsh_visualizer.py`` has a separate, older
renderer with a zig-zag offset (``col_idx % 2``) that does not match
YINSH's hex geometry. That module is left untouched here; if/when it's
revisited for the TensorBoard pipeline, the transform in this file is
the correct one.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..game.board import Board
from ..game.constants import (
    HEX_DIRECTIONS,
    Player,
    PieceType,
    Position,
    VALID_POSITIONS,
    is_valid_position,
)


_SQRT3_OVER_2 = math.sqrt(3.0) / 2.0

# Visual style. Kept inline rather than configurable — opinionated defaults
# are more useful than a knob-fest for a viewer module.
_COLORS = {
    "background": "#fbfaf6",
    "grid": "#9aa0a6",
    "position_dot": "#5f6368",
    "white_ring_face": "#ffffff",
    "white_ring_edge": "#202124",
    "black_ring_face": "#202124",
    "black_ring_edge": "#202124",
    "white_marker": "#dadce0",
    "black_marker": "#3c4043",
    "marker_edge": "#202124",
    "last_move_from": "#fbbc04",
    "last_move_to": "#34a853",
    "highlight": "#4285f4",
    "label": "#5f6368",
}


def position_to_xy(col_letter: str, row: int) -> Tuple[float, float]:
    """Map a YINSH (column, row) to skewed-cartesian screen coordinates."""
    col_idx = ord(col_letter) - ord("A")
    x = col_idx * _SQRT3_OVER_2
    y = (row - 1) - col_idx * 0.5
    return x, y


def _position_xy(pos: Position) -> Tuple[float, float]:
    return position_to_xy(pos.column, pos.row)


def _all_valid_positions() -> list[Position]:
    return [
        Position(column=col, row=row)
        for col, rows in VALID_POSITIONS.items()
        for row in rows
    ]


def _draw_grid_lines(ax: Axes) -> None:
    """Draw the three families of hex lines connecting valid positions.

    For each of the three forward hex axes, walk every valid position and
    draw a line segment to its neighbour along that axis if the neighbour
    is also valid. Drawing only forward axes avoids double-drawing each
    segment.
    """
    forward_axes = [(0, 1), (1, 0), (1, 1)]
    for pos in _all_valid_positions():
        x0, y0 = _position_xy(pos)
        for dcol, drow in forward_axes:
            nb = Position(column=chr(ord(pos.column) + dcol), row=pos.row + drow)
            if not is_valid_position(nb):
                continue
            x1, y1 = _position_xy(nb)
            ax.plot(
                [x0, x1], [y0, y1],
                color=_COLORS["grid"], linewidth=0.8, zorder=1,
            )


def _draw_position_dots(ax: Axes, radius: float = 0.05) -> None:
    for pos in _all_valid_positions():
        x, y = _position_xy(pos)
        ax.add_patch(patches.Circle(
            (x, y), radius,
            facecolor=_COLORS["position_dot"], edgecolor="none", zorder=2,
        ))


def _draw_ring(ax: Axes, pos: Position, piece: PieceType, *, scale: float) -> None:
    x, y = _position_xy(pos)
    if piece == PieceType.WHITE_RING:
        face, edge = _COLORS["white_ring_face"], _COLORS["white_ring_edge"]
    else:
        face, edge = _COLORS["black_ring_face"], _COLORS["black_ring_edge"]
    outer_r = 0.34 * scale
    width = 0.10 * scale
    ax.add_patch(patches.Annulus(
        (x, y), outer_r, width,
        facecolor=face, edgecolor=edge, linewidth=1.2, zorder=5,
    ))


def _draw_marker(ax: Axes, pos: Position, piece: PieceType, *, scale: float) -> None:
    x, y = _position_xy(pos)
    face = (_COLORS["white_marker"]
            if piece == PieceType.WHITE_MARKER else _COLORS["black_marker"])
    ax.add_patch(patches.Circle(
        (x, y), 0.18 * scale,
        facecolor=face, edgecolor=_COLORS["marker_edge"], linewidth=0.8, zorder=4,
    ))


def _draw_position_labels(ax: Axes) -> None:
    for pos in _all_valid_positions():
        x, y = _position_xy(pos)
        ax.text(
            x + 0.18, y - 0.18, f"{pos.column}{pos.row}",
            fontsize=6, color=_COLORS["label"], zorder=3,
            ha="left", va="top",
        )


def _draw_last_move(ax: Axes, frm: Position, to: Position) -> None:
    x0, y0 = _position_xy(frm)
    x1, y1 = _position_xy(to)
    ax.add_patch(patches.Circle(
        (x0, y0), 0.40,
        facecolor="none", edgecolor=_COLORS["last_move_from"],
        linewidth=2.0, zorder=6,
    ))
    ax.add_patch(patches.Circle(
        (x1, y1), 0.40,
        facecolor="none", edgecolor=_COLORS["last_move_to"],
        linewidth=2.0, zorder=6,
    ))
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>", color=_COLORS["last_move_to"],
            lw=1.5, alpha=0.6, shrinkA=14, shrinkB=14,
        ),
        zorder=6,
    )


def _draw_highlights(ax: Axes, positions: Iterable[Position]) -> None:
    for pos in positions:
        x, y = _position_xy(pos)
        ax.add_patch(patches.Circle(
            (x, y), 0.32,
            facecolor="none", edgecolor=_COLORS["highlight"],
            linewidth=1.6, linestyle="--", zorder=6,
        ))


def render_board(
    board: Board,
    *,
    last_move: Optional[Tuple[Position, Position]] = None,
    highlight: Optional[Iterable[Position]] = None,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8.0, 8.0),
    show_coords: bool = True,
) -> Figure:
    """Render a YINSH board state.

    Parameters
    ----------
    board:
        The ``Board`` to render. Pieces are read via ``board.get_piece(pos)``.
    last_move:
        Optional (from, to) Position pair, highlighted with from/to circles
        and an arrow.
    highlight:
        Optional positions to mark with a dashed circle (e.g. valid moves
        for the side to move, or positions involved in a captured row).
    title:
        Optional figure title.
    ax:
        Optional existing Axes to draw into. If ``None``, a new Figure is
        created with the given ``figsize``.
    figsize:
        Figure size if ``ax`` is None.
    show_coords:
        If True, draws small algebraic position labels near each intersection.

    Returns
    -------
    The matplotlib ``Figure`` containing the rendered board. If ``ax`` was
    provided, returns ``ax.figure``.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.set_facecolor(_COLORS["background"])
    if fig is not None:
        fig.patch.set_facecolor(_COLORS["background"])

    _draw_grid_lines(ax)
    _draw_position_dots(ax)
    if show_coords:
        _draw_position_labels(ax)

    scale = 1.0
    for pos in _all_valid_positions():
        piece = board.get_piece(pos)
        if piece is None:
            continue
        if piece.is_ring() if hasattr(piece, "is_ring") else piece in (
            PieceType.WHITE_RING, PieceType.BLACK_RING
        ):
            _draw_ring(ax, pos, piece, scale=scale)
        else:
            _draw_marker(ax, pos, piece, scale=scale)

    if last_move is not None:
        _draw_last_move(ax, last_move[0], last_move[1])
    if highlight is not None:
        _draw_highlights(ax, highlight)

    xs = [position_to_xy(c, r)[0]
          for c, rows in VALID_POSITIONS.items() for r in rows]
    ys = [position_to_xy(c, r)[1]
          for c, rows in VALID_POSITIONS.items() for r in rows]
    pad = 0.8
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=12, color=_COLORS["label"], pad=10)

    return fig


__all__ = ["render_board", "position_to_xy"]
