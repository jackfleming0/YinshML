// YINSH analysis board — drag-drop position composer + network evaluation.
//
// Rendering is delegated to ./board3d.js (three.js). app.js owns logical
// state, API calls, and DOM glue; board3d.js owns the 3D scene and the
// pixel↔board-position conversion via raycasting.
//
// Hex geometry is ported from yinsh_ml/viz/board_render.py:
//   screen_x = col_idx * sqrt(3)/2
//   screen_y = (row - 1) - col_idx * 0.5

import {
  initBoard3D,
  setPieces,
  setHover,
  setSelection,
  setArrow,
  clearArrow,
  pixelToBoardPos,
  resetView,
  animateMove,
  setCapturedRings,
  setSelectableHighlights,
} from "./board3d.js";

// ---------- Constants ----------
const COLUMNS = "ABCDEFGHIJK".split("");
const SQRT3_2 = Math.sqrt(3) / 2;

// VALID_POSITIONS — mirrors yinsh_ml/game/constants.py.
const VALID_POSITIONS = {
  A: [2, 3, 4, 5],
  B: [1, 2, 3, 4, 5, 6, 7],
  C: [1, 2, 3, 4, 5, 6, 7, 8],
  D: [1, 2, 3, 4, 5, 6, 7, 8, 9],
  E: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  F: [2, 3, 4, 5, 6, 7, 8, 9, 10],
  G: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
  H: [3, 4, 5, 6, 7, 8, 9, 10, 11],
  I: [4, 5, 6, 7, 8, 9, 10, 11],
  J: [5, 6, 7, 8, 9, 10, 11],
  K: [7, 8, 9, 10],
};

const ALL_POSITIONS = [];
for (const col of COLUMNS) {
  for (const row of VALID_POSITIONS[col]) ALL_POSITIONS.push({ col, row });
}

function posKey(col, row) { return `${col}${row}`; }
function isValidPos(col, row) {
  const rows = VALID_POSITIONS[col];
  return rows ? rows.includes(row) : false;
}

// Generate a UUID. Browsers ≥2022 have crypto.randomUUID natively; the
// fallback covers older runtimes. Used to tag every /api/move with a
// `play_session_id` so the server log can be grouped into games offline.
function _uuid() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) return crypto.randomUUID();
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
}

// Geometry is owned by board3d.js — see posToWorld / pixelToBoardPos there.

// ---------- State ----------
const state = {
  pieces: new Map(),         // posKey -> "WHITE_RING" | "BLACK_RING" | "WHITE_MARKER" | "BLACK_MARKER"
  armedTool: null,           // currently selected palette tool
  hoverPos: null,            // {col,row} or null — for drop preview
  hoverArrow: null,          // {from:{col,row}, to:{col,row}} | null — for top-move hover
  lastResult: null,          // last /api/evaluate response
  models: [],
  mode: "play",              // "setup" | "play" — Play is the default landing
  opponent: "engine",        // "self" | "engine" — Engine is the default opponent
  legalMoves: [],            // list of move dicts (from /api/evaluate or /api/move response)
  selectedSource: null,      // {col,row} of the ring picked in Play mode, awaiting destination
  history: [],               // stack of {position, move_description}
  autoEval: true,            // whether to auto-evaluate after each move
  busy: false,               // true while a move is in-flight (blocks double-clicks)
  gameOver: false,
  winner: null,
  moveMaker: null,           // engine's _move_maker, threaded through /api/move to keep
                             // capture sequences correct across the stateless server's
                             // GameState rebuilds. Null outside of capture sequences.
  lineMode: null,            // null when not walking a line. When active:
                             //   { pvIndex, topMoveDesc, baseSnapshot, steps, currentStep }
                             // Setup/Play interactions are disabled while non-null.
  game: null,                // null unless opponent === "engine". When active:
                             //   { phase: "setup"|"playing"|"ended",
                             //     humanSide, computerSide, computerSims,
                             //     spoilersEnabled, thinking, winner }
  review: null,              // null outside Review mode. When a game is loaded:
                             //   { tableId, metadata, steps, currentStep,
                             //     baseSnapshot } — baseSnapshot stores the
                             //   non-review state we restore on exit so a
                             //   composed Setup position survives the trip.
  playSessionId: _uuid(),    // sent with every /api/move so server-side logs
                             // can be reconstructed into discrete games offline.
                             // Regenerated on Clear / startGame / setup→play.
};


// ---------- 3D rendering glue ----------
const boardContainer = document.getElementById("board-3d");
const { domElement: boardEl } = initBoard3D(boardContainer);

const resetViewBtn = document.getElementById("reset-view-btn");
if (resetViewBtn) resetViewBtn.addEventListener("click", () => resetView());

function render() {
  setPieces(state.pieces);
  if (state.hoverPos) {
    setHover(state.hoverPos.col, state.hoverPos.row);
  } else {
    setHover(null, null);
  }
  if (state.selectedSource) {
    setSelection(state.selectedSource.col, state.selectedSource.row);
  } else {
    setSelection(null, null);
  }
  if (state.hoverArrow) {
    // moveToArrow yields {from: {col,row}, to: {col,row}|null}; setArrow
    // takes "A1"-style position strings. Convert here so board3d stays
    // protocol-agnostic.
    const { from, to } = state.hoverArrow;
    const fromStr = from ? `${from.col}${from.row}` : null;
    const toStr = to ? `${to.col}${to.row}` : null;
    setArrow(fromStr, toStr);
  } else {
    clearArrow();
  }
  // Capture-phase highlights — player picks by sight, not by reading the
  // coordinate label in the analysis panel.
  //   - RING_REMOVAL: each removable ring of the side-to-move gets a red halo.
  //   - ROW_COMPLETION: every marker that's part of ANY candidate row gets a
  //     red halo. The analysis panel's hover-arrow still disambiguates which
  //     specific row is selected when multiple candidates exist.
  if (phaseSel.value === "RING_REMOVAL" && state.legalMoves.length > 0) {
    const removable = state.legalMoves
      .filter((m) => m.type === "REMOVE_RING" && m.source)
      .map((m) => m.source);
    setSelectableHighlights(removable, "ring");
  } else if (phaseSel.value === "ROW_COMPLETION" && state.legalMoves.length > 0) {
    // Union of marker positions across all candidate rows. Set dedupes
    // overlap between candidates (markers shared by multiple rows show
    // up once, not N times).
    const markerSet = new Set();
    for (const mv of state.legalMoves) {
      if (mv.type === "REMOVE_MARKERS" && Array.isArray(mv.markers)) {
        for (const p of mv.markers) markerSet.add(p);
      }
    }
    setSelectableHighlights(Array.from(markerSet), "marker");
  } else {
    setSelectableHighlights([], "ring");
  }
}

// ---------- Drag-drop ----------
let drag = null;   // { tool, ghostEl } | null
const palette = document.getElementById("palette");

function startDrag(tool, clientX, clientY) {
  if (drag) endDrag();
  const ghostEl = document.createElement("div");
  ghostEl.className = "ghost";
  ghostEl.innerHTML = ghostMarkup(tool);
  ghostEl.style.left = `${clientX}px`;
  ghostEl.style.top = `${clientY}px`;
  document.body.appendChild(ghostEl);
  drag = { tool, ghostEl };
  document.body.style.cursor = "grabbing";
}

function endDrag() {
  if (!drag) return;
  drag.ghostEl.remove();
  drag = null;
  document.body.style.cursor = "";
}

function ghostMarkup(tool) {
  const map = {
    WHITE_RING: '<svg viewBox="-12 -12 24 24"><circle r="9" fill="#fafafa" stroke="#161616" stroke-width="1.4"/><circle r="5.2" fill="#dcdcdc" stroke="#161616" stroke-width="1.0"/></svg>',
    BLACK_RING: '<svg viewBox="-12 -12 24 24"><circle r="9" fill="#1f1f1f" stroke="#070707" stroke-width="1.4"/><circle r="5.2" fill="#cfd2d6" stroke="#070707" stroke-width="1.0"/></svg>',
    WHITE_MARKER: '<svg viewBox="-12 -12 24 24"><circle r="7" fill="#fafafa" stroke="#161616" stroke-width="1.2"/></svg>',
    BLACK_MARKER: '<svg viewBox="-12 -12 24 24"><circle r="7" fill="#1f1f1f" stroke="#070707" stroke-width="1.2"/></svg>',
    ERASE: '<div style="display:flex;align-items:center;justify-content:center;width:100%;height:100%;font-size:28px;color:#b91c1c;">⌫</div>',
  };
  return map[tool] || "";
}

palette.addEventListener("mousedown", (ev) => {
  if (state.mode !== "setup") return;   // palette disabled in play mode
  if (state.lineMode) return;            // and in line mode
  const btn = ev.target.closest(".tool");
  if (!btn) return;
  ev.preventDefault();
  const tool = btn.dataset.tool;
  setArmedTool(tool);
  startDrag(tool, ev.clientX, ev.clientY);
});

function setArmedTool(tool) {
  state.armedTool = tool;
  for (const el of palette.querySelectorAll(".tool")) {
    el.classList.toggle("armed", el.dataset.tool === tool);
  }
}

document.addEventListener("mousemove", (ev) => {
  if (!drag) return;
  drag.ghostEl.style.left = `${ev.clientX}px`;
  drag.ghostEl.style.top = `${ev.clientY}px`;
});

document.addEventListener("mouseup", (ev) => {
  if (!drag) return;
  // Was the drop over the board element?
  const rect = boardEl.getBoundingClientRect();
  if (ev.clientX >= rect.left && ev.clientX <= rect.right &&
      ev.clientY >= rect.top && ev.clientY <= rect.bottom) {
    handlePlace(ev.clientX, ev.clientY, drag.tool);
  }
  endDrag();
});

// Dwell tooltip — shows the position name (e.g. "H5") after the mouse
// stays over the same intersection for ~350ms. Visual confirmation of the
// hovered cell without tracing the grid lines.
const boardTooltip = document.createElement("div");
boardTooltip.className = "board-tooltip";
boardTooltip.setAttribute("data-visible", "false");
document.body.appendChild(boardTooltip);
let _tooltipPos = null;       // "E5" string of currently-tracked position
let _tooltipTimer = null;
let _tooltipLastEv = null;    // last mousemove event, for positioning at fire time
const TOOLTIP_DWELL_MS = 350;

function _hideTooltip() {
  if (_tooltipTimer) { clearTimeout(_tooltipTimer); _tooltipTimer = null; }
  boardTooltip.setAttribute("data-visible", "false");
  _tooltipPos = null;
}

function _scheduleTooltip(ev, posStr) {
  if (_tooltipPos === posStr) {
    // Same intersection — keep current timer or visible state, just track cursor
    _tooltipLastEv = ev;
    if (boardTooltip.getAttribute("data-visible") === "true") {
      boardTooltip.style.left = `${ev.clientX}px`;
      boardTooltip.style.top = `${ev.clientY}px`;
    }
    return;
  }
  // New intersection — reset
  if (_tooltipTimer) clearTimeout(_tooltipTimer);
  boardTooltip.setAttribute("data-visible", "false");
  _tooltipPos = posStr;
  _tooltipLastEv = ev;
  _tooltipTimer = setTimeout(() => {
    if (_tooltipPos !== posStr || !_tooltipLastEv) return;
    boardTooltip.textContent = posStr;
    boardTooltip.style.left = `${_tooltipLastEv.clientX}px`;
    boardTooltip.style.top = `${_tooltipLastEv.clientY}px`;
    boardTooltip.setAttribute("data-visible", "true");
  }, TOOLTIP_DWELL_MS);
}

// Board pointer: hover preview (setup) / select-and-move (play) + dwell tooltip.
boardEl.addEventListener("mousemove", (ev) => {
  const near = pixelToBoardPos(ev.clientX, ev.clientY);
  if (state.mode === "setup") {
    state.hoverPos = (state.armedTool && near) ? { col: near.col, row: near.row } : null;
  } else {
    state.hoverPos = null;  // Play mode: no drop-cursor preview
  }
  if (near) _scheduleTooltip(ev, `${near.col}${near.row}`);
  else _hideTooltip();
  render();
});

boardEl.addEventListener("mouseleave", () => {
  state.hoverPos = null;
  _hideTooltip();
  render();
});

boardEl.addEventListener("click", (ev) => {
  // While walking a PV the board is display-only — clicks are ignored.
  if (state.lineMode) return;
  if (state.mode === "setup") {
    if (!state.armedTool) return;
    handlePlace(ev.clientX, ev.clientY, state.armedTool);
    return;
  }
  handlePlayClick(ev.clientX, ev.clientY);
});

boardEl.addEventListener("contextmenu", (ev) => {
  ev.preventDefault();
  if (state.mode !== "setup") return;   // no erase in play mode
  const near = pixelToBoardPos(ev.clientX, ev.clientY);
  if (near) {
    state.pieces.delete(posKey(near.col, near.row));
    state.hoverArrow = null;
    state.lastResult = null;
    updateDerivedStats();
    render();
  }
});

function legalDestinationsFrom(src) {
  const srcStr = `${src.col}${src.row}`;
  const dests = [];
  for (const mv of state.legalMoves) {
    if (mv.type === "MOVE_RING" && mv.source === srcStr && mv.destination) {
      dests.push(parsePos(mv.destination));
    }
  }
  return dests;
}

function handlePlayClick(px, py) {
  if (state.busy) return;
  // Play+Engine: only the human plays via clicks. Engine moves are
  // auto-applied by computerMakeMove(). Block any click while it's not
  // the human's turn so the user can't accidentally play for the engine.
  // Also block clicks entirely if Opponent=Engine but no game is active —
  // the user is sitting in the NEW GAME stepper and shouldn't be able to
  // play moves on the board until they hit Start game.
  if (state.opponent === "engine") {
    if (!state.game || state.game.phase !== "playing") {
      setStatus("Click 'Start game' on the right to begin a vs-Engine game.", null);
      return;
    }
    // currentSide() reads the radio (kept in sync by applyNewPosition);
    // state.lastResult.side_to_move goes stale between turns when spoilers
    // are off, so prefer the radio.
    if (currentSide() !== state.game.humanSide) return;
  }
  const near = pixelToBoardPos(px, py);
  if (!near) {
    // Click on empty area → deselect
    state.selectedSource = null;
    render();
    return;
  }
  const { col, row } = near;
  const posStr = `${col}${row}`;
  const pieceHere = state.pieces.get(posKey(col, row));
  // Always read side/phase from the post-move-fresh sources (radio +
  // phaseSel), not state.lastResult — the latter goes stale between turns
  // in Play+Engine when spoilers are off (no auto-evaluate runs).
  const side = currentSide();
  const sideRing = side === "WHITE" ? "WHITE_RING" : "BLACK_RING";
  const currentPhase = phaseSel.value;

  // If there's a selected source and this position is a legal MOVE_RING dest → apply
  if (state.selectedSource) {
    const dests = legalDestinationsFrom(state.selectedSource);
    if (dests.some((d) => d.col === col && d.row === row)) {
      const move = {
        type: "MOVE_RING",
        source: `${state.selectedSource.col}${state.selectedSource.row}`,
        destination: posStr,
      };
      state.selectedSource = null;
      applyMove(move);
      return;
    }
  }

  // Click on a ring of the current side → select it
  if (pieceHere === sideRing) {
    // Only select if there are MOVE_RING moves available from this source.
    const hasMoves = state.legalMoves.some(
      (m) => m.type === "MOVE_RING" && m.source === posStr,
    );
    if (hasMoves) {
      state.selectedSource = { col, row };
      render();
      return;
    }
  }

  // RING_PLACEMENT: clicking on an empty position places a ring of side-to-move
  if (currentPhase === "RING_PLACEMENT" && !pieceHere) {
    const placeMove = state.legalMoves.find(
      (m) => m.type === "PLACE_RING" && m.source === posStr,
    );
    if (placeMove) {
      applyMove({ type: "PLACE_RING", source: posStr });
      return;
    }
  }

  // RING_REMOVAL: clicking own ring removes it
  if (currentPhase === "RING_REMOVAL" && pieceHere === sideRing) {
    const removeMove = state.legalMoves.find(
      (m) => m.type === "REMOVE_RING" && m.source === posStr,
    );
    if (removeMove) {
      applyMove({ type: "REMOVE_RING", source: posStr });
      return;
    }
  }

  // Nothing actionable — just deselect
  state.selectedSource = null;
  render();
}

function currentSide() {
  const el = document.querySelector('input[name="side"]:checked');
  return el ? el.value : "WHITE";
}

async function applyMove(moveSpec) {
  state.busy = true;
  state.hoverArrow = null;
  // Snapshot the pre-move pieces for the animation diff. Must be a copy —
  // applyNewPosition mutates state.pieces in place.
  const prevPiecesSnapshot = new Map(state.pieces);
  try {
    const res = await fetch("/api/move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...currentPositionPayload(), move: moveSpec }),
    });
    const data = await res.json();
    if (!data.ok) {
      setStatus((data.errors || ["move failed"]).join(" · "), "error");
      return;
    }
    // Push history (pre-move position) before mutating.
    state.history.push({
      position: currentPositionPayload(),
      move: data.applied_move,
    });
    updateUndoBtn();
    appendHistoryEntry(data.applied_move);
    // Build the new-pieces map without mutating state.pieces yet — we want
    // sidebar state (side/phase/badge) to stay aligned with the board
    // during the animation, so applyNewPosition runs AFTER animateMove.
    const newPiecesMap = new Map();
    for (const p of data.new_position.pieces) newPiecesMap.set(p.pos, p.piece);
    await animateMove({
      move: data.applied_move,
      prevPieces: prevPiecesSnapshot,
      newPieces: newPiecesMap,
    });
    // Commit pieces + sidebar all at once (in sync with the canonical scene).
    applyNewPosition(data.new_position);
    state.legalMoves = data.legal_moves || [];
    state.gameOver = data.game_over;
    state.winner = data.winner;
    // Snap the scene to canonical state before any downstream await
    // (engine reply / autoEval fetch) so transient animation meshes don't
    // linger across that fetch.
    render();
    if (state.gameOver) {
      setStatus(
        `Game over — ${state.winner ? state.winner + " wins" : "draw"}.`,
        "success",
      );
      state.legalMoves = [];
      state.lastResult = null;
      renderResult(null);
      if (state.winner) {
        updateTurnBadge(state.winner, "GAME_OVER");
      }
      // If we're in Game mode, swap the result banner in instead of just
      // a status message.
      if (state.game && state.game.phase === "playing") {
        handleGameEnd();
      }
    } else if (state.game && state.game.phase === "playing") {
      // Game-mode: don't auto-evaluate after every move, but DO trigger the
      // engine to play if it's now its turn. The engine's own evaluate
      // happens inside computerMakeMove with the configured sim budget.
      const nextSide = data.new_position.side_to_move;
      const nextPhase = data.new_position.phase;
      const inCapture = nextPhase === "ROW_COMPLETION" || nextPhase === "RING_REMOVAL";
      if (nextSide === state.game.computerSide) {
        setTimeout(computerMakeMove, 600);
      } else if (state.game.spoilersEnabled || inCapture) {
        // Refresh the analysis. Spoilers-on: user wants the panel updated.
        // Capture phase: the analysis panel is the human's ONLY UI to pick
        // which row/ring to remove, so it needs fresh top_moves regardless
        // of the spoilers setting. force-show-analysis CSS will reveal it.
        await evaluate({ skipPositionCheck: true });
      }
    } else if (state.autoEval) {
      await evaluate({ skipPositionCheck: true });
    } else {
      // Even with auto-eval off, surface the new phase so multi-row
      // captures don't look like a bug.
      let note = "";
      if (data.new_position.phase === "ROW_COMPLETION") {
        note = " — capture: pick a row to remove";
      } else if (data.new_position.phase === "RING_REMOVAL") {
        note = " — capture: remove a ring";
      }
      setStatus(
        `Move applied: ${data.applied_move.description}${note}. ` +
        `${data.new_position.side_to_move} to move. (auto-evaluate off)`,
        "success",
      );
    }
  } catch (e) {
    setStatus("Move request failed: " + e.message, "error");
  } finally {
    state.busy = false;
    render();
  }
}

function applyNewPosition(newPos) {
  // Update pieces map
  state.pieces.clear();
  for (const p of newPos.pieces) {
    state.pieces.set(p.pos, p.piece);
  }
  // Update phase + side
  phaseSel.value = newPos.phase;
  const sideRadio = document.querySelector(
    `input[name="side"][value="${newPos.side_to_move}"]`,
  );
  if (sideRadio) sideRadio.checked = true;
  // Persist _move_maker across the capture sequence — without this, the
  // server's stateless GameState loses it on every rebuild and the player
  // never flips at the end of RING_REMOVAL.
  state.moveMaker = newPos.move_maker || null;
  updateDerivedStats();
  updateTurnBadge(newPos.side_to_move, newPos.phase);
  // Force-show the analysis panel during capture sequences in Game mode —
  // the human's only UI for picking which row/ring to capture is the
  // top-moves panel, so spoilers-off can't hide it then.
  const inCapture = (newPos.phase === "ROW_COMPLETION" || newPos.phase === "RING_REMOVAL");
  document.body.classList.toggle("force-show-analysis", inCapture);
}

function updateTurnBadge(side, phase) {
  if (!turnBadge) return;
  turnBadge.dataset.side = side;
  turnSideEl.textContent = side;
  // Surface non-MAIN_GAME phases prominently so multi-row captures are
  // unambiguous: if "X to move" looks wrong but the phase reads
  // ROW_COMPLETION / RING_REMOVAL, that's the same player legitimately
  // continuing a capture sequence — not a turn-flip bug.
  let phaseLabel = "";
  let alert = false;
  if (phase === "ROW_COMPLETION") {
    phaseLabel = "still capturing: pick row";
    alert = true;
  } else if (phase === "RING_REMOVAL") {
    phaseLabel = "still capturing: pick ring";
    alert = true;
  } else if (phase === "RING_PLACEMENT") {
    phaseLabel = "ring placement";
  } else if (phase === "GAME_OVER") {
    phaseLabel = "game over";
  }
  turnPhaseEl.textContent = phaseLabel;
  turnPhaseEl.classList.toggle("alert", alert);
}

function currentPositionPayload() {
  const pieces = [];
  for (const [key, piece] of state.pieces) pieces.push({ pos: key, piece });
  return {
    pieces,
    phase: phaseSel.value,
    side_to_move: currentSide(),
    scores: derivedScores(),
    move_maker: state.moveMaker,
    play_session_id: state.playSessionId,
  };
}

function handlePlace(px, py, tool) {
  const near = pixelToBoardPos(px, py);
  if (!near) return;
  const key = posKey(near.col, near.row);
  const existing = state.pieces.get(key);
  if (tool === "ERASE") {
    state.pieces.delete(key);
  } else {
    // Enforce ring cap: max 5 of each color. The piece at `key` is being
    // replaced, so don't count it against the cap.
    if (tool === "WHITE_RING" || tool === "BLACK_RING") {
      const { w, b } = countRings();
      const replacingSameColor = existing === tool;
      if (!replacingSameColor) {
        if (tool === "WHITE_RING" && w >= 5) {
          setStatus("Can't place a 6th white ring (max 5 per side).", "error");
          return;
        }
        if (tool === "BLACK_RING" && b >= 5) {
          setStatus("Can't place a 6th black ring (max 5 per side).", "error");
          return;
        }
      }
    }
    state.pieces.set(key, tool);
  }
  updateDerivedStats();
  // Composing a new position invalidates the prior arrow overlay.
  state.hoverArrow = null;
  state.lastResult = null;
  render();
}

// ---------- Sidebar wiring ----------
const $ = (id) => document.getElementById(id);
const modelSel = $("model");
const phaseSel = $("phase");
const whiteRingsCountEl = $("white-rings-count");
const blackRingsCountEl = $("black-rings-count");
const whiteScoreEl = $("white-score-derived");
const blackScoreEl = $("black-score-derived");
const numSims = $("num-sims");
const evalModeEl = $("eval-mode");
const heuristicWeightEl = $("heuristic-weight");
const heuristicWeightField = $("heuristic-weight-field");
const cPuctEl = $("c-puct");
const fpuReductionEl = $("fpu-reduction");
const ownerTokenEl = $("owner-token");
const resetAdvancedBtn = $("reset-advanced");

// Owner token: persisted to localStorage so it survives reloads (the owner
// pastes it once). Empty for every normal/public visitor. Sent with eval
// requests to bypass the public YNS_MAX_NUM_SIMS cap server-side.
const OWNER_TOKEN_KEY = "yns_owner_token";
if (ownerTokenEl) {
  ownerTokenEl.value = localStorage.getItem(OWNER_TOKEN_KEY) || "";
  ownerTokenEl.addEventListener("change", () => {
    const v = ownerTokenEl.value.trim();
    if (v) localStorage.setItem(OWNER_TOKEN_KEY, v);
    else localStorage.removeItem(OWNER_TOKEN_KEY);
  });
}
const ownerToken = () => (ownerTokenEl ? ownerTokenEl.value.trim() : "");

// Advanced-MCTS defaults — kept here as the single source of truth for
// the reset button and the server-side defaults (they must agree).
const MCTS_DEFAULTS = {
  evaluation_mode: "pure_neural",
  heuristic_weight: 0.5,
  c_puct: 1.0,
  fpu_reduction: 0.25,
};

function syncHeuristicWeightEnabled() {
  // The heuristic_weight knob is meaningless in pure_neural mode — disable
  // it visually so the user doesn't tweak a dial that has no effect.
  const mode = evalModeEl.value;
  const disabled = (mode === "pure_neural");
  heuristicWeightEl.disabled = disabled;
  heuristicWeightField.style.opacity = disabled ? "0.55" : "1";
}

evalModeEl.addEventListener("change", syncHeuristicWeightEnabled);
resetAdvancedBtn.addEventListener("click", () => {
  evalModeEl.value = MCTS_DEFAULTS.evaluation_mode;
  heuristicWeightEl.value = MCTS_DEFAULTS.heuristic_weight;
  cPuctEl.value = MCTS_DEFAULTS.c_puct;
  fpuReductionEl.value = MCTS_DEFAULTS.fpu_reduction;
  syncHeuristicWeightEnabled();
});
syncHeuristicWeightEnabled();
const evalBtn = $("evaluate");
const clearBtn = $("clear");
const playFromHereBtn = $("play-from-here");
const statusEl = $("status");
const valueRow = $("value-row");
const valueFill = $("value-fill");
const valueNum = $("value-num");
const valueLabel = $("value-label");
const bestValueRow = $("best-value-row");
const bestValueFill = $("best-value-fill");
const bestValueNum = $("best-value-num");
const valueHelp = $("value-help");
const topMovesEl = $("top-moves");
const playControls = $("play-controls");
const undoBtn = $("undo");
const autoEvalEl = $("auto-eval");
const boardHint = $("board-hint");
const historyBlock = $("history-block");
const moveHistoryEl = $("move-history");
const paletteBlock = $("palette-block");
// Game mode elements
const gameSetupBlock = $("game-setup-block");
const gameStatusBlock = $("game-status-block");
const gameHumanSideEl = $("game-human-side");
const gameSimsEl = $("game-sims");
// Play-mode engine is hard-capped here regardless of the owner-token cap
// bypass: a game against the engine should stay at a sane, snappy budget
// (the "Deep — 3200 sims" dropdown ceiling), not balloon into a multi-minute
// per-move analysis. The owner bypass is for one-off position analysis only;
// the play-mode payload (currentPositionPayload) deliberately omits the token,
// and this clamp guards against a tampered dropdown value too.
const PLAY_MODE_MAX_SIMS = 3200;
const gameModelEl = $("game-model");
const gameSpoilersEl = $("game-spoilers");
const gameSpoilersInlineEl = $("game-spoilers-inline");
const gameStartBtn = $("game-start");
const gameNewBtn = $("game-new");
const gameReviewBtn = $("game-review");
const gameResignBtn = $("game-resign");
const gameThinkingEl = $("game-thinking");
const gameResultEl = $("game-result");
const gameResultLabelEl = $("game-result-label");
const gameEngineLabelEl = $("game-engine-label");
const gameHumanLabelEl = $("game-human-label");
const turnBadge = $("turn-badge");
const turnSideEl = $("turn-side");
const turnPhaseEl = $("turn-phase");
const sideFieldEl = document.querySelector(".side-field");
const lineNav = $("line-nav");
const lineMoveDescEl = $("line-move-desc");
const lineStepCounterEl = $("line-step-counter");
const lineJustPlayedEl = $("line-just-played");
const linePrevBtn = $("line-prev");
const lineNextBtn = $("line-next");
const lineReturnBtn = $("line-return");
// Review mode (BGA game import + step-through)
const reviewBlock = $("review-block");
const reviewInput = $("review-input");
const reviewImportBtn = $("review-import");
const reviewClearBtn = $("review-clear");
const reviewStatusEl = $("review-status");
const reviewMetaEl = $("review-meta");
const reviewWhiteEl = $("review-white-player");
const reviewBlackEl = $("review-black-player");
const reviewResultEl = $("review-result");
const reviewStepCounterEl = $("review-step-counter");
const reviewControlsEl = $("review-controls");
const reviewPrevBtn = $("review-prev");
const reviewNextBtn = $("review-next");

clearBtn.addEventListener("click", () => {
  state.pieces.clear();
  state.hoverArrow = null;
  state.lastResult = null;
  // Fresh play session — moves played after a Clear belong to a new game.
  state.playSessionId = _uuid();
  updateDerivedStats();
  renderResult(null);
  render();
});

// Phase change can flip RING_PLACEMENT → MAIN_GAME, which changes derived score.
phaseSel.addEventListener("change", updateDerivedStats);

evalBtn.addEventListener("click", evaluate);

if (playFromHereBtn) {
  playFromHereBtn.addEventListener("click", () => {
    // Carries the composed state.pieces / phase / side into Play mode.
    // setMode("play") auto-evaluates if there's a position.
    setMode("play");
  });
}

document.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && document.activeElement?.tagName !== "INPUT" &&
      document.activeElement?.tagName !== "SELECT" && document.activeElement?.tagName !== "TEXTAREA") {
    evaluate();
  }
  if (ev.key === "Escape") {
    setArmedTool(null);
    state.hoverPos = null;
    render();
  }
});

async function loadModels() {
  try {
    const res = await fetch("/api/models");
    const list = await res.json();
    state.models = list;
    modelSel.innerHTML = "";
    if (gameModelEl) gameModelEl.innerHTML = "";
    if (list.length === 0) {
      const empty = document.createElement("option");
      empty.textContent = "(no models found in models/)";
      empty.value = "";
      modelSel.appendChild(empty);
      if (gameModelEl) gameModelEl.appendChild(empty.cloneNode(true));
      return;
    }
    for (const m of list) {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = `${m.label} · ${m.checkpoint}`;
      modelSel.appendChild(opt);
      if (gameModelEl) gameModelEl.appendChild(opt.cloneNode(true));
    }
    // Prefer yngine_volume_15ch_pretrain if present.
    const preferred = list.find((m) => m.label === "yngine_volume_15ch_pretrain");
    if (preferred) {
      modelSel.value = preferred.id;
      if (gameModelEl) gameModelEl.value = preferred.id;
    }
  } catch (e) {
    setStatus("Could not load models: " + e.message, "error");
  }
}

function countRings() {
  let w = 0, b = 0;
  for (const piece of state.pieces.values()) {
    if (piece === "WHITE_RING") w++;
    else if (piece === "BLACK_RING") b++;
  }
  return { w, b };
}

function derivedScores() {
  // Score is 0 during RING_PLACEMENT (game hasn't fully started).
  // Otherwise: score = 5 − rings_on_board, since every captured row removes
  // one ring from the board. The two together always sum to 5 per player.
  if (phaseSel.value === "RING_PLACEMENT") return { WHITE: 0, BLACK: 0 };
  const { w, b } = countRings();
  return { WHITE: Math.max(0, 5 - w), BLACK: Math.max(0, 5 - b) };
}

function updateDerivedStats() {
  const { w, b } = countRings();
  const scores = derivedScores();
  whiteRingsCountEl.textContent = w;
  blackRingsCountEl.textContent = b;
  whiteScoreEl.textContent = scores.WHITE;
  blackScoreEl.textContent = scores.BLACK;
  // Mirror the score into the corner reserve slots — captured rings appear
  // as 3D rings in white/black corners so the score reads at a glance.
  setCapturedRings(scores.WHITE, scores.BLACK);
  // Grey out ring tools when the cap is hit.
  for (const btn of palette.querySelectorAll(".tool")) {
    const tool = btn.dataset.tool;
    if (tool === "WHITE_RING") btn.classList.toggle("disabled", w >= 5);
    else if (tool === "BLACK_RING") btn.classList.toggle("disabled", b >= 5);
  }
}

function currentPayload() {
  const pieces = [];
  for (const [key, piece] of state.pieces) {
    pieces.push({ pos: key, piece });
  }
  const sideEl = document.querySelector('input[name="side"]:checked');
  return {
    model_id: modelSel.value,
    pieces,
    phase: phaseSel.value,
    side_to_move: sideEl ? sideEl.value : "WHITE",
    scores: derivedScores(),
    num_sims: Math.max(0, parseInt(numSims.value, 10) || 0),
    owner_token: ownerToken(),
    top_k: 8,
    move_maker: state.moveMaker,
    // Advanced MCTS knobs — server falls back to training defaults if absent,
    // but we always send to keep the cache key stable across requests.
    evaluation_mode: evalModeEl.value,
    heuristic_weight: parseFloat(heuristicWeightEl.value) || MCTS_DEFAULTS.heuristic_weight,
    c_puct: parseFloat(cPuctEl.value) || MCTS_DEFAULTS.c_puct,
    fpu_reduction: parseFloat(fpuReductionEl.value) || MCTS_DEFAULTS.fpu_reduction,
  };
}

async function evaluate(opts = {}) {
  if (!modelSel.value) {
    setStatus("No model selected.", "error");
    return;
  }
  if (!opts.skipPositionCheck && state.pieces.size === 0) {
    setStatus("Place some pieces first (or set phase=RING_PLACEMENT).", "error");
  }
  evalBtn.disabled = true;
  const sims = Math.max(0, parseInt(numSims.value, 10) || 0);
  setStatus(sims > 0 ? `Running MCTS (${sims} sims)…` : "Asking the network…", "thinking");
  try {
    const res = await fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentPayload()),
    });
    const data = await res.json();
    if (!data.ok) {
      setStatus((data.errors || ["evaluation failed"]).join(" · "), "error");
      renderResult(null);
      return;
    }
    state.lastResult = data;
    state.legalMoves = data.legal_moves || [];
    updateTurnBadge(data.side_to_move, data.phase);
    // In Play mode, multi-row capture sequences keep the same player on
    // move across multiple plies. Make this unambiguous in the status
    // banner so it doesn't look like a turn-flip bug.
    let captureNote = "";
    if (state.mode === "play") {
      if (data.phase === "ROW_COMPLETION") {
        captureNote = " · capture: pick a row to remove";
      } else if (data.phase === "RING_REMOVAL") {
        captureNote = " · capture: remove a ring";
      }
    }
    // If the public cap clamped the search, say so out loud instead of
    // silently running fewer sims than requested.
    const capNote = data.capped_from
      ? ` · ⚠ capped to ${data.num_sims} of ${data.capped_from} requested (public limit)`
      : "";
    setStatus(
      `${data.mode === "mcts" ? "MCTS" : "Network policy"} · ` +
      `${data.side_to_move} to move${captureNote} · ${data.num_valid_moves} legal moves${capNote}`,
      data.capped_from ? "warning" : "success",
    );
    renderResult(data);
  } catch (e) {
    setStatus("Request failed: " + e.message, "error");
    renderResult(null);
  } finally {
    evalBtn.disabled = false;
  }
}

function setStatus(text, kind) {
  statusEl.textContent = text;
  statusEl.className = "status" + (kind ? " " + kind : "");
}

function renderResult(data) {
  if (!data) {
    valueRow.hidden = true;
    bestValueRow.hidden = true;
    valueHelp.hidden = true;
    topMovesEl.innerHTML = "";
    state.hoverArrow = null;
    render();
    return;
  }
  // Universal chess-convention POV: + = WHITE winning, − = BLACK winning,
  // in every mode regardless of whose turn it is. Server returns
  // side-to-move POV; we flip when BLACK is to move for display.
  data = toWhitePov(data);
  // Value bars — pure_neural values ∈ [-1, 1] for side_to_move; hybrid /
  // pure_heuristic produce unbounded heuristic-units scores (the heuristic's
  // 7-feature weighted sum isn't tanh'd), so we display the raw number but
  // clamp the bar render to [-1, 1] to keep the UI sane.
  const mode = (evalModeEl && evalModeEl.value) || "pure_neural";
  const heuristicMode = mode !== "pure_neural";
  const v = data.value;
  const vBar = Math.max(-1, Math.min(1, v));
  valueRow.hidden = false;
  const modeTag = heuristicMode ? " · heuristic units" : "";
  // Universal WHITE-POV convention — explicit in the label so a new user
  // doesn't mistake "−0.5 with BLACK to move" for "BLACK losing."
  valueLabel.textContent = `Search avg (WHITE POV)${modeTag}`;
  valueNum.textContent = heuristicMode ? v.toFixed(2) : v.toFixed(3);
  paintValueBar(valueFill, vBar);

  if (data.best_move_value !== null && data.best_move_value !== undefined) {
    const bv = Math.max(-1, Math.min(1, data.best_move_value));
    bestValueRow.hidden = false;
    bestValueNum.textContent = bv.toFixed(3);
    paintValueBar(bestValueFill, bv);
  } else {
    bestValueRow.hidden = true;
  }
  valueHelp.hidden = false;

  // Top moves
  topMovesEl.innerHTML = "";
  // Sign of the headline search_avg — used to flag rows whose after-value
  // disagrees. For heuristic modes, the headline is in heuristic units so
  // sign is still comparable to the network's [-1, 1] best_move_value.
  const headlineSign = Math.sign(data.value || 0);
  // In Play+Engine with spoilers off + capture phase, resort the list by
  // canonical board position so the user picks blindly — the engine's
  // preferred row is no longer at position 1. The percentages / visits /
  // value bars are hidden via CSS in this mode.
  const inCaptureBlind = (
    state.game?.phase === "playing" &&
    !state.game.spoilersEnabled &&
    (data.phase === "ROW_COMPLETION" || data.phase === "RING_REMOVAL")
  );
  const moveCanonKey = (mv) => {
    if (mv.type === "REMOVE_MARKERS" && mv.markers && mv.markers.length) return mv.markers[0];
    return mv.source || "";
  };
  const moves = inCaptureBlind
    ? [...data.top_moves].sort((a, b) => moveCanonKey(a).localeCompare(moveCanonKey(b)))
    : data.top_moves;
  moves.forEach((mv, i) => {
    const li = document.createElement("li");
    if (i === 0) li.classList.add("top");
    li.dataset.idx = i;
    // Per-row best_move_value cell — sign-colored, with a row-level
    // "divergent" flag when the sign disagrees with the headline search_avg.
    let bmvClass = "unknown";
    let bmvText = "—";
    let bmv = mv.best_move_value;
    if (typeof bmv === "number") {
      bmvText = (bmv >= 0 ? "+" : "") + bmv.toFixed(2);
      if (Math.abs(bmv) < 0.05) bmvClass = "neutral";
      else if (bmv > 0) bmvClass = "positive";
      else bmvClass = "negative";
      // Flag opposite-sign divergence: only meaningful when both have
      // a clear sign (skip near-zero noise on either side).
      const bmvSign = Math.sign(bmv);
      if (headlineSign !== 0 && bmvSign !== 0 && bmvSign !== headlineSign && Math.abs(bmv) > 0.05) {
        li.classList.add("divergent");
      }
    }
    li.innerHTML = `
      <span class="rank">${i + 1}.</span>
      <span class="move-desc">${formatMove(mv)}</span>
      <span class="bmv ${bmvClass}" title="best_move_value — network value at position after this move (side-to-move POV)">${bmvText}</span>
      <span class="prob-cell">
        ${mv.visits !== undefined ? `<span class="visits">${mv.visits}v</span>` : ""}
        <span class="prob-bar"><span class="prob-fill" style="width:${(mv.prob * 100).toFixed(1)}%"></span></span>
        <span class="prob-num">${(mv.prob * 100).toFixed(1)}%</span>
        <button class="step-into" type="button">▶</button>
      </span>`;
    li.addEventListener("mouseenter", () => {
      state.hoverArrow = moveToArrow(mv);
      render();
    });
    li.addEventListener("mouseleave", () => {
      state.hoverArrow = null;
      render();
    });
    li.addEventListener("click", (ev) => {
      if (ev.target.closest(".step-into")) return;
      if (state.busy) return;
      if (state.lineMode) return;
      // Click-to-apply works in Play mode (self-play). In Play+Engine, it
      // also works during a capture sequence (REMOVE_MARKERS / REMOVE_RING)
      // since the human must pick a candidate — the analysis panel is
      // forced visible there even with spoilers off.
      if (state.mode !== "play") return;
      const inCapture = data && (data.phase === "ROW_COMPLETION" || data.phase === "RING_REMOVAL");
      if (state.opponent === "engine") {
        if (!(state.game?.phase === "playing" && inCapture)) return;
        // currentSide() reflects the post-move radio state, kept in sync by
        // applyNewPosition. data.side_to_move can be stale between turns
        // when spoilers are off.
        if (currentSide() !== state.game.humanSide) return;
      }
      applyMove({
        type: mv.type,
        source: mv.source,
        destination: mv.destination,
        markers: mv.markers,
      });
    });
    if (state.mode === "play") li.style.cursor = "pointer";
    // Step-into-line button (▶). Only meaningful when MCTS produced a PV.
    const stepBtn = li.querySelector(".step-into");
    if (stepBtn) {
      const hasPv = Array.isArray(mv.principal_variation) && mv.principal_variation.length > 0;
      stepBtn.disabled = !hasPv;
      stepBtn.title = hasPv
        ? "Step into this line on the board"
        : "Need MCTS sims > 0 to see the principal variation";
      stepBtn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        if (hasPv) enterLineMode(i);
      });
    }
    topMovesEl.appendChild(li);
  });

  // Auto-show the top move arrow on result — UNLESS we're playing vs the
  // engine with spoilers off. The on-board arrow is rendered by three.js
  // and bypasses the CSS hide of the result panel. In capture phases the
  // panel is force-shown so the human can pick, but we still suppress the
  // arrow there: the arrow highlights the engine's preferred source/row,
  // which is a spoiler.
  if (data.top_moves.length > 0) {
    const spoilersHide = state.game?.phase === "playing" && !state.game.spoilersEnabled;
    state.hoverArrow = spoilersHide ? null : moveToArrow(data.top_moves[0]);
  }
  render();
}

function toWhitePov(data) {
  // Chess convention: + means WHITE winning, − means BLACK winning, in EVERY
  // mode regardless of whose turn it is. The server returns values in
  // side-to-move POV (matches MCTS internals), so we flip when BLACK is to
  // move to put everything in WHITE's frame for display. The transformation
  // is uniform across Setup / Play / Game so users have one mental model.
  //
  // Applies to: headline search_avg, best_move_value, and the per-row
  // best_move_value pills inside top_moves. Move probabilities + visits are
  // not POV-dependent so they pass through unchanged.
  if (!data || typeof data.side_to_move !== "string") return data;
  if (data.side_to_move === "WHITE") return data;
  const out = { ...data };
  if (typeof out.value === "number") out.value = -out.value;
  if (typeof out.best_move_value === "number") out.best_move_value = -out.best_move_value;
  if (Array.isArray(out.top_moves)) {
    out.top_moves = out.top_moves.map((m) => ({
      ...m,
      best_move_value: (typeof m.best_move_value === "number") ? -m.best_move_value : m.best_move_value,
    }));
  }
  return out;
}

function paintValueBar(fillEl, v) {
  if (v >= 0) {
    fillEl.classList.remove("negative");
    fillEl.style.left = "50%";
    fillEl.style.width = `${(v / 2) * 100}%`;
  } else {
    fillEl.classList.add("negative");
    fillEl.style.left = `${50 - (Math.abs(v) / 2) * 100}%`;
    fillEl.style.width = `${(Math.abs(v) / 2) * 100}%`;
  }
}

function formatMove(mv) {
  if (mv.type === "PLACE_RING") return `place @ ${mv.source}`;
  if (mv.type === "MOVE_RING") return `${mv.source} → ${mv.destination}`;
  if (mv.type === "REMOVE_RING") return `remove ring @ ${mv.source}`;
  if (mv.type === "REMOVE_MARKERS") return `remove row [${(mv.markers || []).join(",")}]`;
  return mv.description || JSON.stringify(mv);
}

function parsePos(s) {
  if (!s) return null;
  return { col: s[0], row: parseInt(s.slice(1), 10) };
}

function moveToArrow(mv) {
  const src = parsePos(mv.source);
  if (mv.type === "MOVE_RING") return { from: src, to: parsePos(mv.destination) };
  if (mv.type === "PLACE_RING" || mv.type === "REMOVE_RING") return { from: src, to: null };
  if (mv.type === "REMOVE_MARKERS" && mv.markers && mv.markers.length) {
    return { from: parsePos(mv.markers[0]), to: parsePos(mv.markers[mv.markers.length - 1]) };
  }
  return null;
}

// Right-click erases — keep stats in sync after erase.
boardEl.addEventListener("contextmenu", () => updateDerivedStats());

// ---------- Mode + Opponent toggles ----------
document.querySelectorAll(".mode-btn").forEach((btn) => {
  btn.addEventListener("click", () => setMode(btn.dataset.mode));
});
document.querySelectorAll(".opp-btn").forEach((btn) => {
  btn.addEventListener("click", () => setOpponent(btn.dataset.opponent));
});

function applyModeStyling(mode) {
  document.querySelectorAll(".mode-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.mode === mode);
    b.setAttribute("aria-selected", b.dataset.mode === mode ? "true" : "false");
  });
  // Body classes drive most of the per-mode CSS hide/show. These need to be
  // set on initial page load too, not just on user toggle — otherwise the
  // body has no mode class and per-mode CSS never fires.
  document.body.classList.toggle("mode-setup", mode === "setup");
  document.body.classList.toggle("mode-play", mode === "play");
  document.body.classList.toggle("mode-review", mode === "review");

  paletteBlock.classList.toggle("hidden-in-play", mode !== "setup");
  playControls.hidden = mode !== "play";
  // History block is meaningful in Play (live moves) and Review (BGA replay).
  // In Setup there's no game flow, so hide it.
  historyBlock.hidden = mode === "setup";
  if (sideFieldEl) sideFieldEl.classList.toggle("locked", mode !== "setup");
  // The Review block is gated by its own `hidden` attribute (we toggle here)
  // AND by a `body.mode-review` CSS rule — the latter exists so a stale
  // `hidden=false` from a prior session doesn't bleed through.
  if (reviewBlock) reviewBlock.hidden = mode !== "review";

  const sub = document.getElementById("sidebar-subtitle");
  if (sub) {
    if (mode === "setup") sub.textContent = "Compose a position, ask the network.";
    else if (mode === "review") sub.textContent = "Step through a BGA replay with engine analysis.";
    else sub.textContent = "Play YINSH — vs yourself or the engine.";
  }
}

function applyOpponentStyling(opp) {
  document.querySelectorAll(".opp-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.opponent === opp);
    b.setAttribute("aria-selected", b.dataset.opponent === opp ? "true" : "false");
  });
  document.body.classList.toggle("opponent-self", opp === "self");
  document.body.classList.toggle("opponent-engine", opp === "engine");
}

async function setMode(mode) {
  if (state.mode === mode) return;
  // Leaving Review tears down the loaded game and restores the prior pieces.
  // Without this the next mode would land on the last-viewed BGA step instead
  // of whatever the user had before.
  if (state.mode === "review" && mode !== "review") {
    exitReviewMode({ silent: true });
  }
  state.mode = mode;
  applyModeStyling(mode);
  state.selectedSource = null;

  if (mode === "review") {
    boardHint.textContent = "Paste a BGA URL on the right to import a game.";
    setArmedTool(null);
    // Tear down any in-flight Play game so leaving Review back to Play doesn't
    // resume an unrelated game state.
    if (state.game) endGameSession();
    state.history = [];
    state.gameOver = false;
    state.winner = null;
    moveHistoryEl.innerHTML = "";
    updateUndoBtn();
    enterReviewMode();
    render();
    return;
  }

  if (mode === "play") {
    boardHint.textContent = "Click a ring of the side to move, then click a destination. Or click any top-move row to apply.";
    setArmedTool(null);
    // Entering Play from Setup = starting a new game from the composed
    // position. Fresh play session so the logs cleanly separate "composed
    // and played" from prior activity on the page.
    state.playSessionId = _uuid();
    if (!state.lastResult && state.pieces.size > 0) {
      await evaluate({ skipPositionCheck: true });
    } else if (!state.lastResult) {
      setStatus("Place rings/markers in Set up mode, then come back to Play.", null);
    }
    // Honor the current Opponent setting. If Engine, route into the
    // engine-session lifecycle; otherwise self-play behavior (no game state).
    applyOpponentStyling(state.opponent);
    if (state.opponent === "engine") {
      if (!state.game) enterGameSetup();
      else showGameUI();
    } else {
      if (state.game) endGameSession();
      gameSetupBlock.hidden = true;
      gameStatusBlock.hidden = true;
    }
  } else if (mode === "setup") {
    boardHint.textContent = "Drag a piece from the palette onto a board intersection. Right-click to erase.";
    state.history = [];
    state.gameOver = false;
    state.winner = null;
    updateUndoBtn();
    moveHistoryEl.innerHTML = "";
    if (state.game) endGameSession();
    gameSetupBlock.hidden = true;
    gameStatusBlock.hidden = true;
  }
  render();
}

async function setOpponent(opp) {
  if (state.opponent === opp) return;
  state.opponent = opp;
  applyOpponentStyling(opp);
  // Opponent only takes effect inside Play mode; if the user toggles it
  // while in Set up, we record the preference for when they switch to Play.
  if (state.mode !== "play") return;
  if (opp === "engine") {
    if (!state.game) enterGameSetup();
    else showGameUI();
  } else if (opp === "self") {
    if (state.game) endGameSession();
    gameSetupBlock.hidden = true;
    gameStatusBlock.hidden = true;
  }
  render();
}

// ---------- Game mode lifecycle ----------

function enterGameSetup() {
  state.game = { phase: "setup" };
  gameSetupBlock.hidden = false;
  gameStatusBlock.hidden = true;
  gameResultEl.hidden = true;
  gameThinkingEl.hidden = true;
  document.body.classList.remove("game-playing", "game-ended", "spoilers-off");
  document.body.classList.add("game-setup");
  boardHint.textContent = "Configure your game on the right, then click Start.";
}

async function startGame() {
  // Resolve the human side choice.
  let humanSide = gameHumanSideEl.value;
  if (humanSide === "random") humanSide = Math.random() < 0.5 ? "WHITE" : "BLACK";
  const computerSide = humanSide === "WHITE" ? "BLACK" : "WHITE";
  const computerSims = Math.min(PLAY_MODE_MAX_SIMS, parseInt(gameSimsEl.value, 10) || 800);
  const spoilersEnabled = !!gameSpoilersEl.checked;
  // Sync the main model dropdown to the game-setup choice so all the
  // existing /api/evaluate plumbing (which reads from modelSel.value) picks
  // the right one.
  if (gameModelEl && gameModelEl.value) modelSel.value = gameModelEl.value;

  state.game = {
    phase: "playing",
    humanSide,
    computerSide,
    computerSims,
    spoilersEnabled,
    thinking: false,
    winner: null,
  };
  // Fresh play session — the next /api/move belongs to this new game.
  state.playSessionId = _uuid();

  // Start a fresh YINSH game from RING_PLACEMENT. Server is stateless;
  // we just initialize the local state and let the user (or computer)
  // play their first move.
  state.pieces.clear();
  state.history = [];
  state.gameOver = false;
  state.winner = null;
  state.moveMaker = null;
  state.lastResult = null;
  state.legalMoves = [];
  state.hoverArrow = null;
  state.selectedSource = null;
  phaseSel.value = "RING_PLACEMENT";
  const whiteRadio = document.querySelector('input[name="side"][value="WHITE"]');
  if (whiteRadio) whiteRadio.checked = true;

  // Sync the inline spoilers toggle with the setup choice.
  gameSpoilersInlineEl.checked = spoilersEnabled;

  showGameUI();
  updateDerivedStats();
  updateTurnBadge("WHITE", "RING_PLACEMENT");
  moveHistoryEl.innerHTML = "";
  updateUndoBtn();
  render();

  // Always fetch the initial evaluation so legal moves are available
  // for the human's first click (and so the engine has a snapshot if
  // it moves first).
  await evaluate({ skipPositionCheck: true });

  // If the engine plays White, kick it off.
  if (computerSide === "WHITE") {
    setTimeout(computerMakeMove, 600);
  }
}

function showGameUI() {
  const g = state.game;
  if (!g) return;
  gameSetupBlock.hidden = g.phase !== "setup";
  gameStatusBlock.hidden = g.phase === "setup";
  document.body.classList.toggle("game-setup", g.phase === "setup");
  document.body.classList.toggle("game-playing", g.phase === "playing");
  document.body.classList.toggle("game-ended", g.phase === "ended");
  document.body.classList.toggle("spoilers-off", g.phase === "playing" && !g.spoilersEnabled);
  gameResultEl.hidden = g.phase !== "ended";
  // Engine + human labels
  const modelName = (modelSel.options[modelSel.selectedIndex]?.text || "engine").split(" ·")[0];
  gameEngineLabelEl.textContent = g.computerSide
    ? `${modelName} · ${g.computerSims} sims · plays ${g.computerSide}`
    : "—";
  gameHumanLabelEl.textContent = g.humanSide || "—";
  if (g.phase === "playing") {
    boardHint.textContent = g.spoilersEnabled
      ? "Your turn — click a ring then a destination. Engine analysis is visible."
      : "Your turn — click a ring then a destination. Engine analysis is hidden.";
  } else if (g.phase === "ended") {
    boardHint.textContent = "Game complete. New game or review the position.";
  }
}

function endGameSession() {
  state.game = null;
  document.body.classList.remove("game-setup", "game-playing", "game-ended", "spoilers-off");
  gameSetupBlock.hidden = true;
  gameStatusBlock.hidden = true;
  gameThinkingEl.hidden = true;
  gameResultEl.hidden = true;
}

async function computerMakeMove() {
  if (!state.game || state.game.phase !== "playing") return;
  if (state.busy) {
    // Re-check after a short delay; another move is in-flight.
    setTimeout(computerMakeMove, 250);
    return;
  }
  // Confirm it's actually the computer's turn — defensive against races.
  // Source of truth is the side radio (kept in sync by applyNewPosition);
  // state.lastResult.side_to_move is stale when spoilers are off because
  // no auto-evaluate runs after /api/move in that mode.
  const sideToMove = currentSide();
  if (sideToMove !== state.game.computerSide) return;

  state.game.thinking = true;
  gameThinkingEl.hidden = false;
  const thinkingStartedAt = Date.now();
  const MIN_THINKING_MS = 1500;

  let topMove = null;
  try {
    // Force a fresh evaluation with the game's configured sim budget,
    // independent of whatever num_sims the user has set in the sidebar.
    const payload = {
      ...currentPositionPayload(),
      model_id: modelSel.value,
      num_sims: state.game.computerSims,
      top_k: 1,
    };
    const res = await fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!data.ok || !data.top_moves || !data.top_moves.length) {
      setStatus("Engine couldn't find a move. Game may be stuck.", "error");
      return;
    }
    topMove = data.top_moves[0];
    // Cache the engine's evaluation so spoiler-on users see it.
    state.lastResult = data;
    state.legalMoves = data.legal_moves || [];
    if (state.game.spoilersEnabled) renderResult(data);
  } finally {
    // Wait for minimum thinking time so it doesn't feel jarring.
    const elapsed = Date.now() - thinkingStartedAt;
    if (elapsed < MIN_THINKING_MS) {
      await new Promise((r) => setTimeout(r, MIN_THINKING_MS - elapsed));
    }
    state.game.thinking = false;
    gameThinkingEl.hidden = true;
  }

  if (!topMove) return;
  await applyMove({
    type: topMove.type,
    source: topMove.source,
    destination: topMove.destination,
    markers: topMove.markers,
  });
}

function handleGameEnd() {
  if (!state.game) return;
  state.game.phase = "ended";
  // Determine winner from the scores in the current position.
  const whiteScore = parseInt($("white-score-derived").textContent, 10) || 0;
  const blackScore = parseInt($("black-score-derived").textContent, 10) || 0;
  let winner = null;
  if (whiteScore > blackScore) winner = "WHITE";
  else if (blackScore > whiteScore) winner = "BLACK";
  state.game.winner = winner;
  state.winner = winner;

  let label, kind;
  if (winner === null) {
    label = `Draw — ${whiteScore}-${blackScore}`;
    kind = "draw";
  } else if (winner === state.game.humanSide) {
    label = `You win! ${winner} ${whiteScore}-${blackScore}`;
    kind = "won";
  } else {
    label = `Engine wins. ${winner} ${whiteScore}-${blackScore}`;
    kind = "lost";
  }
  gameResultLabelEl.textContent = label;
  gameResultLabelEl.className = "game-result-banner " + kind;
  showGameUI();
}

gameStartBtn.addEventListener("click", startGame);
gameNewBtn.addEventListener("click", enterGameSetup);
gameReviewBtn.addEventListener("click", () => {
  // Review = peek the engine analysis at the final position. Simplest
  // version: just enable spoilers and let the user look at the analysis.
  // The history list already shows every move played; clicking a move
  // in history doesn't navigate yet (deferred to v2).
  if (!state.game) return;
  state.game.spoilersEnabled = true;
  gameSpoilersInlineEl.checked = true;
  showGameUI();
  evaluate({ skipPositionCheck: true });
});
gameResignBtn.addEventListener("click", () => {
  if (!state.game || state.game.phase !== "playing") return;
  if (!confirm("End this game?")) return;
  state.game.phase = "ended";
  state.game.winner = null;
  gameResultLabelEl.textContent = "Game ended (resigned)";
  gameResultLabelEl.className = "game-result-banner";
  showGameUI();
});
gameSpoilersInlineEl.addEventListener("change", async () => {
  if (!state.game) return;
  state.game.spoilersEnabled = gameSpoilersInlineEl.checked;
  document.body.classList.toggle("spoilers-off", !state.game.spoilersEnabled);
  if (state.game.spoilersEnabled) {
    // Flipping spoilers ON mid-game: state.lastResult is likely a stale
    // snapshot from before / between turns (no evaluates ran while
    // spoilers were off). Refresh so the panel shows current analysis,
    // not pre-game options.
    await evaluate({ skipPositionCheck: true });
  }
  showGameUI();
});

undoBtn.addEventListener("click", async () => {
  if (state.history.length === 0 || state.busy) return;
  state.busy = true;
  try {
    const entry = state.history.pop();
    applyNewPosition(entry.position);
    updateUndoBtn();
    popHistoryEntry();
    state.gameOver = false;
    state.winner = null;
    state.selectedSource = null;
    if (state.autoEval) await evaluate({ skipPositionCheck: true });
  } finally {
    state.busy = false;
    render();
  }
});

autoEvalEl.addEventListener("change", () => {
  state.autoEval = autoEvalEl.checked;
});

function updateUndoBtn() {
  undoBtn.disabled = state.history.length === 0;
}

function appendHistoryEntry(move) {
  const li = document.createElement("li");
  const ply = state.history.length;  // count after push
  li.innerHTML = `<span class="ply">${ply}.</span><span>${formatMoveShort(move)}</span>`;
  moveHistoryEl.appendChild(li);
  moveHistoryEl.scrollTop = moveHistoryEl.scrollHeight;
}

function popHistoryEntry() {
  if (moveHistoryEl.lastChild) moveHistoryEl.removeChild(moveHistoryEl.lastChild);
}

function formatMoveShort(mv) {
  if (mv.type === "PLACE_RING") return `place @ ${mv.source}`;
  if (mv.type === "MOVE_RING") return `${mv.source} → ${mv.destination}`;
  if (mv.type === "REMOVE_RING") return `−ring ${mv.source}`;
  if (mv.type === "REMOVE_MARKERS") return `−row [${(mv.markers || []).join(",")}]`;
  return mv.description || "?";
}

// ---------- Line mode (step-through PV) ----------
function enterLineMode(pvIndex) {
  if (!state.lastResult || !state.lastResult.top_moves) return;
  const topMove = state.lastResult.top_moves[pvIndex];
  if (!topMove || !Array.isArray(topMove.principal_variation) || topMove.principal_variation.length === 0) {
    setStatus("No principal variation available for that move.", "error");
    return;
  }

  // If already in a line, return to the base first so we restore cleanly
  // before switching to the new line.
  if (state.lineMode) exitLineMode({ silent: true });

  // Snapshot the "real" position so we can restore it on Return.
  const baseSnapshot = {
    pieces: new Map(state.pieces),
    phase: phaseSel.value,
    sideToMove: currentSide(),
    moveMaker: state.moveMaker,
    lastResult: state.lastResult,
    legalMoves: state.legalMoves,
    gameOver: state.gameOver,
    winner: state.winner,
  };

  // Build steps array. Step 0 = base position (the real one we'll return to).
  // Steps 1..N = position-after each successive PV move.
  const basePayload = currentPositionPayload();
  const steps = [
    { position: basePayload, move: null, player: null },
  ];
  for (const pvStep of topMove.principal_variation) {
    steps.push({
      position: pvStep.position_after,
      move: pvStep.move,
      player: pvStep.player,
    });
  }

  state.lineMode = {
    pvIndex,
    topMoveDesc: topMove.description,
    baseSnapshot,
    steps,
    currentStep: 0,
  };
  document.body.classList.add("line-mode");
  lineNav.hidden = false;
  // Start at step 1 (= after the chosen top move). Step 0 is "rewind to base",
  // which the user can navigate back to via ← prev.
  applyLineStep(1);
}

function applyLineStep(idx) {
  if (!state.lineMode) return;
  const N = state.lineMode.steps.length;
  idx = Math.max(0, Math.min(N - 1, idx));
  state.lineMode.currentStep = idx;
  const step = state.lineMode.steps[idx];
  applyNewPosition(step.position);
  // Selection state from real mode doesn't apply mid-line.
  state.selectedSource = null;
  // Show the move that landed us at this step (none at step 0).
  if (step.move) {
    state.hoverArrow = moveToArrow(step.move);
  } else {
    state.hoverArrow = null;
  }
  updateLineNav();
  render();
}

function exitLineMode(opts = {}) {
  if (!state.lineMode) return;
  const snap = state.lineMode.baseSnapshot;
  // Restore pieces map
  state.pieces.clear();
  for (const [k, v] of snap.pieces) state.pieces.set(k, v);
  phaseSel.value = snap.phase;
  const sideRadio = document.querySelector(`input[name="side"][value="${snap.sideToMove}"]`);
  if (sideRadio) sideRadio.checked = true;
  state.moveMaker = snap.moveMaker;
  state.lastResult = snap.lastResult;
  state.legalMoves = snap.legalMoves;
  state.gameOver = snap.gameOver;
  state.winner = snap.winner;
  state.lineMode = null;
  state.hoverArrow = null;
  state.selectedSource = null;
  document.body.classList.remove("line-mode");
  lineNav.hidden = true;
  updateDerivedStats();
  updateTurnBadge(snap.sideToMove, snap.phase);
  // Re-display the prior evaluation result if we have it.
  if (snap.lastResult) renderResult(snap.lastResult);
  render();
  if (!opts.silent) setStatus("Returned to analysis position.", null);
}

function updateLineNav() {
  if (!state.lineMode) return;
  const { topMoveDesc, steps, currentStep } = state.lineMode;
  const totalPlies = steps.length - 1;  // step 0 is the base, not a played ply
  lineMoveDescEl.textContent = topMoveDesc;
  lineStepCounterEl.textContent = `step ${currentStep}/${totalPlies}`;
  if (currentStep === 0) {
    lineJustPlayedEl.textContent = "← rewound to base position";
    lineJustPlayedEl.classList.remove("tentative");
  } else {
    const step = steps[currentStep];
    const playerTag = step.player ? `${step.player[0]}:` : "";
    const desc = step.move ? formatMoveShort(step.move) : "";
    // Show visit count for this ply — lets the user see when MCTS's
    // confidence has dropped. Single-digit visits = tentative; that gets
    // a visual flag too.
    const visits = step.move && state.lineMode && state.lineMode.steps[currentStep] ?
      (state.lastResult?.top_moves?.[state.lineMode.pvIndex]?.principal_variation?.[currentStep - 1]?.visits) : null;
    const visitsTag = (typeof visits === "number") ? ` (${visits}v)` : "";
    lineJustPlayedEl.textContent = `${playerTag} ${desc}${visitsTag}`;
    lineJustPlayedEl.classList.toggle("tentative", typeof visits === "number" && visits < 10);
  }
  linePrevBtn.disabled = currentStep <= 0;
  lineNextBtn.disabled = currentStep >= steps.length - 1;
}

linePrevBtn.addEventListener("click", () => {
  if (state.lineMode) applyLineStep(state.lineMode.currentStep - 1);
});
lineNextBtn.addEventListener("click", () => {
  if (state.lineMode) applyLineStep(state.lineMode.currentStep + 1);
});
lineReturnBtn.addEventListener("click", () => exitLineMode());

// Keyboard nav for line walking — left/right arrows step, Esc returns.
document.addEventListener("keydown", (ev) => {
  if (!state.lineMode) return;
  if (ev.key === "ArrowLeft") { ev.preventDefault(); applyLineStep(state.lineMode.currentStep - 1); }
  else if (ev.key === "ArrowRight") { ev.preventDefault(); applyLineStep(state.lineMode.currentStep + 1); }
  else if (ev.key === "Escape") { ev.preventDefault(); exitLineMode(); }
});

// ---------- Screenshot import (Set up mode) ----------
//
// Drop-zone + paste + file-picker → POST /api/import_screenshot →
// applyNewPosition. The endpoint itself is server-side (Anthropic API
// key never reaches the browser); we only handle the upload UX.
const importDropzone = $("import-dropzone");
const importFileInput = $("import-file-input");
const importStatusEl = $("import-status");
const importAutoAnalyzeEl = $("import-auto-analyze");

const IMPORT_ALLOWED_TYPES = new Set([
  "image/png", "image/jpeg", "image/webp", "image/gif",
]);
const IMPORT_MAX_BYTES = 5 * 1024 * 1024;

let importBusy = false;

function setImportStatus(text, level) {
  if (!importStatusEl) return;
  importStatusEl.classList.remove("is-error", "is-success", "is-thinking");
  if (!text) {
    importStatusEl.hidden = true;
    importStatusEl.innerHTML = "";
    return;
  }
  if (level === "error") importStatusEl.classList.add("is-error");
  else if (level === "success") importStatusEl.classList.add("is-success");
  else if (level === "thinking") importStatusEl.classList.add("is-thinking");
  importStatusEl.innerHTML = text;
  importStatusEl.hidden = false;
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      // Result is `data:<mime>;base64,<b64>` — strip the prefix so the
      // server gets bare base64 (it also accepts the data: form, but
      // bare is cleaner on the wire).
      const result = reader.result;
      if (typeof result !== "string") {
        reject(new Error("FileReader returned non-string result"));
        return;
      }
      const comma = result.indexOf(",");
      resolve(comma === -1 ? result : result.slice(comma + 1));
    };
    reader.onerror = () => reject(reader.error || new Error("FileReader failed"));
    reader.readAsDataURL(file);
  });
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

async function importScreenshotFromFile(file) {
  if (importBusy) return;
  if (!file) return;
  if (!IMPORT_ALLOWED_TYPES.has(file.type)) {
    setImportStatus(
      `Can't import ${escapeHtml(file.type || "this file")} — try PNG, JPEG, WEBP, or GIF.`,
      "error",
    );
    return;
  }
  if (file.size > IMPORT_MAX_BYTES) {
    const mb = (file.size / (1024 * 1024)).toFixed(1);
    setImportStatus(
      `Image is ${mb}MB — max 5MB. Resize or screenshot the board area only.`,
      "error",
    );
    return;
  }

  // Confirm before clobbering an in-progress composed position.
  if (state.pieces.size > 0) {
    const ok = window.confirm(
      "Importing will replace the current composed position. Continue?",
    );
    if (!ok) return;
  }

  importBusy = true;
  importDropzone.classList.add("is-busy");
  setImportStatus(
    "Parsing image with Claude vision (typically 3–8s)…",
    "thinking",
  );

  let b64;
  try {
    b64 = await fileToBase64(file);
  } catch (e) {
    importBusy = false;
    importDropzone.classList.remove("is-busy");
    setImportStatus(`Couldn't read file: ${escapeHtml(e.message)}`, "error");
    return;
  }

  try {
    const res = await fetch("/api/import_screenshot", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_base64: b64,
        mime_type: file.type,
      }),
    });
    let data;
    try {
      data = await res.json();
    } catch {
      setImportStatus(
        `Server returned non-JSON (HTTP ${res.status}). Check the server log.`,
        "error",
      );
      return;
    }
    if (!data.ok) {
      const errs = (data.errors || ["import failed"]).join(" · ");
      setImportStatus(escapeHtml(errs), "error");
      return;
    }

    // Reuse applyNewPosition (same path /api/move uses). It mutates
    // state.pieces, sets phase + side, refreshes derived stats.
    applyNewPosition(data.position);
    state.lastResult = null;
    renderResult(null);
    render();

    const pieceCount = data.position?.pieces?.length || 0;
    let msg = `Imported ${pieceCount} piece${pieceCount === 1 ? "" : "s"} · `
      + `confidence: <strong>${escapeHtml(data.confidence || "medium")}</strong> · `
      + `phase: ${escapeHtml(data.position?.phase || "?")} · `
      + `${escapeHtml(data.position?.side_to_move || "?")} to move`;
    if (data.notes) {
      msg += `<span class="import-status-notes">${escapeHtml(data.notes)}</span>`;
    }
    setImportStatus(msg, "success");

    if (importAutoAnalyzeEl?.checked) {
      // skipPositionCheck because the imported state may be a real
      // RING_PLACEMENT position with 0 pieces and we don't want the
      // auto-analyze to error on that.
      await evaluate({ skipPositionCheck: true });
    }
  } catch (e) {
    setImportStatus(`Request failed: ${escapeHtml(e.message)}`, "error");
  } finally {
    importBusy = false;
    importDropzone.classList.remove("is-busy");
  }
}

if (importDropzone && importFileInput) {
  importDropzone.addEventListener("click", () => {
    if (importBusy) return;
    importFileInput.value = "";  // allow re-picking the same file
    importFileInput.click();
  });
  importDropzone.addEventListener("keydown", (ev) => {
    if (importBusy) return;
    if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault();
      importFileInput.value = "";
      importFileInput.click();
    }
  });

  importFileInput.addEventListener("change", () => {
    const file = importFileInput.files?.[0];
    if (file) importScreenshotFromFile(file);
  });

  importDropzone.addEventListener("dragover", (ev) => {
    ev.preventDefault();
    importDropzone.classList.add("is-dragover");
  });
  importDropzone.addEventListener("dragleave", () => {
    importDropzone.classList.remove("is-dragover");
  });
  importDropzone.addEventListener("drop", (ev) => {
    ev.preventDefault();
    importDropzone.classList.remove("is-dragover");
    const file = ev.dataTransfer?.files?.[0];
    if (file) importScreenshotFromFile(file);
  });

  // Global paste — works anywhere on the page while in Setup mode. Skip
  // when the user is pasting text into an input (model name, sim count,
  // etc.) so we don't intercept their non-image clipboard.
  document.addEventListener("paste", (ev) => {
    if (state.mode !== "setup") return;
    if (importBusy) return;
    const target = ev.target;
    const tag = target?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    const items = ev.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
      if (item.kind === "file" && item.type.startsWith("image/")) {
        ev.preventDefault();
        const file = item.getAsFile();
        if (file) importScreenshotFromFile(file);
        return;
      }
    }
  });
}
// ---------- Review mode (BGA game import + step-through) ----------
//
// Lives alongside but does NOT share state with line-mode (PV walking). Two
// distinct workflows: line-mode walks an engine PV from the live position,
// review-mode walks a finished human game from a paste. The board renderer
// is shared but the navigation surfaces and state are independent.

function enterReviewMode() {
  // Snapshot whatever the user had open so a non-empty Setup or completed
  // game survives the trip into review and back.
  const baseSnapshot = {
    pieces: new Map(state.pieces),
    phase: phaseSel.value,
    sideToMove: currentSide(),
    moveMaker: state.moveMaker,
    lastResult: state.lastResult,
    legalMoves: state.legalMoves,
    gameOver: state.gameOver,
    winner: state.winner,
  };
  // Start with no game loaded — JS UI shows the input + Import button.
  state.review = {
    tableId: null,
    metadata: null,
    steps: null,
    currentStep: 0,
    baseSnapshot,
  };
  if (reviewInput) reviewInput.focus();
  setReviewStatus(null);
  reviewMetaEl.hidden = true;
  reviewControlsEl.hidden = true;
  reviewClearBtn.hidden = true;
  // Reset the board to a clean state for the empty pre-import view.
  state.pieces.clear();
  state.hoverArrow = null;
  state.selectedSource = null;
  state.lastResult = null;
  phaseSel.value = "RING_PLACEMENT";
  const whiteRadio = document.querySelector('input[name="side"][value="WHITE"]');
  if (whiteRadio) whiteRadio.checked = true;
  updateDerivedStats();
  updateTurnBadge("WHITE", "RING_PLACEMENT");
  renderResult(null);
}

function exitReviewMode(opts = {}) {
  if (!state.review) return;
  const snap = state.review.baseSnapshot;
  state.review = null;
  if (snap) {
    state.pieces.clear();
    for (const [k, v] of snap.pieces) state.pieces.set(k, v);
    phaseSel.value = snap.phase;
    const sideRadio = document.querySelector(`input[name="side"][value="${snap.sideToMove}"]`);
    if (sideRadio) sideRadio.checked = true;
    state.moveMaker = snap.moveMaker;
    state.lastResult = snap.lastResult;
    state.legalMoves = snap.legalMoves;
    state.gameOver = snap.gameOver;
    state.winner = snap.winner;
    updateDerivedStats();
    updateTurnBadge(snap.sideToMove, snap.phase);
    if (snap.lastResult) renderResult(snap.lastResult); else renderResult(null);
  }
  state.hoverArrow = null;
  state.selectedSource = null;
  moveHistoryEl.innerHTML = "";
  if (!opts.silent) setStatus("Left review mode.", null);
}

function setReviewStatus(text, kind) {
  if (!reviewStatusEl) return;
  if (!text) {
    reviewStatusEl.hidden = true;
    reviewStatusEl.textContent = "";
    reviewStatusEl.className = "review-status";
    return;
  }
  reviewStatusEl.hidden = false;
  reviewStatusEl.textContent = text;
  reviewStatusEl.className = "review-status" + (kind ? " " + kind : "");
}

async function loadBgaGame() {
  if (!state.review) return;
  const raw = (reviewInput.value || "").trim();
  if (!raw) {
    setReviewStatus("Paste a BGA URL or table id first.", "error");
    return;
  }
  reviewImportBtn.disabled = true;
  setReviewStatus("Importing from BGA…", "thinking");
  try {
    const res = await fetch("/api/import_bga", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url_or_table_id: raw }),
    });
    const data = await res.json();
    if (!data.ok) {
      setReviewStatus((data.errors || ["import failed"]).join(" · "), "error");
      return;
    }
    state.review.tableId = data.table_id;
    state.review.metadata = data.metadata;
    state.review.steps = data.steps || [];
    state.review.currentStep = 0;
    renderReviewMetadata(data);
    setReviewStatus(
      data.cached
        ? `Loaded table ${data.table_id} (cached).`
        : `Imported table ${data.table_id} from BGA.`,
      "success",
    );
    reviewMetaEl.hidden = false;
    reviewControlsEl.hidden = false;
    reviewClearBtn.hidden = false;
    // Build the move-history list from the parsed plies so the user can see
    // the whole game at a glance and click any ply to jump there.
    moveHistoryEl.innerHTML = "";
    state.review.steps.forEach((step, i) => {
      if (i === 0) return;  // step 0 is the empty board
      const li = document.createElement("li");
      li.dataset.stepIdx = i;
      li.innerHTML = `<span class="ply">${i}.</span><span>${formatMoveShort(step.move)}</span>`;
      li.style.cursor = "pointer";
      li.addEventListener("click", () => applyReviewStep(i));
      moveHistoryEl.appendChild(li);
    });
    historyBlock.hidden = false;
    await applyReviewStep(1);  // land on after-first-move so the board isn't empty
  } catch (e) {
    setReviewStatus("Request failed: " + e.message, "error");
  } finally {
    reviewImportBtn.disabled = false;
  }
}

function renderReviewMetadata(data) {
  const md = data.metadata || {};
  const players = md.players || [];
  const white = players.find((p) => p.color === "WHITE") || {};
  const black = players.find((p) => p.color === "BLACK") || {};
  const whiteLabel = white.name + (white.rating ? ` · ${white.rating}` : "");
  const blackLabel = black.name + (black.rating ? ` · ${black.rating}` : "");
  reviewWhiteEl.textContent = whiteLabel || "—";
  reviewBlackEl.textContent = blackLabel || "—";
  const result = md.result || {};
  let resultText = "—";
  if (result.winner) {
    resultText = `${result.winner} wins · ${result.score || ""}`.trim();
  } else if (result.score) {
    resultText = `Draw · ${result.score}`;
  }
  reviewResultEl.textContent = resultText;
}

async function applyReviewStep(idx) {
  if (!state.review || !state.review.steps) return;
  const N = state.review.steps.length;
  idx = Math.max(0, Math.min(N - 1, idx));
  state.review.currentStep = idx;
  const step = state.review.steps[idx];

  // Animate the move that landed us here (when stepping forward by one).
  // Skip animation for jumps and step 0.
  const prevSnapshot = new Map(state.pieces);
  const newPiecesMap = new Map();
  for (const p of step.position.pieces) newPiecesMap.set(p.pos, p.piece);

  if (step.move && idx === state.review.currentStep) {
    try {
      await animateMove({
        move: step.move,
        prevPieces: prevSnapshot,
        newPieces: newPiecesMap,
      });
    } catch (_) { /* animation is best-effort */ }
  }

  applyNewPosition(step.position);
  state.hoverArrow = step.move ? moveToArrow(step.move) : null;
  state.selectedSource = null;
  updateReviewNav();
  highlightReviewHistoryEntry(idx);
  render();
  // Evaluate the current position so the analysis panel shows engine
  // commentary at this ply. Don't await — let it stream in.
  evaluateReviewPosition();
}

function updateReviewNav() {
  if (!state.review || !state.review.steps) return;
  const { steps, currentStep } = state.review;
  const totalPlies = steps.length - 1;
  reviewStepCounterEl.textContent = `${currentStep}/${totalPlies}`;
  reviewPrevBtn.disabled = currentStep <= 0;
  reviewNextBtn.disabled = currentStep >= steps.length - 1;
}

function highlightReviewHistoryEntry(idx) {
  for (const li of moveHistoryEl.querySelectorAll("li")) {
    li.classList.toggle("current", parseInt(li.dataset.stepIdx, 10) === idx);
  }
  // Scroll the current ply into view if it's outside the visible window.
  const currentLi = moveHistoryEl.querySelector("li.current");
  if (currentLi && typeof currentLi.scrollIntoView === "function") {
    currentLi.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
}

async function evaluateReviewPosition() {
  if (!state.review || !state.review.steps) return;
  if (!modelSel.value) {
    // Models still loading — defer one tick and retry once.
    setTimeout(evaluateReviewPosition, 250);
    return;
  }
  const step = state.review.steps[state.review.currentStep];
  if (!step || !step.position) return;
  const payload = {
    ...step.position,
    model_id: modelSel.value,
    num_sims: Math.max(0, parseInt(numSims.value, 10) || 0),
    owner_token: ownerToken(),
    top_k: 8,
    evaluation_mode: evalModeEl.value,
    heuristic_weight: parseFloat(heuristicWeightEl.value) || MCTS_DEFAULTS.heuristic_weight,
    c_puct: parseFloat(cPuctEl.value) || MCTS_DEFAULTS.c_puct,
    fpu_reduction: parseFloat(fpuReductionEl.value) || MCTS_DEFAULTS.fpu_reduction,
  };
  try {
    const res = await fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!data.ok) {
      setStatus((data.errors || ["evaluation failed"]).join(" · "), "error");
      renderResult(null);
      return;
    }
    state.lastResult = data;
    state.legalMoves = data.legal_moves || [];
    renderResult(data);
    setStatus(
      `Step ${state.review.currentStep}/${state.review.steps.length - 1} · ` +
      `${data.side_to_move} to move · ${data.num_valid_moves} legal moves`,
      "success",
    );
  } catch (e) {
    setStatus("Request failed: " + e.message, "error");
  }
}

if (reviewImportBtn) reviewImportBtn.addEventListener("click", loadBgaGame);
if (reviewInput) {
  reviewInput.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      ev.preventDefault();
      loadBgaGame();
    }
  });
}
if (reviewPrevBtn) {
  reviewPrevBtn.addEventListener("click", () => {
    if (state.review) applyReviewStep(state.review.currentStep - 1);
  });
}
if (reviewNextBtn) {
  reviewNextBtn.addEventListener("click", () => {
    if (state.review) applyReviewStep(state.review.currentStep + 1);
  });
}
if (reviewClearBtn) {
  reviewClearBtn.addEventListener("click", () => {
    // "Clear" within Review = ditch the loaded game and reset the input,
    // but stay in Review mode so the user can paste another URL.
    state.review.tableId = null;
    state.review.metadata = null;
    state.review.steps = null;
    state.review.currentStep = 0;
    reviewInput.value = "";
    setReviewStatus(null);
    reviewMetaEl.hidden = true;
    reviewControlsEl.hidden = true;
    reviewClearBtn.hidden = true;
    moveHistoryEl.innerHTML = "";
    state.pieces.clear();
    state.hoverArrow = null;
    state.lastResult = null;
    phaseSel.value = "RING_PLACEMENT";
    updateDerivedStats();
    updateTurnBadge("WHITE", "RING_PLACEMENT");
    renderResult(null);
    render();
    reviewInput.focus();
  });
}

// Keyboard nav for review walking. Independent of line-mode's listener so
// they can coexist; ←/→/Esc are handled here only when review is active.
document.addEventListener("keydown", (ev) => {
  if (!state.review || !state.review.steps) return;
  // Don't steal arrow keys from text input — the user might be editing the URL.
  if (document.activeElement && (
      document.activeElement.tagName === "INPUT" ||
      document.activeElement.tagName === "TEXTAREA" ||
      document.activeElement.tagName === "SELECT")) return;
  if (ev.key === "ArrowLeft") {
    ev.preventDefault();
    applyReviewStep(state.review.currentStep - 1);
  } else if (ev.key === "ArrowRight") {
    ev.preventDefault();
    applyReviewStep(state.review.currentStep + 1);
  } else if (ev.key === "Escape") {
    ev.preventDefault();
    setMode("play");
  }
});

// ---------- Boot ----------
applyModeStyling(state.mode);
applyOpponentStyling(state.opponent);
loadModels();
updateDerivedStats();
render();
// Default landing = Play + vs Engine → land users straight in the New Game
// stepper instead of an empty board with no obvious next action. Users who
// want to compose a position click "Set up position" to escape into Setup.
if (state.mode === "play" && state.opponent === "engine" && !state.game) {
  enterGameSetup();
}
