// YINSH analysis board — drag-drop position composer + network evaluation.
//
// Hex geometry is ported from yinsh_ml/viz/board_render.py:
//   screen_x = col_idx * sqrt(3)/2
//   screen_y = (row - 1) - col_idx * 0.5
// Canvas Y is inverted relative to that (down is +y on screen), so we flip
// when going to pixels.

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

// ---------- Geometry ----------
function mathXY(col, row) {
  const colIdx = col.charCodeAt(0) - 65;
  return { x: colIdx * SQRT3_2, y: (row - 1) - colIdx * 0.5 };
}

const BBOX = (() => {
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  for (const { col, row } of ALL_POSITIONS) {
    const { x, y } = mathXY(col, row);
    if (x < xMin) xMin = x;
    if (x > xMax) xMax = x;
    if (y < yMin) yMin = y;
    if (y > yMax) yMax = y;
  }
  return { xMin, xMax, yMin, yMax };
})();

class BoardGeom {
  constructor(canvas) {
    const W = canvas.width, H = canvas.height;
    this.padLeft = 56;
    this.padRight = 24;
    this.padTop = 24;
    this.padBottom = 44;
    const innerW = W - this.padLeft - this.padRight;
    const innerH = H - this.padTop - this.padBottom;
    const xRange = BBOX.xMax - BBOX.xMin;
    const yRange = BBOX.yMax - BBOX.yMin;
    this.scale = Math.min(innerW / xRange, innerH / yRange);
    // center inside the inner rect
    const usedW = xRange * this.scale;
    const usedH = yRange * this.scale;
    this.originX = this.padLeft + (innerW - usedW) / 2;
    this.originY = this.padTop + (innerH - usedH) / 2;
  }
  posToPixel(col, row) {
    const { x, y } = mathXY(col, row);
    return {
      px: this.originX + (x - BBOX.xMin) * this.scale,
      py: this.originY + (BBOX.yMax - y) * this.scale,  // flip Y for canvas
    };
  }
  pixelToNearestPos(px, py) {
    let best = null, bestDist = Infinity;
    for (const { col, row } of ALL_POSITIONS) {
      const { px: x, py: y } = this.posToPixel(col, row);
      const d = (x - px) ** 2 + (y - py) ** 2;
      if (d < bestDist) { bestDist = d; best = { col, row, dist: Math.sqrt(d) }; }
    }
    return best;
  }
}

// ---------- State ----------
const state = {
  pieces: new Map(),         // posKey -> "WHITE_RING" | "BLACK_RING" | "WHITE_MARKER" | "BLACK_MARKER"
  armedTool: null,           // currently selected palette tool
  hoverPos: null,            // {col,row} or null — for drop preview
  hoverArrow: null,          // {from:{col,row}, to:{col,row}} | null — for top-move hover
  lastResult: null,          // last /api/evaluate response
  models: [],
  mode: "setup",             // "setup" | "play"
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
};

// ---------- Rendering ----------
// Palette tuned for the "serious analysis tool" aesthetic — deeper saturated
// blue-gray tile, restrained piece colors with subtle specular highlights,
// doubled drop shadows for piece weight. Matches the visual-style preference
// for deep saturated darks + low-opacity borders + gentle corners.
const COLORS = {
  bg: "#eeece6",
  tile1: "#b4c8d4",          // top of board gradient
  tile2: "#94aab9",          // bottom of board gradient
  tileEdgeOuter: "#6a7e8c",
  tileEdgeInner: "rgba(255, 255, 255, 0.35)",
  grid: "#1a2128",
  gridDot: "#384654",
  label: "#4a5563",
  labelMuted: "#7a8694",
  // Pieces — multi-stop gradients keyed off these. Pure-black/pure-white
  // looks fake; warm-white and cool-charcoal feel like physical objects.
  whiteRingCore: "#f9f6ee",
  whiteRingEdge: "#bfb7a6",
  whiteRingStroke: "#1f1d18",
  blackRingCore: "#34373d",
  blackRingEdge: "#0d0f12",
  blackRingStroke: "#040506",
  ringInnerHole: "#aabccb",   // sits over the board, just slightly darker
  whiteMarkerCore: "#f7f5ed",
  whiteMarkerEdge: "#c3bcab",
  blackMarkerCore: "#2a2d33",
  blackMarkerEdge: "#080a0d",
  markerStroke: "#1a1817",
  specularHighlight: "rgba(255, 255, 255, 0.55)",
  pieceShadowNear: "rgba(15, 20, 28, 0.30)",
  pieceShadowFar: "rgba(15, 20, 28, 0.12)",
  hoverFill: "rgba(37, 99, 235, 0.18)",
  hoverStroke: "#2563eb",
  selectStroke: "#1d4ed8",
  selectGlow: "rgba(37, 99, 235, 0.45)",
  arrowSrc: "#d97706",        // deeper amber than #f59e0b
  arrowDst: "#1d4ed8",        // deeper indigo for crispness on the tile
};

// Off-screen cache for the static board background (tile gradient, noise,
// grid lines, dots, labels). Drawn once at init; composited each frame so
// the noise + labels + grid don't get re-rasterized on every piece move.
let _boardBgCache = null;

function buildBoardBackground(geom, W, H) {
  const off = document.createElement("canvas");
  off.width = W;
  off.height = H;
  const bctx = off.getContext("2d");

  const tilePad = 22;
  const tileX = geom.padLeft - tilePad;
  const tileY = geom.padTop - tilePad;
  const tileW = W - geom.padLeft - geom.padRight + 2 * tilePad;
  const tileH = H - geom.padTop - geom.padBottom + 2 * tilePad;

  // 1. Tile drop shadow (subtle, gives the board lift off the page)
  bctx.save();
  bctx.shadowColor = "rgba(15, 20, 28, 0.18)";
  bctx.shadowBlur = 20;
  bctx.shadowOffsetX = 0;
  bctx.shadowOffsetY = 6;
  bctx.fillStyle = COLORS.tile1;
  roundRect(bctx, tileX, tileY, tileW, tileH, 14);
  bctx.fill();
  bctx.restore();

  // 2. Tile gradient fill
  const grad = bctx.createLinearGradient(0, tileY, 0, tileY + tileH);
  grad.addColorStop(0, COLORS.tile1);
  grad.addColorStop(1, COLORS.tile2);
  bctx.fillStyle = grad;
  roundRect(bctx, tileX, tileY, tileW, tileH, 14);
  bctx.fill();

  // 3. Subtle noise overlay — paper-grain effect. ~5% intensity so it reads
  // as texture rather than dirt. Generated once into ImageData and clipped
  // to the tile rect.
  bctx.save();
  bctx.beginPath();
  roundRect(bctx, tileX + 1, tileY + 1, tileW - 2, tileH - 2, 13);
  bctx.clip();
  const noiseImg = bctx.createImageData(Math.ceil(tileW), Math.ceil(tileH));
  const nd = noiseImg.data;
  for (let i = 0; i < nd.length; i += 4) {
    const v = (Math.random() * 14) | 0;   // 0-13
    nd[i] = nd[i + 1] = nd[i + 2] = v < 7 ? 0 : 255;
    nd[i + 3] = 7;                         // ~3% alpha — barely visible
  }
  bctx.putImageData(noiseImg, tileX, tileY);
  bctx.restore();

  // 4. Inner highlight border + outer edge — doubled border per visual-style
  bctx.strokeStyle = COLORS.tileEdgeInner;
  bctx.lineWidth = 1;
  roundRect(bctx, tileX + 1.5, tileY + 1.5, tileW - 3, tileH - 3, 13);
  bctx.stroke();
  bctx.strokeStyle = COLORS.tileEdgeOuter;
  bctx.lineWidth = 1;
  roundRect(bctx, tileX, tileY, tileW, tileH, 14);
  bctx.stroke();

  // 5. Grid lines — three forward hex axes, restrained opacity
  bctx.strokeStyle = COLORS.grid;
  bctx.lineWidth = 0.85;
  bctx.globalAlpha = 0.42;
  bctx.lineCap = "round";
  const FWD = [[0, 1], [1, 0], [1, 1]];
  for (const { col, row } of ALL_POSITIONS) {
    const a = geom.posToPixel(col, row);
    for (const [dc, dr] of FWD) {
      const nc = String.fromCharCode(col.charCodeAt(0) + dc);
      const nr = row + dr;
      if (!isValidPos(nc, nr)) continue;
      const b = geom.posToPixel(nc, nr);
      bctx.beginPath();
      bctx.moveTo(a.px, a.py);
      bctx.lineTo(b.px, b.py);
      bctx.stroke();
    }
  }
  bctx.globalAlpha = 1;

  // 6. Position dots — small gradient for depth
  for (const { col, row } of ALL_POSITIONS) {
    const { px, py } = geom.posToPixel(col, row);
    const dotGrad = bctx.createRadialGradient(px - 0.6, py - 0.6, 0, px, py, 2.6);
    dotGrad.addColorStop(0, COLORS.gridDot);
    dotGrad.addColorStop(1, "rgba(20, 28, 40, 0.6)");
    bctx.fillStyle = dotGrad;
    bctx.beginPath();
    bctx.arc(px, py, 2.4, 0, Math.PI * 2);
    bctx.fill();
  }

  // 7. Axis labels — serif, two-tone (column letters muted, rows muted)
  bctx.font = "600 13px 'Iowan Old Style', 'Palatino Linotype', Palatino, Georgia, serif";
  bctx.fillStyle = COLORS.label;
  bctx.textAlign = "center";
  bctx.textBaseline = "middle";
  for (const col of COLUMNS) {
    const rows = VALID_POSITIONS[col];
    const bottomRow = rows[0];
    const { px, py } = geom.posToPixel(col, bottomRow);
    bctx.fillText(col, px, py + 24);
  }
  bctx.textAlign = "right";
  for (let row = 1; row <= 11; row++) {
    let leftCol = null;
    for (const col of COLUMNS) {
      if (isValidPos(col, row)) { leftCol = col; break; }
    }
    if (!leftCol) continue;
    const { px, py } = geom.posToPixel(leftCol, row);
    bctx.fillText(String(row), px - 20, py);
  }

  return off;
}

function drawBoard(ctx, geom) {
  const W = ctx.canvas.width, H = ctx.canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!_boardBgCache) _boardBgCache = buildBoardBackground(geom, W, H);
  ctx.drawImage(_boardBgCache, 0, 0);
}

function drawPieces(ctx, geom) {
  const cellSize = geom.scale; // approximate spacing of one hex unit
  const ringOuter = cellSize * 0.40;
  const ringInner = cellSize * 0.22;
  const markerR = cellSize * 0.20;

  for (const [key, piece] of state.pieces) {
    const col = key[0];
    const row = parseInt(key.slice(1), 10);
    const { px, py } = geom.posToPixel(col, row);
    if (piece === "WHITE_RING") drawRing(ctx, px, py, ringOuter, ringInner, "white");
    else if (piece === "BLACK_RING") drawRing(ctx, px, py, ringOuter, ringInner, "black");
    else if (piece === "WHITE_MARKER") drawMarker(ctx, px, py, markerR, "white");
    else if (piece === "BLACK_MARKER") drawMarker(ctx, px, py, markerR, "black");
  }
}

function drawRing(ctx, x, y, outerR, innerR, color) {
  const isWhite = color === "white";

  // -- 1. Doubled drop shadow: near (tight, denser) + far (soft, atmospheric)
  ctx.save();
  ctx.shadowColor = COLORS.pieceShadowFar;
  ctx.shadowBlur = outerR * 0.9;
  ctx.shadowOffsetX = 0;
  ctx.shadowOffsetY = outerR * 0.20;
  // Render an invisible disk to project the shadow without drawing the body
  ctx.fillStyle = "rgba(0,0,0,0.001)";
  ctx.beginPath(); ctx.arc(x, y, outerR, 0, Math.PI * 2); ctx.fill();
  ctx.restore();

  ctx.save();
  ctx.shadowColor = COLORS.pieceShadowNear;
  ctx.shadowBlur = outerR * 0.32;
  ctx.shadowOffsetX = 0;
  ctx.shadowOffsetY = outerR * 0.10;

  // -- 2. Outer body — multi-stop radial gradient with specular highlight
  const coreColor = isWhite ? COLORS.whiteRingCore : COLORS.blackRingCore;
  const edgeColor = isWhite ? COLORS.whiteRingEdge : COLORS.blackRingEdge;
  const grad = ctx.createRadialGradient(
    x - outerR * 0.32, y - outerR * 0.34, outerR * 0.08,
    x, y, outerR * 1.02,
  );
  if (isWhite) {
    grad.addColorStop(0, "#ffffff");
    grad.addColorStop(0.25, coreColor);
    grad.addColorStop(0.85, "#d8d2c1");
    grad.addColorStop(1, edgeColor);
  } else {
    grad.addColorStop(0, "#52555c");
    grad.addColorStop(0.30, coreColor);
    grad.addColorStop(0.85, "#1c1f24");
    grad.addColorStop(1, edgeColor);
  }
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(x, y, outerR, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  // -- 3. Outer edge stroke (thin, off-black for both colors so the rings
  //       feel like they share a material category — chess.com-like discipline)
  ctx.strokeStyle = isWhite ? COLORS.whiteRingStroke : COLORS.blackRingStroke;
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.arc(x, y, outerR, 0, Math.PI * 2);
  ctx.stroke();

  // -- 4. Inner hole — punched-through effect: gradient from a slightly
  //       darker board-tile shade at center to the ring's body color at the
  //       inner edge. Looks like a real hole, not a painted disc.
  const holeGrad = ctx.createRadialGradient(
    x - innerR * 0.2, y - innerR * 0.2, 0,
    x, y, innerR,
  );
  holeGrad.addColorStop(0, "#9aacba");
  holeGrad.addColorStop(0.7, COLORS.ringInnerHole);
  holeGrad.addColorStop(1, isWhite ? "#a8b8c5" : "#7e8c98");
  ctx.fillStyle = holeGrad;
  ctx.beginPath();
  ctx.arc(x, y, innerR, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = isWhite ? COLORS.whiteRingStroke : COLORS.blackRingStroke;
  ctx.lineWidth = 0.9;
  ctx.beginPath();
  ctx.arc(x, y, innerR, 0, Math.PI * 2);
  ctx.stroke();

  // -- 5. Specular highlight — small crescent at upper-left for both colors,
  //       restrained alpha so it reads as material polish, not glare
  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, outerR - 1, 0, Math.PI * 2);
  ctx.arc(x, y, innerR + 1, 0, Math.PI * 2, true);
  ctx.clip("evenodd");
  const specGrad = ctx.createRadialGradient(
    x - outerR * 0.40, y - outerR * 0.45, 0,
    x - outerR * 0.40, y - outerR * 0.45, outerR * 0.55,
  );
  specGrad.addColorStop(0, isWhite ? "rgba(255,255,255,0.75)" : "rgba(255,255,255,0.32)");
  specGrad.addColorStop(0.7, "rgba(255,255,255,0.04)");
  specGrad.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = specGrad;
  ctx.fillRect(x - outerR, y - outerR, outerR * 2, outerR * 2);
  ctx.restore();
}

function drawMarker(ctx, x, y, r, color) {
  const isWhite = color === "white";

  // Soft drop shadow (single — markers sit lower than rings)
  ctx.save();
  ctx.shadowColor = COLORS.pieceShadowNear;
  ctx.shadowBlur = r * 0.55;
  ctx.shadowOffsetX = 0;
  ctx.shadowOffsetY = r * 0.18;

  const coreColor = isWhite ? COLORS.whiteMarkerCore : COLORS.blackMarkerCore;
  const edgeColor = isWhite ? COLORS.whiteMarkerEdge : COLORS.blackMarkerEdge;
  const grad = ctx.createRadialGradient(
    x - r * 0.30, y - r * 0.32, r * 0.05,
    x, y, r * 1.02,
  );
  if (isWhite) {
    grad.addColorStop(0, "#ffffff");
    grad.addColorStop(0.4, coreColor);
    grad.addColorStop(1, edgeColor);
  } else {
    grad.addColorStop(0, "#484c54");
    grad.addColorStop(0.4, coreColor);
    grad.addColorStop(1, edgeColor);
  }
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  ctx.strokeStyle = COLORS.markerStroke;
  ctx.lineWidth = 0.9;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.stroke();

  // Small specular for markers too (subtler than rings)
  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, r - 0.5, 0, Math.PI * 2);
  ctx.clip();
  const specGrad = ctx.createRadialGradient(
    x - r * 0.35, y - r * 0.40, 0,
    x - r * 0.35, y - r * 0.40, r * 0.5,
  );
  specGrad.addColorStop(0, isWhite ? "rgba(255,255,255,0.6)" : "rgba(255,255,255,0.22)");
  specGrad.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = specGrad;
  ctx.fillRect(x - r, y - r, r * 2, r * 2);
  ctx.restore();
}

function drawHover(ctx, geom) {
  // Setup mode shows hover on every armed tool drop target.
  if (state.mode === "setup" && state.hoverPos && state.armedTool) {
    const { col, row } = state.hoverPos;
    const { px, py } = geom.posToPixel(col, row);
    ctx.fillStyle = COLORS.hoverFill;
    ctx.strokeStyle = COLORS.hoverStroke;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(px, py, geom.scale * 0.38, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
}

function drawSelection(ctx, geom) {
  if (state.mode !== "play" || !state.selectedSource) return;
  const { col, row } = state.selectedSource;
  const { px, py } = geom.posToPixel(col, row);

  // Soft outer glow for the selected piece
  ctx.save();
  ctx.shadowColor = COLORS.selectGlow;
  ctx.shadowBlur = geom.scale * 0.45;
  ctx.shadowOffsetX = 0;
  ctx.shadowOffsetY = 0;
  ctx.strokeStyle = COLORS.selectStroke;
  ctx.lineWidth = 2.6;
  ctx.beginPath();
  ctx.arc(px, py, geom.scale * 0.48, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();

  // Legal destinations — small filled dots with subtle gradient
  const dests = legalDestinationsFrom(state.selectedSource);
  for (const d of dests) {
    const pt = geom.posToPixel(d.col, d.row);
    const r = geom.scale * 0.12;
    const grad = ctx.createRadialGradient(pt.px - r * 0.3, pt.py - r * 0.3, 0, pt.px, pt.py, r);
    grad.addColorStop(0, "#60a5fa");
    grad.addColorStop(1, COLORS.selectStroke);
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(pt.px, pt.py, r, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawArrow(ctx, geom) {
  if (!state.hoverArrow) return;
  const { from, to } = state.hoverArrow;
  if (!from) return;
  const a = geom.posToPixel(from.col, from.row);

  // Soft drop shadow on all annotation strokes — makes them sit cleanly
  // on top of pieces without competing visually.
  ctx.save();
  ctx.shadowColor = "rgba(15, 20, 28, 0.25)";
  ctx.shadowBlur = 6;
  ctx.shadowOffsetX = 0;
  ctx.shadowOffsetY = 1;

  if (!to) {
    // Single-position move (placement / removal): ring of accent
    ctx.strokeStyle = COLORS.arrowDst;
    ctx.lineWidth = 2.6;
    ctx.beginPath();
    ctx.arc(a.px, a.py, geom.scale * 0.46, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
    return;
  }

  const b = geom.posToPixel(to.col, to.row);
  // From circle (origin marker — amber)
  ctx.strokeStyle = COLORS.arrowSrc;
  ctx.lineWidth = 2.4;
  ctx.beginPath();
  ctx.arc(a.px, a.py, geom.scale * 0.46, 0, Math.PI * 2);
  ctx.stroke();

  // Shaft
  const dx = b.px - a.px, dy = b.py - a.py;
  const len = Math.hypot(dx, dy);
  const ux = dx / len, uy = dy / len;
  const startOffset = geom.scale * 0.46;
  const endOffset = geom.scale * 0.46;
  const sx = a.px + ux * startOffset, sy = a.py + uy * startOffset;
  const ex = b.px - ux * endOffset, ey = b.py - uy * endOffset;
  ctx.strokeStyle = COLORS.arrowDst;
  ctx.lineWidth = 3.0;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(sx, sy);
  ctx.lineTo(ex, ey);
  ctx.stroke();

  // Arrowhead (filled triangle)
  const ah = 11, aw = 7.5;
  ctx.fillStyle = COLORS.arrowDst;
  ctx.beginPath();
  ctx.moveTo(ex, ey);
  ctx.lineTo(ex - ux * ah - uy * aw, ey - uy * ah + ux * aw);
  ctx.lineTo(ex - ux * ah + uy * aw, ey - uy * ah - ux * aw);
  ctx.closePath();
  ctx.fill();

  // Destination circle
  ctx.strokeStyle = COLORS.arrowDst;
  ctx.lineWidth = 2.4;
  ctx.beginPath();
  ctx.arc(b.px, b.py, geom.scale * 0.46, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function render() {
  drawBoard(ctx, geom);
  drawPieces(ctx, geom);
  drawHover(ctx, geom);
  drawSelection(ctx, geom);
  drawArrow(ctx, geom);
}

// ---------- Canvas + geom ----------
const canvas = document.getElementById("board");
const ctx = canvas.getContext("2d");
const geom = new BoardGeom(canvas);

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
  // Was the drop over the canvas?
  const rect = canvas.getBoundingClientRect();
  if (ev.clientX >= rect.left && ev.clientX <= rect.right &&
      ev.clientY >= rect.top && ev.clientY <= rect.bottom) {
    handlePlace(ev.clientX - rect.left, ev.clientY - rect.top, drag.tool);
  }
  endDrag();
});

// Canvas mouse: hover preview (setup) / select-and-move (play).
canvas.addEventListener("mousemove", (ev) => {
  const rect = canvas.getBoundingClientRect();
  const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
  const near = geom.pixelToNearestPos(x, y);
  if (state.mode === "setup") {
    if (state.armedTool && near && near.dist < geom.scale * 0.6) {
      state.hoverPos = { col: near.col, row: near.row };
    } else {
      state.hoverPos = null;
    }
  } else {
    state.hoverPos = null;  // Play mode: no drop-cursor preview
  }
  render();
});

canvas.addEventListener("mouseleave", () => {
  state.hoverPos = null;
  render();
});

canvas.addEventListener("click", (ev) => {
  // While walking a PV the board is display-only — clicks are ignored.
  if (state.lineMode) return;
  const rect = canvas.getBoundingClientRect();
  const cx = ev.clientX - rect.left, cy = ev.clientY - rect.top;
  if (state.mode === "setup") {
    if (!state.armedTool) return;
    handlePlace(cx, cy, state.armedTool);
    return;
  }
  handlePlayClick(cx, cy);
});

canvas.addEventListener("contextmenu", (ev) => {
  ev.preventDefault();
  if (state.mode !== "setup") return;   // no erase in play mode
  const rect = canvas.getBoundingClientRect();
  const near = geom.pixelToNearestPos(ev.clientX - rect.left, ev.clientY - rect.top);
  if (near && near.dist < geom.scale * 0.5) {
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
  const near = geom.pixelToNearestPos(px, py);
  if (!near || near.dist > geom.scale * 0.55) {
    // Click on empty area → deselect
    state.selectedSource = null;
    render();
    return;
  }
  const { col, row } = near;
  const posStr = `${col}${row}`;
  const pieceHere = state.pieces.get(posKey(col, row));
  const side = state.lastResult ? state.lastResult.side_to_move : currentSide();
  const sideRing = side === "WHITE" ? "WHITE_RING" : "BLACK_RING";

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
  if (state.lastResult && state.lastResult.phase === "RING_PLACEMENT" && !pieceHere) {
    const placeMove = state.legalMoves.find(
      (m) => m.type === "PLACE_RING" && m.source === posStr,
    );
    if (placeMove) {
      applyMove({ type: "PLACE_RING", source: posStr });
      return;
    }
  }

  // RING_REMOVAL: clicking own ring removes it
  if (state.lastResult && state.lastResult.phase === "RING_REMOVAL" && pieceHere === sideRing) {
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
    // Push history before mutating.
    state.history.push({
      position: currentPositionPayload(),
      move: data.applied_move,
    });
    updateUndoBtn();
    appendHistoryEntry(data.applied_move);
    // Apply new position to local state.
    applyNewPosition(data.new_position);
    state.legalMoves = data.legal_moves || [];
    state.gameOver = data.game_over;
    state.winner = data.winner;
    if (state.gameOver) {
      setStatus(
        `Game over — ${state.winner ? state.winner + " wins" : "draw"}.`,
        "success",
      );
      // Clear stale evaluation UI from the previous turn — top moves and
      // value bars from the pre-game-over position are misleading once
      // there are no legal moves left.
      state.legalMoves = [];
      state.lastResult = null;
      renderResult(null);
      if (state.winner) {
        updateTurnBadge(state.winner, "GAME_OVER");
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
  };
}

function handlePlace(px, py, tool) {
  const near = geom.pixelToNearestPos(px, py);
  if (!near || near.dist > geom.scale * 0.6) return;
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
const resetAdvancedBtn = $("reset-advanced");

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

clearBtn.addEventListener("click", () => {
  state.pieces.clear();
  state.hoverArrow = null;
  state.lastResult = null;
  updateDerivedStats();
  renderResult(null);
  render();
});

// Phase change can flip RING_PLACEMENT → MAIN_GAME, which changes derived score.
phaseSel.addEventListener("change", updateDerivedStats);

evalBtn.addEventListener("click", evaluate);

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
    if (list.length === 0) {
      const opt = document.createElement("option");
      opt.textContent = "(no models found in models/)";
      opt.value = "";
      modelSel.appendChild(opt);
      return;
    }
    for (const m of list) {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = `${m.label} · ${m.checkpoint}`;
      modelSel.appendChild(opt);
    }
    // Prefer yngine_volume_15ch_pretrain if present.
    const preferred = list.find((m) => m.label === "yngine_volume_15ch_pretrain");
    if (preferred) modelSel.value = preferred.id;
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
    setStatus(
      `${data.mode === "mcts" ? "MCTS" : "Network policy"} · ` +
      `${data.side_to_move} to move${captureNote} · ${data.num_valid_moves} legal moves`,
      "success",
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
  valueLabel.textContent = `Search avg (${data.side_to_move})${modeTag}`;
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
  data.top_moves.forEach((mv, i) => {
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
      // Don't trigger the row-click action if the user clicked the step-into
      // button inside it.
      if (ev.target.closest(".step-into")) return;
      if (state.mode !== "play") return;
      if (state.busy) return;
      if (state.lineMode) return;
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

  // Auto-show the top move arrow on result.
  if (data.top_moves.length > 0) {
    state.hoverArrow = moveToArrow(data.top_moves[0]);
  }
  render();
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
canvas.addEventListener("contextmenu", () => updateDerivedStats());

// ---------- Mode toggle ----------
document.querySelectorAll(".mode-btn").forEach((btn) => {
  btn.addEventListener("click", () => setMode(btn.dataset.mode));
});

async function setMode(mode) {
  if (state.mode === mode) return;
  state.mode = mode;
  document.querySelectorAll(".mode-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.mode === mode);
    b.setAttribute("aria-selected", b.dataset.mode === mode ? "true" : "false");
  });
  paletteBlock.classList.toggle("hidden-in-play", mode === "play");
  playControls.hidden = mode !== "play";
  historyBlock.hidden = mode !== "play";
  // The side radio in the sidebar is a setup control; in Play mode it's
  // display-only (the source of truth is whatever side_to_move the server
  // last returned). Lock it visually so the user doesn't mistake it for
  // an editable knob.
  if (sideFieldEl) sideFieldEl.classList.toggle("locked", mode === "play");
  state.selectedSource = null;
  if (mode === "play") {
    boardHint.textContent = "Click a ring of the side to move, then click a destination. Or click any top-move row to apply.";
    setArmedTool(null);
    // Need legal moves to play. Only auto-evaluate if a real position exists;
    // an empty board would just produce a "no legal moves" error which is
    // confusing on first switch.
    if (!state.lastResult && state.pieces.size > 0) {
      await evaluate({ skipPositionCheck: true });
    } else if (!state.lastResult) {
      setStatus("Place rings/markers in Setup mode, then come back to Play.", null);
    }
  } else {
    boardHint.textContent = "Drag a piece from the palette onto a board intersection. Right-click to erase.";
    // Setup mode invalidates play history (the position can be edited).
    state.history = [];
    state.gameOver = false;
    state.winner = null;
    updateUndoBtn();
    moveHistoryEl.innerHTML = "";
  }
  render();
}

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

// ---------- Boot ----------
loadModels();
updateDerivedStats();
render();
