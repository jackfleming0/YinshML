// board3d.js — three.js rendering module for the YINSH analysis board.
//
// Replaces the previous 2D canvas renderer. Exposes a small imperative API
// that app.js calls (initBoard3D, setPieces, setHover, setSelection,
// setArrow, pixelToBoardPos). Visual idiom matches BoardGameArena / the
// Rio Grande physical set: cool icy-blue translucent board with a hex
// grid printed on it, terrazzo-speckled cream/dark discs with a teal rim,
// camera tiltable from top-down to ~45° with zoom.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ---------- Logical coordinates ----------
// Hex math is the same as the 2D version — we reuse the layout from
// yinsh_ml/viz/board_render.py.
const COLUMNS = "ABCDEFGHIJK".split("");
const SQRT3_2 = Math.sqrt(3) / 2;
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
function isValidPos(col, row) {
  const rows = VALID_POSITIONS[col];
  return rows ? rows.includes(row) : false;
}
function mathXY(col, row) {
  const colIdx = col.charCodeAt(0) - 65;
  return { x: colIdx * SQRT3_2, y: (row - 1) - colIdx * 0.5 };
}

// World-space bounds of the hex lattice — pre-computed so we can size
// the board slab and center the scene around it.
const BBOX = (() => {
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  for (const { col, row } of ALL_POSITIONS) {
    const { x, y } = mathXY(col, row);
    if (x < xMin) xMin = x; if (x > xMax) xMax = x;
    if (y < yMin) yMin = y; if (y > yMax) yMax = y;
  }
  return { xMin, xMax, yMin, yMax };
})();
const BBOX_CX = (BBOX.xMin + BBOX.xMax) / 2;
const BBOX_CY = (BBOX.yMin + BBOX.yMax) / 2;

// Logical → world transform. We render with Y-up in three.js, and the
// board lies on the X-Z plane. So board-x → world-X, board-y → world-Z
// (negated so "row 1" is at the back / negative Z when looking down).
function posToWorld(col, row) {
  const { x, y } = mathXY(col, row);
  return new THREE.Vector3(x - BBOX_CX, 0, -(y - BBOX_CY));
}

// ---------- Materials & colors ----------
const PALETTE = {
  page: 0x2b2f33,
  boardTop: 0xc7d9e3,     // light icy blue (center)
  boardEdge: 0x91a8b8,    // deeper at corners
  boardSide: 0x6b7e8c,    // visible edge of the slab
  grid: 0x1a2128,         // hex lines printed on the board
  hash: 0x1a2128,         // × hash marks at intersections
  pieceLight: 0xf7f5ee,   // bright cream-white disc face
  pieceLightSpeckle: 0x161a1f,
  pieceDark: 0x161a1f,    // near-black disc face
  pieceDarkSpeckle: 0xe2e6ec,
  pieceRim: 0x2aa7c0,     // teal — the YINSH signature visual
  hover: 0xffd766,
  selection: 0xffb347,
  arrowSrc: 0xd97706,
  arrowDst: 0x2aa7c0,
};

// Build a CanvasTexture with terrazzo-style speckle for piece faces. One
// texture per color is shared across all pieces of that color (no per-piece
// allocation).
function buildSpeckleTexture(baseHex, fleckHex, size = 256, density = 0.0075) {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d");
  // Solid base
  ctx.fillStyle = "#" + baseHex.toString(16).padStart(6, "0");
  ctx.fillRect(0, 0, size, size);
  // Very subtle radial highlight only — no dark vignette. The disc reads
  // bright; volume comes from the doubled drop shadow + scene lighting.
  const vg = ctx.createRadialGradient(size * 0.4, size * 0.35, 0, size * 0.5, size * 0.5, size * 0.7);
  vg.addColorStop(0, "rgba(255, 255, 255, 0.06)");
  vg.addColorStop(1, "rgba(0, 0, 0, 0.04)");
  ctx.fillStyle = vg;
  ctx.fillRect(0, 0, size, size);
  // Terrazzo flecks — bold, varied size, deterministic per build but
  // sufficiently dense that the pattern reads as "spackled plastic."
  const fleckRGB = `${(fleckHex >> 16) & 0xff}, ${(fleckHex >> 8) & 0xff}, ${fleckHex & 0xff}`;
  const total = Math.floor(size * size * density);
  for (let i = 0; i < total; i++) {
    const x = Math.random() * size;
    const y = Math.random() * size;
    const r = 0.6 + Math.random() * 2.4;
    const alpha = 0.45 + Math.random() * 0.35;
    ctx.fillStyle = `rgba(${fleckRGB}, ${alpha.toFixed(3)})`;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }
  // Smaller / softer flecks for visual depth
  for (let i = 0; i < total * 1.5; i++) {
    const x = Math.random() * size;
    const y = Math.random() * size;
    const r = 0.3 + Math.random() * 0.9;
    const alpha = 0.15 + Math.random() * 0.20;
    ctx.fillStyle = `rgba(${fleckRGB}, ${alpha.toFixed(3)})`;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

// ---------- Module state ----------
const M = {
  scene: null,
  camera: null,
  renderer: null,
  controls: null,
  raycaster: new THREE.Raycaster(),
  pointer: new THREE.Vector2(),
  boardMesh: null,                 // raycast target
  boardGroup: null,                // board + grid + hashes
  piecesGroup: null,               // parented to scene; child of board for tilt-following
  arrowsGroup: null,
  hoverMesh: null,
  selectionMesh: null,
  pieces: new Map(),               // key "E5" → Object3D
  textures: {},                    // speckle textures, lazy-built
  container: null,
  size: { w: 760, h: 760 },
  raycastReady: false,
};

// ---------- Public API ----------

export function initBoard3D(container) {
  M.container = container;
  const { clientWidth, clientHeight } = container;
  M.size.w = clientWidth || 760;
  M.size.h = clientHeight || 760;

  // Scene
  M.scene = new THREE.Scene();
  M.scene.background = new THREE.Color(PALETTE.page);

  // Camera — orthographic-like isometric default. Frustum tuned so the
  // whole board sits comfortably in view with a tilt of ~25°.
  M.camera = new THREE.PerspectiveCamera(28, M.size.w / M.size.h, 0.1, 100);
  M.camera.position.set(0, 18, 12);
  M.camera.lookAt(0, 0, 0);
  // Snapshot the default view so resetView() can tween back to it.
  M.defaultCameraPos = M.camera.position.clone();
  M.defaultTarget = new THREE.Vector3(0, 0, 0);

  // Renderer
  M.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  M.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  M.renderer.setSize(M.size.w, M.size.h);
  M.renderer.outputColorSpace = THREE.SRGBColorSpace;
  M.renderer.shadowMap.enabled = true;
  M.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(M.renderer.domElement);

  // Controls — constrained: pitch 0..45°, no roll, dolly limits, no pan
  M.controls = new OrbitControls(M.camera, M.renderer.domElement);
  M.controls.enableDamping = true;
  // Damping factor: higher = stops faster. Default 0.05 felt too loose;
  // 0.15 reads as "the camera has weight but settles promptly."
  M.controls.dampingFactor = 0.15;
  M.controls.enablePan = false;
  M.controls.minPolarAngle = 0;                  // top-down
  M.controls.maxPolarAngle = Math.PI * 0.28;     // ~50° from straight-down
  M.controls.minDistance = 12;
  M.controls.maxDistance = 28;
  M.controls.target.set(0, 0, 0);
  // Lower rotateSpeed so each drag is more deliberate — board feels less
  // skittish, closer to the BGA "this is heavy furniture" feel.
  M.controls.rotateSpeed = 0.40;
  M.controls.zoomSpeed = 0.65;
  // Right-click is reserved for the app (erase in setup mode) — don't let
  // OrbitControls capture it.
  M.controls.mouseButtons = {
    LEFT: THREE.MOUSE.ROTATE,
    MIDDLE: THREE.MOUSE.DOLLY,
    RIGHT: null,
  };

  // Lighting — key + fill + bottom bounce, soft shadows
  const ambient = new THREE.AmbientLight(0xffffff, 0.78);
  M.scene.add(ambient);

  const key = new THREE.DirectionalLight(0xffffff, 0.85);
  key.position.set(6, 14, 8);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.camera.left = -12;
  key.shadow.camera.right = 12;
  key.shadow.camera.top = 12;
  key.shadow.camera.bottom = -12;
  key.shadow.camera.near = 1;
  key.shadow.camera.far = 40;
  key.shadow.radius = 4;
  key.shadow.bias = -0.0005;
  M.scene.add(key);

  const fill = new THREE.DirectionalLight(0xc0d4e0, 0.30);
  fill.position.set(-8, 6, -4);
  M.scene.add(fill);

  const rim = new THREE.DirectionalLight(0xffffff, 0.18);
  rim.position.set(0, 4, -10);
  M.scene.add(rim);

  // Groups
  M.boardGroup = new THREE.Group();
  M.piecesGroup = new THREE.Group();
  M.arrowsGroup = new THREE.Group();
  M.scene.add(M.boardGroup);
  M.scene.add(M.piecesGroup);
  M.scene.add(M.arrowsGroup);

  // Build static board geometry
  _buildBoard();
  _buildHoverMesh();
  _buildSelectionMesh();

  // Speckle textures (built once, reused)
  M.textures.lightFace = buildSpeckleTexture(PALETTE.pieceLight, PALETTE.pieceLightSpeckle);
  M.textures.darkFace = buildSpeckleTexture(PALETTE.pieceDark, PALETTE.pieceDarkSpeckle);

  // Render loop
  function loop() {
    M.controls.update();
    M.renderer.render(M.scene, M.camera);
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);

  // Resize
  window.addEventListener("resize", () => _onResize());
  _onResize();

  M.raycastReady = true;
  return {
    domElement: M.renderer.domElement,
  };
}

export function setPieces(pieceMap) {
  // Reconcile state.pieces (Map<"E5","WHITE_RING">) with the scene graph.
  // For now: nuke and rebuild — fine at <30 pieces, swap to diffing later
  // if we see jank.
  for (const obj of M.pieces.values()) {
    M.piecesGroup.remove(obj);
    _disposeObject(obj);
  }
  M.pieces.clear();

  for (const [key, kind] of pieceMap) {
    const col = key[0];
    const row = parseInt(key.slice(1), 10);
    if (!isValidPos(col, row)) continue;
    const mesh = _buildPiece(kind);
    const world = posToWorld(col, row);
    mesh.position.copy(world);
    M.piecesGroup.add(mesh);
    M.pieces.set(key, mesh);
  }
}

export function setHover(col, row) {
  if (!M.hoverMesh) return;
  if (!col || row == null || !isValidPos(col, row)) {
    M.hoverMesh.visible = false;
    return;
  }
  const w = posToWorld(col, row);
  M.hoverMesh.position.set(w.x, 0.02, w.z);
  M.hoverMesh.visible = true;
}

export function setSelection(col, row) {
  if (!M.selectionMesh) return;
  if (!col || row == null || !isValidPos(col, row)) {
    M.selectionMesh.visible = false;
    return;
  }
  const w = posToWorld(col, row);
  M.selectionMesh.position.set(w.x, 0.025, w.z);
  M.selectionMesh.visible = true;
}

export function setArrow(srcPos, dstPos) {
  clearArrow();
  if (!srcPos) return;
  const sCol = srcPos[0], sRow = parseInt(srcPos.slice(1), 10);
  if (!isValidPos(sCol, sRow)) return;
  const a = posToWorld(sCol, sRow);
  a.y = 0.30;
  const arrowGroup = new THREE.Group();
  // Source disc — always rendered (the origin highlight, regardless of move type)
  const srcGeom = new THREE.RingGeometry(0.30, 0.38, 40);
  const srcMat = new THREE.MeshBasicMaterial({
    color: PALETTE.arrowSrc,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.95,
  });
  const srcDisc = new THREE.Mesh(srcGeom, srcMat);
  srcDisc.rotation.x = -Math.PI / 2;
  srcDisc.position.set(a.x, 0.020, a.z);
  arrowGroup.add(srcDisc);

  // Shaft + head — only when there's a destination (MOVE_RING, REMOVE_MARKERS)
  if (dstPos) {
    const dCol = dstPos[0], dRow = parseInt(dstPos.slice(1), 10);
    if (isValidPos(dCol, dRow)) {
      const b = posToWorld(dCol, dRow);
      b.y = 0.30;
      const shaftDir = new THREE.Vector3().subVectors(b, a);
      const shaftLen = shaftDir.length();
      const shaftMat = new THREE.MeshBasicMaterial({
        color: PALETTE.arrowDst,
        transparent: true,
        opacity: 0.90,
      });
      const shaftGeom = new THREE.CylinderGeometry(0.055, 0.055, shaftLen * 0.78, 16);
      const shaft = new THREE.Mesh(shaftGeom, shaftMat);
      shaft.position.copy(new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5));
      shaft.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        shaftDir.clone().normalize(),
      );
      arrowGroup.add(shaft);

      const headGeom = new THREE.ConeGeometry(0.16, 0.36, 20);
      const head = new THREE.Mesh(headGeom, shaftMat);
      const headPos = a.clone().lerp(b, 0.90);
      head.position.copy(headPos);
      head.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        shaftDir.clone().normalize(),
      );
      arrowGroup.add(head);
    }
  }
  M.arrowsGroup.add(arrowGroup);
}

export function clearArrow() {
  for (let i = M.arrowsGroup.children.length - 1; i >= 0; i--) {
    const c = M.arrowsGroup.children[i];
    M.arrowsGroup.remove(c);
    _disposeObject(c);
  }
}

export function pixelToBoardPos(clientX, clientY) {
  if (!M.raycastReady) return null;
  const rect = M.renderer.domElement.getBoundingClientRect();
  M.pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
  M.pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;
  M.raycaster.setFromCamera(M.pointer, M.camera);
  const hits = M.raycaster.intersectObject(M.boardMesh, false);
  if (!hits.length) return null;
  const p = hits[0].point;
  // World → math (invert posToWorld)
  const mathX = p.x + BBOX_CX;
  const mathY = -p.z + BBOX_CY;
  // Find the nearest valid intersection by exhaustive search (85 cells)
  let best = null, bestD2 = Infinity;
  for (const { col, row } of ALL_POSITIONS) {
    const { x, y } = mathXY(col, row);
    const d2 = (x - mathX) ** 2 + (y - mathY) ** 2;
    if (d2 < bestD2) { bestD2 = d2; best = { col, row }; }
  }
  // Snap radius: ~0.4 hex-unit so clicks well outside cells return null
  if (bestD2 > 0.4 * 0.4) return null;
  return best;
}

export function resize(w, h) {
  M.size.w = w; M.size.h = h;
  _onResize();
}

// Smoothly tween the camera back to the default isometric view. Cancels any
// in-flight reset.
let _resetTweenId = null;
export function resetView(durationMs = 450) {
  if (_resetTweenId) cancelAnimationFrame(_resetTweenId);
  const startPos = M.camera.position.clone();
  const startTarget = M.controls.target.clone();
  const endPos = M.defaultCameraPos.clone();
  const endTarget = M.defaultTarget.clone();
  const t0 = performance.now();
  // Stop any orbit inertia so the tween doesn't fight residual damping.
  M.controls.update();
  function step(now) {
    const t = Math.min(1, (now - t0) / durationMs);
    // ease-out cubic — fast start, gentle landing
    const k = 1 - Math.pow(1 - t, 3);
    M.camera.position.lerpVectors(startPos, endPos, k);
    M.controls.target.lerpVectors(startTarget, endTarget, k);
    M.controls.update();
    if (t < 1) {
      _resetTweenId = requestAnimationFrame(step);
    } else {
      _resetTweenId = null;
    }
  }
  _resetTweenId = requestAnimationFrame(step);
}

// ---------- Internals ----------

function _onResize() {
  const w = M.size.w, h = M.size.h;
  M.camera.aspect = w / h;
  M.camera.updateProjectionMatrix();
  M.renderer.setSize(w, h);
}

function _buildBoard() {
  // Board slab — a thin BoxGeometry sized to comfortably contain the
  // hex lattice with margin for off-board reserve rings.
  const padX = 1.6, padY = 1.6;
  const slabW = (BBOX.xMax - BBOX.xMin) + padX * 2;
  const slabD = (BBOX.yMax - BBOX.yMin) + padY * 2;
  const slabH = 0.18;

  // Vertex-colored material — we shade the corners deeper so the surface
  // reads as a polished laminate with an atmospheric vignette, not a flat
  // pastel slab.
  const slabGeom = new THREE.BoxGeometry(slabW, slabH, slabD, 24, 1, 24);
  _applyBoardVignetteColors(slabGeom, slabW, slabD);
  const slabMat = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.55,
    metalness: 0.05,
  });
  const slab = new THREE.Mesh(slabGeom, slabMat);
  slab.receiveShadow = true;
  slab.position.y = -slabH / 2;     // top surface at y=0
  M.boardGroup.add(slab);
  M.boardMesh = slab;

  // Hex grid — three forward axes, printed-on-board feel
  const gridLines = [];
  const FWD = [[0, 1], [1, 0], [1, 1]];
  for (const { col, row } of ALL_POSITIONS) {
    const a = posToWorld(col, row);
    for (const [dc, dr] of FWD) {
      const nc = String.fromCharCode(col.charCodeAt(0) + dc);
      const nr = row + dr;
      if (!isValidPos(nc, nr)) continue;
      const b = posToWorld(nc, nr);
      gridLines.push(a.x, 0.005, a.z, b.x, 0.005, b.z);
    }
  }
  const gridGeom = new THREE.BufferGeometry();
  gridGeom.setAttribute("position", new THREE.Float32BufferAttribute(gridLines, 3));
  const gridMat = new THREE.LineBasicMaterial({ color: PALETTE.grid, transparent: true, opacity: 0.78 });
  const gridLineSegs = new THREE.LineSegments(gridGeom, gridMat);
  M.boardGroup.add(gridLineSegs);

  // × hash marks at every intersection
  const hashSize = 0.10;
  const hashLines = [];
  for (const { col, row } of ALL_POSITIONS) {
    const a = posToWorld(col, row);
    hashLines.push(a.x - hashSize, 0.006, a.z - hashSize, a.x + hashSize, 0.006, a.z + hashSize);
    hashLines.push(a.x - hashSize, 0.006, a.z + hashSize, a.x + hashSize, 0.006, a.z - hashSize);
  }
  const hashGeom = new THREE.BufferGeometry();
  hashGeom.setAttribute("position", new THREE.Float32BufferAttribute(hashLines, 3));
  const hashMat = new THREE.LineBasicMaterial({ color: PALETTE.hash, transparent: true, opacity: 0.92 });
  const hashSegs = new THREE.LineSegments(hashGeom, hashMat);
  M.boardGroup.add(hashSegs);

  // A-K column letters + 1-11 row numbers, printed on the board edges
  _buildCoordinateLabels(slabW, slabD);

  // Off-board ring reserves — ghost outlines in two corners
  _buildRingReserves(slabW, slabD);
}

function _buildCoordinateLabels(slabW, slabD) {
  // Render all coordinate labels onto a single full-board canvas, then map
  // that canvas as a texture onto a transparent plane just above the board
  // top. This way the labels tilt + zoom with the board as one piece.
  const PX_PER_UNIT = 96;  // resolution per board-unit
  const margin = 1.0;       // unit margin around the lattice
  const usableW = (BBOX.xMax - BBOX.xMin) + margin * 2;
  const usableD = (BBOX.yMax - BBOX.yMin) + margin * 2;
  const canvasW = Math.ceil(usableW * PX_PER_UNIT);
  const canvasH = Math.ceil(usableD * PX_PER_UNIT);

  const c = document.createElement("canvas");
  c.width = canvasW;
  c.height = canvasH;
  const ctx = c.getContext("2d");
  ctx.fillStyle = "rgba(0, 0, 0, 0)";
  ctx.fillRect(0, 0, canvasW, canvasH);

  // Math → canvas-pixel transform. Canvas origin is top-left; board's
  // math-y grows upward, so we flip y when rendering.
  const toPx = (mx, my) => ({
    px: (mx - BBOX.xMin + margin) * PX_PER_UNIT,
    py: (BBOX.yMax - my + margin) * PX_PER_UNIT,
  });

  // Quiet indicators, not focal elements. Smaller, lighter, lower contrast —
  // the eye registers them when it looks for them and ignores them otherwise.
  ctx.fillStyle = "rgba(30, 44, 56, 0.42)";
  ctx.font = `400 ${Math.round(PX_PER_UNIT * 0.22)}px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  // Column letters — placed just BELOW the lowest valid row of each column.
  // Pushed slightly further out (0.70 units) so they read as edge labels,
  // not crowding the lattice.
  for (const col of COLUMNS) {
    const rows = VALID_POSITIONS[col];
    const bottomRow = rows[0];
    const { x, y } = mathXY(col, bottomRow);
    const { px, py } = toPx(x, y - 0.70);
    ctx.fillText(col, px, py);
  }

  // Row numbers — placed just LEFT of the leftmost valid column for each row
  ctx.textAlign = "right";
  for (let row = 1; row <= 11; row++) {
    let leftCol = null;
    for (const col of COLUMNS) {
      if (VALID_POSITIONS[col].includes(row)) { leftCol = col; break; }
    }
    if (!leftCol) continue;
    const { x, y } = mathXY(leftCol, row);
    const { px, py } = toPx(x - 0.65, y);
    ctx.fillText(String(row), px, py);
  }

  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  tex.needsUpdate = true;

  // Plane positioned to match the canvas extent in world space.
  const planeW = usableW;
  const planeD = usableD;
  const planeGeom = new THREE.PlaneGeometry(planeW, planeD);
  const planeMat = new THREE.MeshBasicMaterial({
    map: tex,
    transparent: true,
    depthWrite: false,
  });
  const labelsPlane = new THREE.Mesh(planeGeom, planeMat);
  labelsPlane.rotation.x = -Math.PI / 2;
  // Slightly above the grid lines so the labels read crisply
  labelsPlane.position.set(
    (BBOX.xMin + BBOX.xMax) / 2 - BBOX_CX,
    0.010,
    -((BBOX.yMin + BBOX.yMax) / 2 - BBOX_CY),
  );
  M.boardGroup.add(labelsPlane);
}

function _applyBoardVignetteColors(geom, w, d) {
  // Per-vertex colors: lighter near the center of the top face, deeper at
  // edges. We address only the top-face vertices; for other faces we paint
  // a uniform "side" color so the slab edge reads as the board's printed
  // border.
  const pos = geom.attributes.position;
  const colors = new Float32Array(pos.count * 3);
  const center = new THREE.Color(PALETTE.boardTop);
  const edge = new THREE.Color(PALETTE.boardEdge);
  const side = new THREE.Color(PALETTE.boardSide);
  const halfW = w / 2, halfD = d / 2;
  for (let i = 0; i < pos.count; i++) {
    const y = pos.getY(i);
    if (y > 0) {
      // top face — radial blend
      const x = pos.getX(i), z = pos.getZ(i);
      const r = Math.min(1, Math.hypot(x / halfW, z / halfD));
      // ease-out cubic for a softer falloff
      const t = 1 - Math.pow(1 - r, 3);
      const c = center.clone().lerp(edge, t * 0.85);
      colors[i * 3] = c.r; colors[i * 3 + 1] = c.g; colors[i * 3 + 2] = c.b;
    } else {
      colors[i * 3] = side.r; colors[i * 3 + 1] = side.g; colors[i * 3 + 2] = side.b;
    }
  }
  geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
}

function _buildRingReserves(slabW, slabD) {
  // Ghost outline rings in two opposite corners showing un-placed rings.
  // Visual flavor only — not state-tracked yet.
  const ringR = 0.40, innerR = 0.22;
  const corners = [
    { x: -slabW / 2 + 1.0, z: -slabD / 2 + 1.0, dx: 1.0 },
    { x: slabW / 2 - 1.0, z: slabD / 2 - 1.0, dx: -1.0 },
  ];
  for (const c of corners) {
    for (let i = 0; i < 3; i++) {
      const geom = new THREE.RingGeometry(innerR, ringR, 40);
      const mat = new THREE.MeshBasicMaterial({
        color: PALETTE.boardEdge,
        transparent: true,
        opacity: 0.32,
        side: THREE.DoubleSide,
      });
      const m = new THREE.Mesh(geom, mat);
      m.rotation.x = -Math.PI / 2;
      m.position.set(c.x + i * c.dx * 0.9, 0.004, c.z);
      M.boardGroup.add(m);
    }
  }
}

function _buildHoverMesh() {
  const geom = new THREE.RingGeometry(0.30, 0.36, 32);
  const mat = new THREE.MeshBasicMaterial({
    color: PALETTE.hover,
    transparent: true,
    opacity: 0.55,
    side: THREE.DoubleSide,
  });
  const m = new THREE.Mesh(geom, mat);
  m.rotation.x = -Math.PI / 2;
  m.visible = false;
  M.hoverMesh = m;
  M.boardGroup.add(m);
}

function _buildSelectionMesh() {
  const geom = new THREE.RingGeometry(0.38, 0.44, 32);
  const mat = new THREE.MeshBasicMaterial({
    color: PALETTE.selection,
    transparent: true,
    opacity: 0.85,
    side: THREE.DoubleSide,
  });
  const m = new THREE.Mesh(geom, mat);
  m.rotation.x = -Math.PI / 2;
  m.visible = false;
  M.selectionMesh = m;
  M.boardGroup.add(m);
}

function _buildPiece(kind) {
  // Cylinder-based discs / rings. Outer rim wrapped in teal to echo the
  // YINSH visual signature.
  const isWhite = kind.startsWith("WHITE");
  const isRing = kind.endsWith("RING");
  const faceTex = isWhite ? M.textures.lightFace : M.textures.darkFace;

  const group = new THREE.Group();

  if (isRing) {
    const outerR = 0.40;
    const innerR = 0.24;
    const h = 0.10;
    // Lathe a square cross-section to make a ring with vertical sides
    const points = [
      new THREE.Vector2(innerR, 0),
      new THREE.Vector2(outerR, 0),
      new THREE.Vector2(outerR, h),
      new THREE.Vector2(innerR, h),
      new THREE.Vector2(innerR, 0),
    ];
    const ringGeom = new THREE.LatheGeometry(points, 64);
    const ringMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      map: faceTex,
      roughness: 0.62,
      metalness: 0.02,
    });
    const ring = new THREE.Mesh(ringGeom, ringMat);
    ring.castShadow = true;
    ring.receiveShadow = true;
    group.add(ring);
    // Teal rim — a thin annulus sitting just above the top face, slightly
    // wider than the outer radius so it reads as a printed/painted stroke.
    const rimGeom = new THREE.RingGeometry(outerR - 0.015, outerR + 0.020, 64);
    const rimMat = new THREE.MeshBasicMaterial({
      color: PALETTE.pieceRim,
      side: THREE.DoubleSide,
    });
    const rim = new THREE.Mesh(rimGeom, rimMat);
    rim.rotation.x = -Math.PI / 2;
    rim.position.y = h + 0.001;
    group.add(rim);
    // Inner rim — paints the inside edge of the ring with teal as well
    const innerRimGeom = new THREE.RingGeometry(innerR - 0.020, innerR + 0.015, 64);
    const innerRim = new THREE.Mesh(innerRimGeom, rimMat);
    innerRim.rotation.x = -Math.PI / 2;
    innerRim.position.y = h + 0.001;
    group.add(innerRim);
  } else {
    // Marker — short flat cylinder
    const r = 0.26;
    const h = 0.07;
    const geom = new THREE.CylinderGeometry(r, r, h, 64);
    const mat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      map: faceTex,
      roughness: 0.68,
      metalness: 0.02,
    });
    const m = new THREE.Mesh(geom, mat);
    m.castShadow = true;
    m.receiveShadow = true;
    m.position.y = h / 2;
    group.add(m);
    // Teal rim on top face — same idiom as rings
    const rimGeom = new THREE.RingGeometry(r - 0.015, r + 0.020, 64);
    const rimMat = new THREE.MeshBasicMaterial({
      color: PALETTE.pieceRim,
      side: THREE.DoubleSide,
    });
    const rim = new THREE.Mesh(rimGeom, rimMat);
    rim.rotation.x = -Math.PI / 2;
    rim.position.y = h + 0.001;
    group.add(rim);
  }
  return group;
}

function _disposeObject(obj) {
  obj.traverse((c) => {
    if (c.geometry) c.geometry.dispose();
    if (c.material) {
      if (Array.isArray(c.material)) c.material.forEach((m) => m.dispose());
      else c.material.dispose();
    }
  });
}

// Mirror of the position-set so app.js can still call isValidPos via this
// module if it ever needs to (without re-importing constants).
export { isValidPos };
