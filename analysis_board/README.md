# YINSH analysis board

Drag-drop position analyzer driven by the project's AlphaZero-style network.
Compose any board state, choose a model + (optionally) MCTS budget, and see
the top moves with policy probabilities and a value estimate.

## Run

```bash
pip install flask    # if not already installed
python analysis_board/server.py
# http://127.0.0.1:5173
```

Environment knobs:
- `YNS_HOST` (default `127.0.0.1`)
- `YNS_PORT` (default `5173`)
- `YNS_DEVICE` (`cuda` / `mps` / `cpu`; default = auto-detect)

## What it does

- **`GET /api/models`** — scans `models/<name>/` for `best_supervised.pt` (then
  `best.pt`, `final.pt`, `supervised_final.pt`) and exposes the catalog.
- **`POST /api/evaluate`** — JSON body:
  ```json
  {
    "model_id": "yngine_volume_15ch_pretrain/best_supervised.pt",
    "pieces": [{"pos": "E5", "piece": "WHITE_RING"}, ...],
    "phase": "MAIN_GAME",
    "side_to_move": "WHITE",
    "scores": {"WHITE": 0, "BLACK": 0},
    "num_sims": 200,
    "top_k": 8
  }
  ```
  - `num_sims = 0` → raw policy head (one forward pass)
  - `num_sims > 0` → MCTS (visit distribution + root value)

  Response includes `value`, `top_moves[]` with `prob`, optional `visits`, and
  full move metadata.

## Interaction

- **Drag** any palette piece (white/black ring or marker, or the eraser) onto
  an intersection on the board. Snap-to-nearest.
- **Right-click** any placed piece to remove it.
- **Hover** a top move in the sidebar to see its arrow rendered on the board.
- **Enter** evaluates; **Esc** disarms the current tool.

## Geometry

Ported from `yinsh_ml/viz/board_render.py`:

    screen_x = col_idx * sqrt(3)/2
    screen_y = (row - 1) - col_idx * 0.5

The same transform that the verified matplotlib renderer uses. Canvas Y is
flipped relative to the math convention.

## Caveats

- Position validation happens server-side on Evaluate. The UI lets you compose
  illegal positions (rings without enough markers, wrong rings_placed for the
  phase, etc.); the server returns a structured error you'll see in the status
  banner.
- MCTS is constructed once per `(model_id, num_sims)` pair and cached.
  `enable_subtree_reuse=False` so each evaluate is a fresh root.
- The wrapper auto-detects channels / blocks / value-head type from the
  checkpoint; works with both 6ch and 15ch encoders.
