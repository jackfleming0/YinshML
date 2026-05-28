"""Offline smoke test for analysis_board.server — no Flask binding.

Imports the server module, discovers models, builds a mid-game position,
runs both raw-policy and small MCTS paths. Prints top moves + value.

Run:
    source venv/bin/activate
    python analysis_board/_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Late import so sys.path is set.
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "server_mod", ROOT / "analysis_board" / "server.py"
)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)
server._models = server.discover_models()
assert server._models, "no models discovered"
print("models:")
for m in server._models:
    print(f"  - {m['id']}  (label={m['label']})")

# Pick the 15ch anchor if present, else first.
preferred = next(
    (m for m in server._models if m["label"] == "yngine_volume_15ch_pretrain"),
    server._models[0],
)
print(f"\nUsing model: {preferred['id']}")

# Build a representative MAIN_GAME position: 5 rings each at typical starting spots,
# a couple of markers scattered.
pieces = [
    {"pos": "E5", "piece": "WHITE_RING"},
    {"pos": "G7", "piece": "WHITE_RING"},
    {"pos": "C4", "piece": "WHITE_RING"},
    {"pos": "I8", "piece": "WHITE_RING"},
    {"pos": "F6", "piece": "WHITE_RING"},
    {"pos": "D4", "piece": "BLACK_RING"},
    {"pos": "F8", "piece": "BLACK_RING"},
    {"pos": "H6", "piece": "BLACK_RING"},
    {"pos": "E7", "piece": "BLACK_RING"},
    {"pos": "G5", "piece": "BLACK_RING"},
    {"pos": "E6", "piece": "WHITE_MARKER"},
    {"pos": "F5", "piece": "WHITE_MARKER"},
]

def run(num_sims):
    payload = {
        "model_id": preferred["id"],
        "pieces": pieces,
        "phase": "MAIN_GAME",
        "side_to_move": "WHITE",
        "scores": {"WHITE": 0, "BLACK": 0},
        "num_sims": num_sims,
        "top_k": 5,
    }
    # Flask test client path
    with server.app.test_client() as c:
        r = c.post("/api/evaluate", json=payload)
        return r.get_json()

print("\n--- raw policy (num_sims=0) ---")
out = run(0)
assert out["ok"], out
print(f"value: {out['value']:+.3f}  |  side={out['side_to_move']}  legal={out['num_valid_moves']}")
for i, m in enumerate(out["top_moves"][:5], 1):
    print(f"  {i}. {m['description']:<40s}  prob={m['prob'] * 100:5.1f}%")

print("\n--- MCTS (num_sims=50) ---")
out = run(50)
assert out["ok"], out
print(f"value: {out['value']:+.3f}  |  side={out['side_to_move']}  legal={out['num_valid_moves']}")
for i, m in enumerate(out["top_moves"][:5], 1):
    v = f"{m.get('visits', 0)}v"
    print(f"  {i}. {m['description']:<40s}  prob={m['prob'] * 100:5.1f}%  ({v})")

# Error path: send invalid (too many WHITE rings).
print("\n--- error case (over-stacked) ---")
bad = {
    "model_id": preferred["id"],
    "pieces": [{"pos": p, "piece": "WHITE_RING"} for p in ["E5","E6","E7","E8","E9","F5"]],
    "phase": "MAIN_GAME",
    "side_to_move": "WHITE",
    "scores": {"WHITE": 0, "BLACK": 0},
    "num_sims": 0,
}
with server.app.test_client() as c:
    r = c.post("/api/evaluate", json=bad)
    out = r.get_json()
print(f"  ok={out['ok']}  errors={out.get('errors')}")

print("\nSMOKE OK")
