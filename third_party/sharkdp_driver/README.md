# sharkdp_driver — vendored Rust driver for sharkdp/yinsh

A line-based stdin/stdout protocol bridge around the
[sharkdp/yinsh](https://github.com/sharkdp/yinsh) negamax engine, mirroring
`third_party/yngine_driver` so `yinsh_ml.sharkdp.Sharkdp` can drive it with the
**same wire protocol** (and reuse `yinsh_ml/yngine/move_codec.py` verbatim).

## Why the source lives here

The actual sharkdp clone (`third_party/sharkdp_yinsh/`) is gitignored — we don't
vendor the whole engine. But the driver crate is *our* code, and it must compile
*inside* sharkdp's Cargo workspace (path deps on its `yinsh`/`yinsh_ai` crates).
So we keep the crate source here (`Cargo.toml`, `src/main.rs`) and copy it into
the clone at build time.

## Build

```bash
bash third_party/sharkdp_driver/install.sh
```

Clones sharkdp at the pinned commit `de76c19b8d27efaa4e78fc57b06076708bcc46d6`,
copies this crate in, and runs `cargo build --release -p yinsh_driver`. Output:
`third_party/sharkdp_yinsh/target/release/yinsh-driver` (the path
`yinsh_ml.sharkdp.default_sharkdp_binary_path()` resolves; override with the
`SHARKDP_DRIVER` env var). Requires a Rust toolchain.

## Protocol & design

See the header comment in `src/main.rs` — wire format, the verified coordinate
bijection (`x_wire = x_shark+5`, `y_wire = 5−y_shark`), and the turn-bridging
(sharkdp splits a ring move into `PlaceMarker`+`Wait`+`MoveRing`; the driver
recombines them into one wire `M`). Validated by
`scripts/smoke_sharkdp_bridge.py --mode referee`.
