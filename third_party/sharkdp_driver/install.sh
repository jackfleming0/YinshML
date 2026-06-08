#!/usr/bin/env bash
# Reproduce the sharkdp/yinsh engine + the `yinsh-driver` protocol binary that
# `yinsh_ml.sharkdp.Sharkdp` drives. The driver crate must live *inside* the
# sharkdp Cargo workspace (it has path deps on sharkdp's `yinsh`/`yinsh_ai`
# crates), so this script clones sharkdp at a pinned commit and copies the
# vendored crate source (Cargo.toml + src/main.rs, kept here) into it before
# building. Needs a Rust toolchain (https://rustup.rs).
#
#   bash third_party/sharkdp_driver/install.sh
#
# Produces: third_party/sharkdp_yinsh/target/release/yinsh-driver
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HERE/../sharkdp_yinsh"        # gitignored clone (not vendored)
PIN=de76c19b8d27efaa4e78fc57b06076708bcc46d6   # sharkdp/yinsh pinned commit

if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/sharkdp/yinsh.git "$DEST"
fi
git -C "$DEST" checkout "$PIN"

mkdir -p "$DEST/crates/yinsh_driver/src"
cp "$HERE/Cargo.toml"   "$DEST/crates/yinsh_driver/Cargo.toml"
cp "$HERE/src/main.rs"  "$DEST/crates/yinsh_driver/src/main.rs"

( cd "$DEST" && cargo build --release -p yinsh_driver )
echo "Built: $DEST/target/release/yinsh-driver"
