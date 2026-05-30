#!/usr/bin/env bash
# Build the yngine_driver binary. Idempotent — re-runs cmake configure +
# build into ./build-release/. The Python bridge resolves this binary at
# `third_party/yngine_driver/build-release/yngine_driver`.
set -euo pipefail

cd "$(dirname "$0")"

# Ensure the yngine submodule (and its Xoshiro-cpp sub-submodule) are checked
# out. Idempotent — exits cleanly if already initialized.
git submodule update --init --recursive ../yngine

# Patch upstream allocators.cpp for macOS (Apple Silicon). Upstream gates the
# mmap path on __linux__ || EMSCRIPTEN and static_asserts(false) elsewhere,
# even though Darwin has the same mmap signature. Idempotent: skips if the
# patched marker is already present. We mutate the submodule worktree in
# place; `git submodule update` would revert it, so build.sh is the canonical
# invocation.
ALLOC=../yngine/yngine/allocators.cpp
if [ -f "$ALLOC" ] && ! grep -q '__APPLE__' "$ALLOC"; then
    python3 - "$ALLOC" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
src = p.read_text()
src = src.replace(
    'defined(__linux__) || defined(EMSCRIPTEN)',
    'defined(__linux__) || defined(EMSCRIPTEN) || defined(__APPLE__)',
)
# The destructor's munmap branch is gated on plain __linux__ — widen it too.
src = src.replace(
    '#if defined(__linux__)\n    munmap',
    '#if defined(__linux__) || defined(__APPLE__)\n    munmap',
)
p.write_text(src)
PY
    echo "patched: $ALLOC (added __APPLE__ to mmap branch)"
fi

# Pick cmake. Homebrew installs to /opt/homebrew on Apple Silicon and is not
# always on PATH for non-interactive shells.
CMAKE="${CMAKE:-}"
if [ -z "$CMAKE" ]; then
    if command -v cmake >/dev/null 2>&1; then
        CMAKE=cmake
    elif [ -x /opt/homebrew/bin/cmake ]; then
        CMAKE=/opt/homebrew/bin/cmake
    else
        echo "error: cmake not found. Install via 'brew install cmake'." >&2
        exit 1
    fi
fi

BUILD_DIR="${BUILD_DIR:-build-release}"

"$CMAKE" -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
"$CMAKE" --build "$BUILD_DIR" --parallel

echo
echo "built: $(pwd)/$BUILD_DIR/yngine_driver"
