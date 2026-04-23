#!/usr/bin/env bash
# Sync a training run directory to a remote backup target.
#
# Per CLOUD_TRAINING_PLAN.md §1.5: called from the supervisor after each
# iteration so cloud runs survive a terminated spot instance. Exits 0 on
# success, non-zero on failure — but the supervisor treats any sync error
# as non-fatal and continues training.
#
# Usage:
#   scripts/sync_run.sh <run_dir>
#
# Destination is read from the SYNC_RUN_DEST env var. If unset, the script
# is a no-op (exits 0 immediately) so a stock local training run doesn't
# need to configure anything. Supported schemes:
#
#   s3://bucket/prefix       → `aws s3 sync` (if `aws` available)
#                              or `rclone sync` (if `rclone` available)
#   gs://bucket/prefix       → `gsutil rsync`
#   r2:remote:path, etc.     → `rclone sync` (any configured rclone remote)
#   user@host:/path          → rsync over ssh
#   /absolute/local/path     → plain rsync
#
# Large-buffer exclusion: replay_buffer.pkl.gz files larger than 500 MB are
# excluded so a single iteration's sync stays tractable. Checkpoints and
# metrics always get through.

set -o pipefail  # no -e / -u: we want to handle errors locally (empty
                 # arrays are intentional when no exclusions apply).

RUN_DIR="${1:-}"
DEST="${SYNC_RUN_DEST:-}"

if [[ -z "${RUN_DIR}" ]]; then
    echo "[sync_run] ERROR: run directory required (usage: sync_run.sh <run_dir>)" >&2
    exit 2
fi

if [[ ! -d "${RUN_DIR}" ]]; then
    echo "[sync_run] ERROR: run_dir '${RUN_DIR}' is not a directory" >&2
    exit 2
fi

if [[ -z "${DEST}" ]]; then
    # Default: silent no-op. Training runs without SYNC_RUN_DEST shouldn't
    # produce noise at every iteration boundary.
    exit 0
fi

RUN_BASENAME="$(basename "${RUN_DIR%/}")"

# --- Build the replay-buffer skip list ---
# rsync and most cloud CLIs want exclusion patterns, not paths, so we
# probe for the buffer and decide at runtime.
EXCLUDE_REPLAY=()
if [[ -f "${RUN_DIR}/replay_buffer.pkl.gz" ]]; then
    BUFFER_SIZE=$(stat -f%z "${RUN_DIR}/replay_buffer.pkl.gz" 2>/dev/null \
                  || stat -c%s "${RUN_DIR}/replay_buffer.pkl.gz" 2>/dev/null \
                  || echo 0)
    THRESHOLD=$((500 * 1024 * 1024))
    if (( BUFFER_SIZE > THRESHOLD )); then
        echo "[sync_run] replay_buffer.pkl.gz is ${BUFFER_SIZE} bytes (>500MB); excluding from sync."
        EXCLUDE_REPLAY=("replay_buffer.pkl.gz")
    fi
fi

dispatch_s3() {
    local target="$1"
    if command -v aws >/dev/null 2>&1; then
        local args=(s3 sync --delete "${RUN_DIR%/}" "${target%/}/${RUN_BASENAME}")
        for pat in ${EXCLUDE_REPLAY[@]+"${EXCLUDE_REPLAY[@]}"}; do
            args+=(--exclude "${pat}")
        done
        aws "${args[@]}"
    elif command -v rclone >/dev/null 2>&1; then
        # rclone understands s3:// when a remote named "s3" is configured.
        local args=(sync "${RUN_DIR%/}" "${target%/}/${RUN_BASENAME}")
        for pat in ${EXCLUDE_REPLAY[@]+"${EXCLUDE_REPLAY[@]}"}; do
            args+=(--exclude "${pat}")
        done
        rclone "${args[@]}"
    else
        echo "[sync_run] ERROR: neither 'aws' nor 'rclone' available; cannot sync to s3://." >&2
        return 3
    fi
}

dispatch_gs() {
    local target="$1"
    if ! command -v gsutil >/dev/null 2>&1; then
        echo "[sync_run] ERROR: 'gsutil' not available; cannot sync to gs://." >&2
        return 3
    fi
    local args=(-m rsync -r -d)
    for pat in "${EXCLUDE_REPLAY[@]}"; do
        args+=(-x "${pat}")
    done
    args+=("${RUN_DIR%/}" "${target%/}/${RUN_BASENAME}")
    gsutil "${args[@]}"
}

dispatch_rclone() {
    # Any rclone-style "remote:path" destination (r2:bucket/path, b2:...,
    # gdrive:..., etc.).
    local target="$1"
    if ! command -v rclone >/dev/null 2>&1; then
        echo "[sync_run] ERROR: 'rclone' not available; cannot sync to '${target}'." >&2
        return 3
    fi
    local args=(sync "${RUN_DIR%/}" "${target%/}/${RUN_BASENAME}")
    for pat in "${EXCLUDE_REPLAY[@]}"; do
        args+=(--exclude "${pat}")
    done
    rclone "${args[@]}"
}

dispatch_rsync() {
    # ssh/local rsync destination.
    local target="$1"
    if ! command -v rsync >/dev/null 2>&1; then
        echo "[sync_run] ERROR: 'rsync' not available; cannot sync." >&2
        return 3
    fi
    local args=(-az --delete)
    for pat in "${EXCLUDE_REPLAY[@]}"; do
        args+=(--exclude "${pat}")
    done
    # Trailing slash on source copies contents into a dir named RUN_BASENAME.
    args+=("${RUN_DIR%/}/" "${target%/}/${RUN_BASENAME}/")
    rsync "${args[@]}"
}

echo "[sync_run] Syncing ${RUN_DIR} → ${DEST}/${RUN_BASENAME}"

case "${DEST}" in
    s3://*)
        dispatch_s3 "${DEST}"
        ;;
    gs://*)
        dispatch_gs "${DEST}"
        ;;
    *:*)
        # Ambiguous: either ssh (user@host:path) or rclone remote (name:path).
        # Heuristic: if it contains '@' before the colon, it's ssh-style rsync.
        prefix="${DEST%%:*}"
        if [[ "${prefix}" == *"@"* ]]; then
            dispatch_rsync "${DEST}"
        else
            dispatch_rclone "${DEST}"
        fi
        ;;
    /*)
        dispatch_rsync "${DEST}"
        ;;
    *)
        echo "[sync_run] ERROR: unrecognized SYNC_RUN_DEST scheme: '${DEST}'" >&2
        exit 2
        ;;
esac

rc=$?
if [[ $rc -eq 0 ]]; then
    echo "[sync_run] sync ok: ${RUN_BASENAME}"
else
    echo "[sync_run] sync FAILED with exit ${rc} (non-fatal to training)" >&2
fi
exit $rc
