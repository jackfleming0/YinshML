#!/bin/bash
# Wait for Boardspace scrape PID to finish, then run validate + convert + training.
set -e

cd "$(dirname "$0")/.."
source venv/bin/activate

BOARDSPACE_PID="${1:-}"

if [ -n "$BOARDSPACE_PID" ]; then
    echo "Waiting for Boardspace scrape PID $BOARDSPACE_PID..."
    while kill -0 "$BOARDSPACE_PID" 2>/dev/null; do
        sleep 30
    done
    echo "Boardspace scrape finished."
fi

# Run validate + convert from existing JSON
echo "=== Validate + convert ==="
python scripts/gather_expert_games.py --validate-only --output-dir expert_games 2>&1 | tee logs/boardspace_validate_convert.log

echo "=== Kicking off supervised pre-training ==="
NPZ_PATH="expert_games/training_data.npz"
if [ ! -f "$NPZ_PATH" ]; then
    echo "ERROR: $NPZ_PATH missing"
    exit 1
fi

python scripts/run_supervised_pretraining.py \
    --data "$NPZ_PATH" \
    --epochs 30 \
    --batch-size 256 \
    --lr 0.001 \
    --output-dir models/supervised \
    2>&1 | tee logs/supervised_pretraining.log
