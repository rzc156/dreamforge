#!/usr/bin/env bash
# Fine-tune on imagined corrections, the replay-only control, then evaluate both
# and regenerate the report.  Designed to run as a single background chain so the
# GPU is used by one job at a time.
#
# Usage: scripts/run_training.sh [RUN_DIR] [STEPS] [BATCH] [SUITE]
set -euo pipefail

RUN="${1:-runs/main}"
STEPS="${2:-3000}"
BATCH="${3:-8}"
SUITE="${4:-libero_spatial}"

cd "$(dirname "$0")/.."

echo "=== [1/5] fine-tune on imagined corrections (all) ==="
rm -rf "$RUN/ft_all"
python -u -m dreamforge.finetune --samples "$RUN/build_all" --out "$RUN/ft_all" \
    --steps "$STEPS" --batch-size "$BATCH" --replay-ratio 0.5 --lr 1e-5

echo "=== [2/5] control: replay-only ==="
rm -rf "$RUN/ft_replay"
python -u -m dreamforge.finetune --samples none --out "$RUN/ft_replay" \
    --steps "$STEPS" --batch-size "$BATCH" --replay-ratio 1.0 --lr 1e-5

echo "=== [3/5] eval fine-tuned ==="
python -u -m dreamforge.eval --policy-path "$RUN/ft_all" --task "$SUITE" \
    --episodes 10 --out "$RUN/eval_ft_all.json"

echo "=== [4/5] eval control ==="
python -u -m dreamforge.eval --policy-path "$RUN/ft_replay" --task "$SUITE" \
    --episodes 10 --out "$RUN/eval_ft_replay.json"

echo "=== [5/5] report ==="
python -u -m dreamforge.report --run "$RUN"
echo "done -> $RUN/report.md"
