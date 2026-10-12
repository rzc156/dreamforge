#!/usr/bin/env bash
# Filter-ladder + robustness extras:
#   1. train on the simulator-verified subset ("lifted")
#   2. evaluate it
#   3. held-out suite check for the main fine-tuned policy (forgetting guard)
#   4. regenerate the report
#
# Prerequisite: runs/main/verify has finished (build_lifted needs it).
# Usage: scripts/run_ladder.sh [RUN_DIR] [SUITE] [HELDOUT_SUITE]
set -euo pipefail

RUN="${1:-runs/main}"
SUITE="${2:-libero_spatial}"
HELDOUT="${3:-libero_object}"

cd "$(dirname "$0")/.."

echo "=== build lifted dataset ==="
python -u -m dreamforge.build --mine "$RUN/mine" --imagine "$RUN/imagine" \
    --out "$RUN/build_lifted" --filter lifted --verify-dir "$RUN/verify"

echo "=== fine-tune on lifted (sim-verified) corrections ==="
rm -rf "$RUN/ft_lifted"
python -u -m dreamforge.finetune --samples "$RUN/build_lifted" --out "$RUN/ft_lifted" \
    --steps 3000 --batch-size 8 --replay-ratio 0.5 --lr 1e-5

echo "=== eval lifted ==="
python -u -m dreamforge.eval --policy-path "$RUN/ft_lifted" --task "$SUITE" \
    --episodes 10 --out "$RUN/eval_ft_lifted.json"

echo "=== held-out suite check ($HELDOUT) for ft_all ==="
python -u -m dreamforge.eval --policy-path "$RUN/ft_all" --task "$HELDOUT" \
    --episodes 5 --out "$RUN/eval_ft_all_heldout.json"

echo "=== report ==="
python -u -m dreamforge.report --run "$RUN"
echo "done -> $RUN/report.md"
