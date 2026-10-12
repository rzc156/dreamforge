#!/usr/bin/env bash
# dreamforge: end-to-end run (mine -> imagine -> verify -> build -> train -> eval -> report)
#
# Usage:
#   scripts/run_main.sh [RUN_DIR] [EPISODES] [SUITE] [WM]
# Example:
#   scripts/run_main.sh runs/main 20 libero_spatial aso
set -euo pipefail

RUN="${1:-runs/main}"
EPISODES="${2:-20}"
SUITE="${3:-libero_spatial}"
WM="${4:-aso}"

mkdir -p "$RUN"
cd "$(dirname "$0")/.."

echo "[1/7] mine   ($SUITE, $EPISODES episodes/task)"
python -u -m dreamforge.mine --task "$SUITE" --episodes "$EPISODES" --out "$RUN/mine"

echo "[2/7] diagnose world-model action response"
python -u -m dreamforge.diagnose --mine "$RUN/mine" --wms base "$WM"

echo "[3/7] imagine (CEM, wm=$WM)"
python -u -m dreamforge.imagine --mine "$RUN/mine" --out "$RUN/imagine" --wm "$WM" --context failure

echo "[4/7] verify (none/imagined/random)"
python -u -m dreamforge.verify --mine "$RUN/mine" --imagine "$RUN/imagine" \
    --out "$RUN/verify" --arms none imagined random --exec-horizon 16

echo "[5/7] build datasets (all / lifted)"
python -u -m dreamforge.build --mine "$RUN/mine" --imagine "$RUN/imagine" \
    --out "$RUN/build_all" --filter all
python -u -m dreamforge.build --mine "$RUN/mine" --imagine "$RUN/imagine" \
    --out "$RUN/build_lifted" --filter lifted --verify-dir "$RUN/verify" || true

echo "[6/7] fine-tune (synthetic) + control (replay-only)"
python -u -m dreamforge.train --samples "$RUN/build_all" --out "$RUN/ft_synth" \
    --steps 3000 --replay-ratio 0.5
python -u -m dreamforge.train --samples none --out "$RUN/ft_replay" \
    --steps 3000 --replay-ratio 1.0

echo "[7/7] evaluate base / synth / replay"
python -u -m dreamforge.eval --policy-path "$(python -c 'from dreamforge.paths import DEFAULTS;print(DEFAULTS.policy_path)')" \
    --task "$SUITE" --episodes 10 --out "$RUN/eval_base.json"
python -u -m dreamforge.eval --policy-path "$RUN/ft_synth" --task "$SUITE" --episodes 10 \
    --out "$RUN/eval_ft_synth.json"
python -u -m dreamforge.eval --policy-path "$RUN/ft_replay" --task "$SUITE" --episodes 10 \
    --out "$RUN/eval_ft_replay.json"

python -u -m dreamforge.report --run "$RUN"
echo "done -> $RUN/report.md"
