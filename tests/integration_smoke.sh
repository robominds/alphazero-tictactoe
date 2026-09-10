#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build}"
CHECKPOINT="$(mktemp -t az_integration_XXXXXX.bin)"
trap 'rm -f "$CHECKPOINT"' EXIT

echo "== train (3 tiny iterations) =="
"$BUILD_DIR/train" 3 "$CHECKPOINT"
test -s "$CHECKPOINT"

echo "== evaluate =="
"$BUILD_DIR/evaluate" "$CHECKPOINT" 3

echo "== play_cli (scripted game) =="
echo "0 1 2 3 4 5 6 7 8" | "$BUILD_DIR/play_cli" "$CHECKPOINT"

echo "integration smoke test passed"
