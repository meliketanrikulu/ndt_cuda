#!/usr/bin/env bash
# Runs one NDT through the Autoware rosbag replay with RViz on a virtual screen and records it, with the CPU, GPU and
# video memory use of the NDT process in a corner. RViz renders on the CPU there, so measurements must not come from these runs.
# usage: record_demo.sh <builtin|ndt_cuda|cuda_ndt_matcher> <output.mp4> <label>
# Environment: RATE (bag replay rate, default 1.0: real time, so the CPU and GPU use in the corner are those of a real run)
set -euo pipefail

MODE="$1"
LABEL="$3"
export RATE="${RATE:-1.0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VIDEO="$(realpath -m "$2")"
OVERLAY_DIR="$(mktemp -d)"
export OVERLAY="$OVERLAY_DIR/overlay.txt"

case "$MODE" in
  builtin | ndt_cuda) PROCESS=autoware_ndt_scan_matcher_node ;;
  cuda_ndt_matcher) PROCESS=cuda_ndt_matcher ;;
esac

mkdir -p "$(dirname "$VIDEO")"
chmod 755 "$OVERLAY_DIR"
python3 "$HERE/video_overlay.py" "$LABEL" "$PROCESS" "$OVERLAY" &
UPDATER=$!
trap 'kill $UPDATER 2>/dev/null || true' EXIT

bash "$HERE/run_benchmark.sh" "$MODE" video true
echo "recorded $VIDEO"
