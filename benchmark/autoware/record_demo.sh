#!/usr/bin/env bash
# Runs one NDT through the Autoware rosbag replay with RViz on a virtual screen and records it, then writes the CPU, GPU
# and video memory use of the scan matcher into a corner (make_video.py). The GPU use comes from a Nsight Systems
# trace, as nvidia-smi shows less than 1 % as 0. RViz renders on the CPU there, so measurements must not come from
# these runs.
# usage: record_demo.sh <builtin|ndt_cuda|cuda_ndt_matcher> <output.mp4> <label>
# Environment: RATE (bag replay rate, default 1.0: real time, so the corner shows the use of a real run)
set -euo pipefail

MODE="$1"
OUTPUT="$(realpath -m "$2")"
LABEL="$3"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="${AUTOWARE_BENCH_ROOT:-$HOME/projects}/autoware_bench/results/$MODE/video"
export RATE="${RATE:-1.0}"
export VIDEO="$RUN/screen.mp4"
# Autoware NDT does not use the GPU.
export GPU_TRACE="$( [[ "$MODE" == builtin ]] && echo 0 || echo 1 )"

bash "$HERE/run_benchmark.sh" "$MODE" video true
mkdir -p "$(dirname "$OUTPUT")"
python3 "$HERE/make_video.py" "$RUN" "$LABEL" "$OUTPUT"
echo "recorded $OUTPUT"
