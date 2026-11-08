#!/usr/bin/env bash
# Runs one NDT through the Autoware rosbag replay and records its results and resource use. Runs on the host.
# usage: run_benchmark.sh <builtin|ndt_cuda|cuda_ndt_matcher> <run name> [rviz: true|false]
#
# Environment: AUTOWARE_BENCH_ROOT (default ~/projects), RATE (bag replay rate, default 0.2 as in the Autoware documentation),
# GPU_TRACE=1 (record the GPU work with Nsight Systems from NSYS_DIR, default /opt/nvidia/nsight-systems/2024.6.2)
set -euo pipefail

MODE="$1"
NAME="$2"
RVIZ="${3:-false}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
ROOT="${AUTOWARE_BENCH_ROOT:-$HOME/projects}"
IMAGE=ghcr.io/autowarefoundation/autoware:universe-devel-cuda-1.5.0
OUTPUT="$ROOT/autoware_bench/results/$MODE/$NAME"

case "$MODE" in
  builtin | ndt_cuda) PROCESS=autoware_ndt_scan_matcher_node ;;
  cuda_ndt_matcher)
    PROCESS=cuda_ndt_matcher
    # The Autoware image plus the build dependencies of cuda_ndt_matcher, see docker/cuda_ndt_matcher.Dockerfile.
    IMAGE=ndt-bench:cuda-matcher
    ;;
  *) echo "unknown mode $MODE" >&2; exit 1 ;;
esac

rm -rf "$OUTPUT"
mkdir -p "$OUTPUT"

# With RViz, the run is recorded on a virtual screen (VIDEO: see record_demo.sh), so no display is needed.
EXTRA_ARGS=()
if [[ "$RVIZ" == "true" ]]; then
  IMAGE="$( [[ "$MODE" == cuda_ndt_matcher ]] && echo ndt-bench:cuda-matcher-video || echo ndt-bench:video )"
  EXTRA_ARGS+=(-e "VIDEO=${VIDEO:?}" -v "$(dirname "$VIDEO"):$(dirname "$VIDEO")")
fi
# The Autoware image has no Nsight Systems, so the one of the host is mounted.
NSYS_DIR="${NSYS_DIR:-/opt/nvidia/nsight-systems/2024.6.2}"
if [[ "${GPU_TRACE:-0}" == 1 ]]; then
  EXTRA_ARGS+=(-v "$NSYS_DIR:$NSYS_DIR:ro" -e "NSYS=$NSYS_DIR/target-linux-x64/nsys")
fi

python3 "$HERE/sample_resources.py" "$PROCESS" "$OUTPUT/resources.csv" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

docker run --rm --name "ndt_bench_$MODE" --gpus all --ipc=host \
  --sysctl net.ipv4.ipfrag_time=3 --sysctl net.ipv4.ipfrag_high_thresh=134217728 \
  -u "$(id -u):$(id -g)" -e HOME=/tmp -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e "CYCLONEDDS_URI=file://$HERE/cyclonedds.xml" -e "ROS_LOG_DIR=$OUTPUT/roslog" \
  -e "RATE=${RATE:-0.2}" -e "CUDA_NDT_MATCHER_DIR=$ROOT/cuda_ndt_matcher" \
  "${EXTRA_ARGS[@]}" -v "$ROOT:$ROOT" -v "$REPO:$REPO" "$IMAGE" bash -c "
    timeout 600 bash '$HERE/run_in_container.sh' '$MODE' '$ROOT/autoware_ws' '$ROOT/autoware_data' '$OUTPUT' '$RVIZ'
    source /opt/autoware/setup.bash
    python3 '$HERE/extract_metrics.py' '$OUTPUT'
  " > "$OUTPUT/run.log" 2>&1
if [[ "${GPU_TRACE:-0}" == 1 ]]; then
  python3 "$HERE/gpu_trace.py" "$OUTPUT" "$PROCESS" "$NSYS_DIR/target-linux-x64/nsys"
fi
echo "results in $OUTPUT"
