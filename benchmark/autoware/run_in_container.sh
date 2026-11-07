#!/usr/bin/env bash
# Runs inside the Autoware container: starts Autoware with one NDT, replays the sample bag and records the results.
# usage: run_in_container.sh <builtin|ndt_cuda|cuda_ndt_matcher> <workspace> <data_dir> <output_dir> [rviz: true|false]
set -o pipefail

MODE="$1"
WORKSPACE="$2"
DATA="$3"
OUTPUT="$4"
RVIZ="${5:-false}"
RATE="${RATE:-0.2}"

source /opt/autoware/setup.bash
case "$MODE" in
  builtin) ;;
  ndt_cuda) source "$WORKSPACE/install/setup.bash" ;;
  cuda_ndt_matcher) source "$CUDA_NDT_MATCHER_DIR/install/setup.bash" ;;
  *) echo "unknown mode $MODE" >&2; exit 1 ;;
esac

mkdir -p "$OUTPUT"
LAUNCH_ARGS=(
  map_path:="$DATA/maps/sample-map-rosbag"
  vehicle_model:=sample_vehicle
  sensor_model:=sample_sensor_kit
  perception:=false planning:=false control:=false
  rviz:="$RVIZ"
)
if [[ -n "${RVIZ_CONFIG:-}" ]]; then LAUNCH_ARGS+=(rviz_config:="$RVIZ_CONFIG"); fi

ros2 launch autoware_launch logging_simulator.launch.xml "${LAUNCH_ARGS[@]}" > "$OUTPUT/autoware.log" 2>&1 &
AUTOWARE=$!
trap 'kill -INT $AUTOWARE 2>/dev/null; sleep 3; kill -9 $AUTOWARE 2>/dev/null; pkill -9 -f /opt/autoware 2>/dev/null' EXIT

# Wait until the map is loaded and the scan matcher service exists.
for _ in $(seq 1 90); do
  ros2 service list 2>/dev/null | grep -q '/localization/pose_estimator/ndt_align_srv' && break
  sleep 2
done
sleep "${SETTLE_SECONDS:-15}"

ros2 bag record -s sqlite3 -o "$OUTPUT/bag" \
  /localization/pose_estimator/exe_time_ms \
  /localization/pose_estimator/iteration_num \
  /localization/pose_estimator/transform_probability \
  /localization/pose_estimator/nearest_voxel_transformation_likelihood \
  /localization/pose_estimator/pose_with_covariance \
  /localization/pose_estimator/initial_pose_with_covariance \
  /localization/kinematic_state \
  /localization/pose_twist_fusion_filter/kinematic_state \
  /localization/initialization_state \
  /sensing/lidar/concatenated/pointcloud/info \
  ${EXTRA_TOPICS:-} \
  > "$OUTPUT/record.log" 2>&1 &
RECORD=$!
sleep 3

date +%s.%N > "$OUTPUT/replay_start"
ros2 bag play "$DATA/recordings/bags/sample-rosbag" -r "$RATE" -s sqlite3 > "$OUTPUT/play.log" 2>&1
date +%s.%N > "$OUTPUT/replay_end"
sleep 5

kill -INT $RECORD 2>/dev/null
wait $RECORD 2>/dev/null
ros2 topic list > "$OUTPUT/topics.txt" 2>/dev/null
