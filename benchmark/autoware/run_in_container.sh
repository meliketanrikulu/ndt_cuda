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
  cuda_ndt_matcher)
    source "$CUDA_NDT_MATCHER_DIR/install/setup.bash"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    export AMENT_PREFIX_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cuda_overlay:$AMENT_PREFIX_PATH"
    ;;
  *) echo "unknown mode $MODE" >&2; exit 1 ;;
esac

mkdir -p "$OUTPUT"
LAUNCH_ARGS=(
  map_path:="$DATA/maps/sample-map-rosbag"
  vehicle_model:=sample_vehicle
  sensor_model:=sample_sensor_kit
  perception:=false planning:=false control:=false
  rviz:=false
)

ros2 launch autoware_launch logging_simulator.launch.xml "${LAUNCH_ARGS[@]}" > "$OUTPUT/autoware.log" 2>&1 &
AUTOWARE=$!

# Autoware's RViz configuration, following the vehicle as the Autoware documentation suggests. With VIDEO set, RViz
# runs on a virtual screen that is recorded, with the text of the file OVERLAY in a corner.
if [[ "$RVIZ" == "true" ]]; then
  if [[ -n "${VIDEO:-}" ]]; then
    export DISPLAY=:99
    Xvfb :99 -screen 0 1920x1080x24 > "$OUTPUT/xvfb.log" 2>&1 &
    sleep 2
  fi
  # On the virtual screen, the window is made to fill the screen. Everything else is Autoware's default configuration.
  sed -e 's/Target Frame: viewer/Target Frame: base_link/' \
    $( [[ -n "${VIDEO:-}" ]] && echo "-e s/^\(\s*\)Height:\s*1565/\1Height:\x201080/ -e s/^\(\s*\)Width:\s*2813/\1Width:\x201920/ -e s/^\(\s*\)X:\s*67\$/\1X:\x200/ -e s/^\(\s*\)Y:\s*27\$/\1Y:\x200/" ) \
    "$(ros2 pkg prefix --share autoware_launch)/rviz/autoware.rviz" > "$OUTPUT/view.rviz"
  rviz2 -d "$OUTPUT/view.rviz" --ros-args -p use_sim_time:=true \
    --params-file "$(ros2 pkg prefix --share sample_vehicle_description)/config/vehicle_info.param.yaml" > "$OUTPUT/rviz.log" 2>&1 &
fi
trap 'kill -INT $AUTOWARE 2>/dev/null; sleep 3; kill -9 $AUTOWARE 2>/dev/null; pkill -9 -f /opt/autoware 2>/dev/null' EXIT

# Wait until the map is loaded and the scan matcher service exists.
for _ in $(seq 1 90); do
  ros2 service list 2>/dev/null | grep -q '/localization/pose_estimator/ndt_align_srv' && break
  sleep 2
done
sleep "${SETTLE_SECONDS:-15}"

ros2 bag record -s sqlite3 -o "$OUTPUT/bag" \
  /localization/util/downsample/pointcloud \
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

if [[ -n "${VIDEO:-}" ]]; then
  # Without a window manager, RViz is placed and sized by hand.
  xdotool search --sync --name RViz windowmove 0 0 windowsize 1920 1080 > /dev/null
  ffmpeg -hide_banner -loglevel error -y -f x11grab -framerate 30 -video_size 1920x1080 -i :99 \
    -vf "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf:textfile=$OVERLAY:reload=1:expansion=none:x=w-tw-20:y=20:fontsize=28:fontcolor=white:line_spacing=8:box=1:boxcolor=black@0.65:boxborderw=10" \
    -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p "$VIDEO" > "$OUTPUT/ffmpeg.log" 2>&1 &
  RECORDER=$!
fi
date +%s.%N > "$OUTPUT/replay_start"
ros2 bag play "$DATA/recordings/bags/sample-rosbag" -r "$RATE" -s sqlite3 > "$OUTPUT/play.log" 2>&1
date +%s.%N > "$OUTPUT/replay_end"
sleep 5
if [[ -n "${RECORDER:-}" ]]; then
  kill -INT "$RECORDER"
  wait "$RECORDER"
fi

kill -INT $RECORD 2>/dev/null
wait $RECORD 2>/dev/null
ros2 topic list > "$OUTPUT/topics.txt" 2>/dev/null
