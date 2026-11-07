#!/usr/bin/env python3
"""Writes the scan matcher topics of a recorded bag to metrics.csv and poses.csv. Runs inside the Autoware container."""

import csv
import sys

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

PREFIX = "/localization/pose_estimator/"
SCAN_TOPIC = "/localization/util/downsample/pointcloud"
SCALARS = {
    "exe_time_ms": "exe_time_ms",
    "iteration_num": "iterations",
    "transform_probability": "tp",
    "nearest_voxel_transformation_likelihood": "nvtl",
}


def main(output_dir):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=f"{output_dir}/bag", storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}

    scalars = {key: [] for key in SCALARS}
    poses = []
    scan_arrivals = {}
    while reader.has_next():
        topic, data, received = reader.read_next()
        if topic == SCAN_TOPIC:
            stamp = deserialize_message(data, types[topic]).header.stamp
            scan_arrivals[(stamp.sec, stamp.nanosec)] = received
            continue
        if not topic.startswith(PREFIX):
            continue
        name = topic[len(PREFIX):]
        msg = deserialize_message(data, types[topic])
        if name in SCALARS:
            scalars[name].append((msg.stamp.sec + msg.stamp.nanosec * 1e-9, msg.data))
        elif name == "pose_with_covariance":
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            # exe_time_ms does not cover the same work in every implementation; this does: scan in, pose out, seen by the same recorder.
            arrival = scan_arrivals.get((msg.header.stamp.sec, msg.header.stamp.nanosec))
            latency = (received - arrival) * 1e-6 if arrival is not None else ""
            poses.append((stamp, p.x, p.y, p.z, q.x, q.y, q.z, q.w, latency))

    with open(f"{output_dir}/metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stamp"] + list(SCALARS.values()))
        # All four topics are published once per alignment, in the same order.
        for row in zip(*(scalars[key] for key in SCALARS)):
            writer.writerow([row[0][0]] + [value for _, value in row])

    with open(f"{output_dir}/poses.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stamp", "x", "y", "z", "qx", "qy", "qz", "qw", "latency_ms"])
        writer.writerows(poses)


if __name__ == "__main__":
    main(sys.argv[1])
