#!/usr/bin/env python3
"""Summarizes the runs of run_benchmark.sh and draws the figures of docs/autoware_benchmark.md.

usage: make_report.py <results dir> <figure dir> <run names, comma separated>
"""

import csv
import math
import os
import re
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODES = {
    "builtin": ("Autoware NDT (CPU)", "#4c72b0"),
    "ndt_cuda": ("ndt_cuda (GPU)", "#55a868"),
    "cuda_ndt_matcher": ("cuda_ndt_matcher (GPU)", "#c44e52"),
}


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def number(value):
    return float(value) if value not in ("", None) else None


def percentile(values, p):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p / 100.0 * len(ordered)))]


def yaw(row):
    qx, qy, qz, qw = (float(row[k]) for k in ("qx", "qy", "qz", "qw"))
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def align_service(log_path):
    """Start and end (Unix time) of the initial pose search, from the log of the pose initializer."""
    start = end = None
    with open(log_path, errors="replace") as f:
        for line in f:
            if "pose_initializer" not in line:
                continue
            stamp = re.search(r"\[(\d+\.\d+)\]", line)
            if stamp and "Call align server" in line:
                start = float(stamp.group(1))
            elif stamp and "align server succeeded" in line and start is not None:
                end = float(stamp.group(1))
    return (start, end) if start and end else (None, None)


def load_run(run_dir):
    metrics = read_csv(f"{run_dir}/metrics.csv")
    poses = read_csv(f"{run_dir}/poses.csv")
    resources = read_csv(f"{run_dir}/resources.csv")
    with open(f"{run_dir}/replay_start") as f:
        start = float(f.read())
    with open(f"{run_dir}/replay_end") as f:
        end = float(f.read())
    window = [r for r in resources if start <= float(r["time"]) <= end]
    align_start, align_end = align_service(f"{run_dir}/autoware.log")
    # After the initial pose search: only tracking, one alignment per scan.
    tracking = [r for r in window if align_end and float(r["time"]) > align_end + 1.0]
    cpu_seconds = 0.0
    for previous, current in zip(tracking, tracking[1:]):
        if number(current["ndt_cpu_percent"]) is not None:
            cpu_seconds += number(current["ndt_cpu_percent"]) / 100.0 * (float(current["time"]) - float(previous["time"]))
    return {
        "metrics": metrics,
        "poses": poses,
        "window": window,
        "start": start,
        "align_seconds": align_end - align_start if align_end else None,
        "tracking": tracking,
        "tracking_cpu_ms_per_second": 1000.0 * cpu_seconds / (float(tracking[-1]["time"]) - float(tracking[0]["time"])),
    }


def values(rows, key):
    return [number(r[key]) for r in rows if number(r.get(key)) is not None]


def summarize(runs):
    latency = [v for run in runs for v in values(run["poses"], "latency_ms")]
    exe = [v for run in runs for v in values(run["metrics"], "exe_time_ms")]
    window = [r for run in runs for r in run["window"]]
    align = [run["align_seconds"] for run in runs if run["align_seconds"]]
    return {
        "alignments": [len(run["metrics"]) for run in runs],
        "latency_median": statistics.median(latency),
        "latency_p95": percentile(latency, 95),
        "latency_p99": percentile(latency, 99),
        "latency_max": max(latency),
        "exe_median": statistics.median(exe),
        "iterations": statistics.mean(v for run in runs for v in values(run["metrics"], "iterations")),
        "tp": statistics.median(v for run in runs for v in values(run["metrics"], "tp")),
        "nvtl": statistics.median(v for run in runs for v in values(run["metrics"], "nvtl")),
        "nvtl_min": min(v for run in runs for v in values(run["metrics"], "nvtl")),
        "cpu_mean": statistics.mean(values(window, "ndt_cpu_percent")),
        "cpu_peak": max(values(window, "ndt_cpu_percent")),
        "gpu_mean": statistics.mean(values(window, "gpu_percent")),
        "gpu_peak": max(values(window, "gpu_percent")),
        "vram": max(values(window, "ndt_vram_mib")),
        "rss": statistics.median(values(window, "ndt_rss_mib")),
        "machine_cpu": statistics.mean(values(window, "machine_cpu_percent")),
        "align_seconds": statistics.median(align) if align else None,
        # The bag plays at 0.2x: the scans arrive at 2 Hz instead of 10 Hz. CPU time per second of bag time is 5x that.
        "tracking_cpu_ms_per_bag_second": 5.0 * statistics.mean(run["tracking_cpu_ms_per_second"] for run in runs),
        "tracking_gpu_mean": statistics.mean(values([r for run in runs for r in run["tracking"]], "gpu_percent")),
    }


def pose_differences(run, reference):
    by_stamp = {round(float(r["stamp"]), 3): r for r in reference["poses"]}
    position, heading = [], []
    for row in run["poses"]:
        other = by_stamp.get(round(float(row["stamp"]), 3))
        if other:
            position.append(math.dist([float(row[k]) for k in "xyz"], [float(other[k]) for k in "xyz"]))
            heading.append(abs((yaw(row) - yaw(other) + math.pi) % (2 * math.pi) - math.pi) * 180.0 / math.pi)
    return position, heading


def plot_latency(data, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    for mode, runs in data.items():
        run = runs[0]
        rows = [r for r in run["poses"] if number(r["latency_ms"]) is not None]
        t0 = float(rows[0]["stamp"])
        ax.plot([float(r["stamp"]) - t0 for r in rows], [float(r["latency_ms"]) for r in rows], label=MODES[mode][0], color=MODES[mode][1], linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("Bag time (s)")
    ax.set_ylabel("Scan in to pose out (ms, log scale)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def plot_resources(data, path):
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    keys = [("ndt_cpu_percent", "NDT process CPU (%, 100 = one core)"), ("gpu_percent", "Whole GPU utilization (%)"), ("ndt_vram_mib", "NDT process video memory (MiB)")]
    for ax, (key, title) in zip(axes, keys):
        for mode, runs in data.items():
            run = runs[0]
            rows = [r for r in run["window"] if number(r[key]) is not None]
            ax.plot([float(r["time"]) - run["start"] for r in rows], [float(r[key]) for r in rows], label=MODES[mode][0], color=MODES[mode][1], linewidth=1.2)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("Replay time (s, the bag plays at 0.2x)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def plot_summary(summary, path):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    names = [MODES[m][0].replace(" (", "\n(") for m in summary]
    colors = [MODES[m][1] for m in summary]
    titles = ("Median scan-to-pose time (ms)", "NDT CPU while tracking\n(ms per second of bag time)", "NDT video memory (MiB)")
    for ax, key, title in zip(axes, ("latency_median", "tracking_cpu_ms_per_bag_second", "vram"), titles):
        bars = ax.bar(names, [summary[m][key] for m in summary], color=colors)
        ax.bar_label(bars, fmt="%.1f", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def main(results, figures, names):
    data = {}
    for mode in MODES:
        runs = []
        for name in names:
            run_dir = f"{results}/{mode}/{name}"
            if os.path.exists(f"{run_dir}/metrics.csv"):
                runs.append(load_run(run_dir))
        if runs:
            data[mode] = runs

    summary = {mode: summarize(runs) for mode, runs in data.items()}
    for mode, s in summary.items():
        print(f"\n{mode} ({len(data[mode])} runs)")
        for key, value in s.items():
            print(f"  {key}: {value if not isinstance(value, float) else round(value, 3)}")

    print("\nDifference from the first run of Autoware NDT, matched by scan time")
    reference = data["builtin"][0]
    for mode, runs in data.items():
        for i, run in enumerate(runs):
            if mode == "builtin" and i == 0:
                continue
            position, heading = pose_differences(run, reference)
            print(f"  {mode} run {i + 1}: scans {len(position)}, position median {statistics.median(position) * 1000:.1f} mm, p95 {percentile(position, 95) * 1000:.1f} mm, max {max(position) * 1000:.1f} mm, heading max {max(heading):.3f} deg")

    os.makedirs(figures, exist_ok=True)
    plot_latency(data, f"{figures}/latency.png")
    plot_resources(data, f"{figures}/resources.png")
    plot_summary(summary, f"{figures}/summary.png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3].split(","))
