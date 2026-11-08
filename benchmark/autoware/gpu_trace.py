#!/usr/bin/env python3
"""Computes how busy one process kept the GPU, from the Nsight Systems report of a run.

nvidia-smi reports whole percents, so a scan matcher that keeps the GPU busy for less than 1 % of the time shows 0.
The report has the start and end of every kernel, copy and memset instead. Their union is the time the GPU worked for
the process. Writes gpu_busy.csv in the run directory: per BIN seconds, the end time (Unix time) and the busy percent.

usage: gpu_trace.py <run dir> <process name> [nsys]
"""

import csv
import os
import sqlite3
import subprocess
import sys

BIN = 0.25
ACTIVITIES = ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET")


def export(run_dir, nsys):
    database = f"{run_dir}/gpu_trace.sqlite"
    if not os.path.exists(database):
        result = subprocess.run([nsys, "export", "--type=sqlite", f"--output={database}", f"{run_dir}/gpu_trace.nsys-rep"],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(result.stdout + result.stderr)
    return database


def busy_intervals(database, process):
    """Merged intervals in nanoseconds of Unix time during which the GPU ran work of the process."""
    db = sqlite3.connect(database)
    epoch = db.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME").fetchone()[0]
    # Linux keeps the first 15 characters of a process name.
    pids = [row[0] for row in db.execute("SELECT globalPid FROM PROCESSES WHERE name IN (?, ?)", (process, process[:15]))]
    if not pids:
        raise SystemExit(f"no process {process} in {database}")
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    marks = ",".join("?" * len(pids))
    intervals = []
    for table in ACTIVITIES:
        if table in tables:
            intervals += db.execute(f"SELECT start, end FROM {table} WHERE globalPid IN ({marks})", pids).fetchall()
    intervals.sort()
    # Kernels of several streams can overlap; the GPU is busy once for them.
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(epoch + start, epoch + end) for start, end in merged]


def main(run_dir, process, nsys="nsys"):
    intervals = busy_intervals(export(run_dir, nsys), process)
    # Every scan of the replay runs on the GPU, so a trace that ends early lost its last buffers.
    with open(f"{run_dir}/replay_end") as f:
        if intervals[-1][1] / 1e9 < float(f.read()) - 10.0:
            raise SystemExit(f"the trace of {process} ends before the replay; CUDA events were lost")
    width = int(BIN * 1e9)
    first = intervals[0][0]
    busy = [0] * ((intervals[-1][1] - first) // width + 1)
    for a, b in intervals:
        # An interval can cross the border of a bin.
        while a < b:
            i = (a - first) // width
            edge = min(b, first + (i + 1) * width)
            busy[i] += edge - a
            a = edge
    with open(f"{run_dir}/gpu_busy.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "busy_percent"])
        for i, nanoseconds in enumerate(busy):
            writer.writerow([round((first + (i + 1) * width) / 1e9, 3), round(100.0 * nanoseconds / width, 4)])
    print(f"{process}: {len(intervals)} busy intervals, {sum(busy) / 1e6:.1f} ms of GPU time")


if __name__ == "__main__":
    main(*sys.argv[1:])
