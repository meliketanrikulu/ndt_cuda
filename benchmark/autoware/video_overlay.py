#!/usr/bin/env python3
"""Keeps a text file with the CPU, GPU and video memory use of one process up to date, for the overlay of a video."""

import glob
import os
import collections
import subprocess
import sys
import threading
import time

TICKS = os.sysconf("SC_CLK_TCK")


def find_process(name):
    for path in glob.glob("/proc/[0-9]*/comm"):
        try:
            with open(path) as f:
                if f.read().strip() == name[:15]:
                    return int(path.split("/")[2])
        except OSError:
            pass
    return None


def cpu_seconds(pid):
    with open(f"/proc/{pid}/stat") as f:
        fields = f.read().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / TICKS


class GpuMonitor:
    """Reads the compute utilization of each process from nvidia-smi once per second. It reports whole percents, so
    an average over the last WINDOW seconds shows the use of a process that keeps the GPU busy for less than 1 %."""

    WINDOW = 3

    def __init__(self):
        self.samples = collections.defaultdict(lambda: collections.deque(maxlen=self.WINDOW))
        self.lock = threading.Lock()
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        monitor = subprocess.Popen(["nvidia-smi", "pmon", "-s", "u", "-d", "1"], stdout=subprocess.PIPE, text=True)
        seen = set()
        for line in monitor.stdout:
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            # One block per second; a process missing from a block or shown with "-" did not use the GPU.
            if fields[1] in seen:
                self.close_block(seen)
                seen = set()
            if fields[1].isdigit():
                seen.add(fields[1])
                with self.lock:
                    self.samples[fields[1]].append(0.0 if fields[3] == "-" else float(fields[3]))

    def close_block(self, seen):
        with self.lock:
            for pid, values in self.samples.items():
                if pid not in seen:
                    values.append(0.0)

    def percent(self, pid):
        with self.lock:
            values = self.samples.get(str(pid))
            return sum(values) / len(values) if values else 0.0


def video_memory(pid):
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        fields = line.split(",")
        if len(fields) == 2 and fields[0].strip() == str(pid):
            return float(fields[1])
    return 0.0


def write(path, text):
    # ffmpeg rereads the file for every frame, so it must never see a partial one.
    with open(path + ".tmp", "w") as f:
        f.write(text)
    os.replace(path + ".tmp", path)


def main(label, process, path):
    gpu = GpuMonitor()
    pid = None
    last_cpu = last_time = None
    write(path, f"{label}\nCPU   --\nGPU   --\nVRAM  --\n")
    while True:
        time.sleep(0.5)
        try:
            if pid is None or not os.path.exists(f"/proc/{pid}"):
                pid = find_process(process)
                last_cpu = None
                continue
            cpu, now = cpu_seconds(pid), time.time()
            if last_cpu is not None:
                percent = 100.0 * (cpu - last_cpu) / (now - last_time)
                write(path, f"{label}\nCPU  {percent:5.1f} %\nGPU  {gpu.percent(pid):5.1f} %\nVRAM {video_memory(pid):5.0f} MiB\n")
            last_cpu, last_time = cpu, now
        except OSError:
            pid = None


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
