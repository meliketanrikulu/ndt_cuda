#!/usr/bin/env python3
"""Samples the CPU of the NDT process and the use of the GPU to a CSV until stopped. Runs on the host."""

import csv
import glob
import os
import signal
import subprocess
import sys
import time

TICKS = os.sysconf("SC_CLK_TCK")
PAGE = os.sysconf("SC_PAGE_SIZE")
PERIOD = 0.5


def find_process(name):
    for path in glob.glob("/proc/[0-9]*/comm"):
        try:
            with open(path) as f:
                if f.read().strip() == name[:15]:
                    return int(path.split("/")[2])
        except OSError:
            pass
    return None


def process_stat(pid):
    """CPU seconds used and resident memory in MiB."""
    with open(f"/proc/{pid}/stat") as f:
        fields = f.read().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / TICKS, int(fields[21]) * PAGE / 2**20


def machine_cpu():
    """Busy and total CPU seconds of the whole machine."""
    with open("/proc/stat") as f:
        values = [int(v) for v in f.readline().split()[1:9]]
    return (sum(values) - values[3] - values[4]) / TICKS, sum(values) / TICKS


def gpu_stat():
    out = subprocess.run(
        ["nvidia-smi", "-i", "0", "--query-gpu=utilization.gpu,utilization.memory,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout
    return [float(v) for v in out.strip().split(",")]


def process_vram(pid):
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout
    for line in out.splitlines():
        fields = line.split(",")
        if len(fields) == 2 and fields[0].strip() == str(pid):
            return float(fields[1])
    return 0.0


def main(process_name, output_path):
    running = [True]
    signal.signal(signal.SIGTERM, lambda *_: running.clear())
    signal.signal(signal.SIGINT, lambda *_: running.clear())

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "ndt_cpu_percent", "ndt_rss_mib", "machine_cpu_percent", "gpu_percent", "gpu_memory_percent", "vram_mib", "ndt_vram_mib"])
        pid = None
        last_cpu = last_time = None
        last_busy, last_total = machine_cpu()
        while running:
            time.sleep(PERIOD)
            now = time.time()
            if pid is None or not os.path.exists(f"/proc/{pid}"):
                pid = find_process(process_name)
                last_cpu = None
            cpu_percent = rss = ""
            if pid is not None:
                try:
                    cpu, rss = process_stat(pid)
                    if last_cpu is not None:
                        cpu_percent = round(100.0 * (cpu - last_cpu) / (now - last_time), 1)
                    last_cpu, last_time = cpu, now
                    rss = round(rss, 1)
                except OSError:
                    pid = None
            busy, total = machine_cpu()
            machine = round(100.0 * (busy - last_busy) / max(total - last_total, 1e-9), 1)
            last_busy, last_total = busy, total
            gpu, gpu_memory, vram = gpu_stat()
            writer.writerow([round(now, 3), cpu_percent, rss, machine, gpu, gpu_memory, vram, process_vram(pid) if pid else 0.0])
            f.flush()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
