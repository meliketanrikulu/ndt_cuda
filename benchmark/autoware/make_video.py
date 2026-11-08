#!/usr/bin/env python3
"""Writes the CPU, GPU and video memory use of the scan matcher into the corner of a recorded run.

CPU and video memory come from resources.csv, the GPU use from gpu_busy.csv (see gpu_trace.py), all in Unix time;
video_start maps them to the time of the screen recording. Runs on the host.

usage: make_video.py <run dir> <label> <output.mp4>
"""

import bisect
import csv
import os
import subprocess
import sys

STEP = 0.5
# The GPU column is the busy share of the last second: one scan matcher call is shorter than a frame.
GPU_WINDOW = 1.0


def read_series(path, key):
    with open(path) as f:
        rows = [(float(r["time"]), r[key]) for r in csv.DictReader(f)]
    return [t for t, _ in rows], [float(v) if v else None for _, v in rows]


def latest(series, t):
    times, values = series
    i = bisect.bisect_right(times, t) - 1
    return values[i] if i >= 0 else None


def window_mean(series, start, end):
    times, values = series
    selected = values[bisect.bisect_right(times, start):bisect.bisect_right(times, end)]
    return sum(selected) / len(selected) if selected else 0.0


def clock(seconds):
    return f"{int(seconds // 3600)}:{int(seconds % 3600 // 60):02d}:{seconds % 60:05.2f}"


def overlay(run_dir, label, duration):
    with open(f"{run_dir}/video_start") as f:
        start = float(f.read())
    cpu = read_series(f"{run_dir}/resources.csv", "ndt_cpu_percent")
    vram = read_series(f"{run_dir}/resources.csv", "ndt_vram_mib")
    gpu = read_series(f"{run_dir}/gpu_busy.csv", "busy_percent") if os.path.exists(f"{run_dir}/gpu_busy.csv") else None

    lines = [
        "[Script Info]", "ScriptType: v4.00+", "PlayResX: 1920", "PlayResY: 1080", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV",
        # BorderStyle 3 draws a box behind the text, black at 65 % opacity.
        "Style: Default,DejaVu Sans Mono,28,&H00FFFFFF,&H59000000,&H59000000,1,3,10,0,9,30,30,30", "",
        "[Events]", "Format: Layer, Start, End, Style, Text",
    ]
    t = 0.0
    while t < duration:
        now = start + t
        cpu_now, vram_now = latest(cpu, now), latest(vram, now)
        gpu_text = f"{window_mean(gpu, now - GPU_WINDOW, now):6.2f} %" if gpu else "   not used"
        rows = [
            label,
            f"CPU  {cpu_now:6.1f} %" if cpu_now is not None else "CPU      --",
            f"GPU  {gpu_text}",
            f"VRAM {vram_now:6.0f} MiB" if vram_now is not None else "VRAM     --",
        ]
        # The box is right-aligned line by line; lines of one width in a monospace font make it one block. \h is a
        # space that libass keeps at the end of a line.
        width = max(len(row) for row in rows)
        text = "\\N".join(row.ljust(width).replace(" ", "\\h") for row in rows)
        lines.append(f"Dialogue: 0,{clock(t)},{clock(min(t + STEP, duration))},Default,{text}")
        t += STEP
    path = f"{run_dir}/overlay.ass"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main(run_dir, label, output):
    screen = f"{run_dir}/screen.mp4"
    duration = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", screen],
                                    capture_output=True, text=True, check=True).stdout)
    subtitles = overlay(run_dir, label, duration)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", screen, "-vf", f"ass={subtitles}",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", output], check=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
