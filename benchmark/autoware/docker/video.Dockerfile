# Adds a virtual screen and a video encoder to a benchmark image, to record RViz without a display.
ARG BASE=ghcr.io/autowarefoundation/autoware:universe-devel-cuda-1.5.0
FROM ${BASE}
RUN apt-get update -qq && apt-get install -y --no-install-recommends xvfb xdotool ffmpeg fonts-dejavu-core
