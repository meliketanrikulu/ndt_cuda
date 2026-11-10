# The Autoware image plus what cuda_ndt_matcher needs to build and run.
ARG BASE=ghcr.io/autowarefoundation/autoware:universe-devel-cuda-humble-2.0.0-amd64
FROM ${BASE}
RUN apt-get update -qq && apt-get install -y --no-install-recommends libcusolver-dev-12-8 libnvjitlink-12-8 ros-humble-example-interfaces ros-humble-test-msgs
