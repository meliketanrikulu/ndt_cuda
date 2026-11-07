# The Autoware image plus what cuda_ndt_matcher needs to build and run.
FROM ghcr.io/autowarefoundation/autoware:universe-devel-cuda-1.5.0
RUN apt-get update -qq && apt-get install -y --no-install-recommends libcusolver-dev-12-4 libnvjitlink-12-4 ros-humble-example-interfaces ros-humble-test-msgs
