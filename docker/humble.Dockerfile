# ROS 2 Humble + CUDA build and test environment for ndt_cuda.
#   docker build -f docker/humble.Dockerfile -t ndt_cuda:humble .
#   docker run --rm --gpus all ndt_cuda:humble    # runs the example on the sample data (needs a GPU)
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates gnupg git \
 && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" > /etc/apt/sources.list.d/ros2.list \
 && apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-ros-base python3-colcon-common-extensions libpcl-dev libeigen3-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /ros2_ws
COPY . src/ndt_cuda

# Build (all default GPU architectures) and run the GPU-less validation test
RUN . /opt/ros/humble/setup.sh \
 && colcon build --packages-select ndt_cuda --cmake-args -DCMAKE_BUILD_TYPE=Release -DNDT_CUDA_BUILD_TESTS=ON \
 && colcon test --packages-select ndt_cuda --event-handlers console_direct+ \
 && colcon test-result --verbose

CMD ["bash", "-c", ". /ros2_ws/install/setup.sh && /ros2_ws/install/ndt_cuda/lib/ndt_cuda/ndt_cuda_align /ros2_ws/src/ndt_cuda/ndt_cuda/data/251370668.pcd /ros2_ws/src/ndt_cuda/ndt_cuda/data/251371071.pcd"]
