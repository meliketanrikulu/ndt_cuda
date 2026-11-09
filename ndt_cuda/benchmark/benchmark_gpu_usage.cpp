// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Melike
//
// Measures the video memory and GPU utilization of ndt_cuda with nvidia-smi, for docs/benchmark.md.
// Every measurement runs in its own process, so that it starts without a CUDA context and without freed memory.
//
// The full run takes about 6 minutes. It should be the only program using the GPU.
//
// usage: ndt_cuda_benchmark_gpu_usage [data_dir]

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <ndt_cuda/ndt_cuda.hpp>

using PointT = pcl::PointXYZ;
using Cloud = pcl::PointCloud<PointT>;
using Ndt = ndt_cuda::NormalDistributionsTransformCUDA<PointT, PointT>;
using Clock = std::chrono::steady_clock;

static std::string g_data_dir;

static double elapsedMs(Clock::time_point from) {
  return std::chrono::duration<double, std::milli>(Clock::now() - from).count();
}

static std::string runCommand(const std::string& command) {
  std::string output;
  FILE* pipe = popen(command.c_str(), "r");
  if (!pipe) return output;
  char buffer[256];
  while (std::fgets(buffer, sizeof(buffer), pipe)) output += buffer;
  pclose(pipe);
  return output;
}

/** \brief Video memory of this process in MiB as reported by the driver, -1 if not available. */
static int processMemoryMiB() {
  const std::string output = runCommand("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits");
  const std::string pid = std::to_string(getpid());
  std::size_t begin = 0;
  while (begin < output.size()) {
    const std::size_t end = output.find('\n', begin);
    const std::string line = output.substr(begin, end - begin);
    const std::size_t comma = line.find(',');
    if (comma != std::string::npos && line.substr(0, comma) == pid) return std::atoi(line.c_str() + comma + 1);
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return -1;
}

/** \brief Video memory in use on the whole GPU in MiB, by all processes and the driver. */
static int deviceMemoryMiB() {
  return std::atoi(runCommand("nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits").c_str());
}

static Cloud::Ptr loadDownsampled(const std::string& path) {
  Cloud::Ptr cloud(new Cloud), filtered(new Cloud);
  if (pcl::io::loadPCDFile(path, *cloud)) {
    std::fprintf(stderr, "failed to load %s\n", path.c_str());
    std::exit(1);
  }
  pcl::VoxelGrid<PointT> voxelgrid;
  voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);
  voxelgrid.setInputCloud(cloud);
  voxelgrid.filter(*filtered);
  return filtered;
}

/** \brief Synthetic map with about num_voxels occupied 1 m voxels, each holding 8 random points. */
static Cloud::Ptr syntheticMap(std::size_t num_voxels) {
  int side = 1;
  while (static_cast<std::size_t>(side) * side * side < num_voxels) side++;
  Cloud::Ptr map(new Cloud);
  map->reserve(num_voxels * 8);
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> offset(0.05f, 0.95f);
  std::size_t voxels = 0;
  for (int x = 0; x < side && voxels < num_voxels; x++) {
    for (int y = 0; y < side && voxels < num_voxels; y++) {
      for (int z = 0; z < side && voxels < num_voxels; z++, voxels++) {
        for (int i = 0; i < 8; i++) {
          PointT pt;
          pt.x = x + offset(rng);
          pt.y = y + offset(rng);
          pt.z = z + offset(rng);
          map->push_back(pt);
        }
      }
    }
  }
  return map;
}

/** \brief Synthetic scan of n points picked at random from the map, with 5 mm of Gaussian noise on each axis. */
static Cloud::Ptr sampleWithNoise(const Cloud& map, std::size_t n) {
  Cloud::Ptr scan(new Cloud);
  scan->reserve(n);
  std::mt19937 rng(2);
  std::uniform_int_distribution<std::size_t> pick(0, map.size() - 1);
  std::normal_distribution<float> noise(0.0f, 0.005f);
  for (std::size_t i = 0; i < n; i++) {
    PointT pt = map[pick(rng)];
    pt.x += noise(rng);
    pt.y += noise(rng);
    pt.z += noise(rng);
    scan->push_back(pt);
  }
  return scan;
}

/** \brief Map and scan of one measurement. num_voxels == 0 selects the sample map; so does num_points == 0 for the sample scan. */
static void makeClouds(std::size_t num_voxels, std::size_t num_points, Cloud::Ptr& map, Cloud::Ptr& scan) {
  map = num_voxels ? syntheticMap(num_voxels) : loadDownsampled(g_data_dir + "/251370668.pcd");
  scan = num_points ? sampleWithNoise(*map, num_points) : loadDownsampled(g_data_dir + "/251371071.pcd");
}

static void measureMemory(std::size_t num_voxels, std::size_t num_points) {
  Cloud::Ptr map, scan;
  makeClouds(num_voxels, num_points, map, scan);
  const int device_before = deviceMemoryMiB();

  // The kd-tree is built on the CPU and uses no video memory, but takes minutes on the largest maps.
  Ndt ndt;
  ndt.setBuildTargetKdTree(false);
  ndt.setResolution(1.0);
  const int context = processMemoryMiB();

  ndt.setInputTarget(map);
  const int with_map = processMemoryMiB();

  ndt.setInputSource(scan);
  Cloud aligned;
  ndt.align(aligned);
  const int with_scan = processMemoryMiB();
  const int device_increase = deviceMemoryMiB() - device_before;

  // Sizes of the GPU buffers, see GpuBackend: voxels, open addressing hash table with load factor <= 0.5, scan.
  const std::size_t voxels = ndt.getNumTargetVoxels();
  std::size_t slots = 1024;
  while (slots < 2 * voxels) slots <<= 1;
  const double map_mib = (voxels * 48.0 + slots * 8.0) / (1024.0 * 1024.0);
  const double scan_mib = scan->size() * 16.0 / (1024.0 * 1024.0);

  std::printf("%10zu %10zu %10zu %9d %9d %9d %9d %11.1f %11.1f\n", map->size(), voxels, scan->size(), context, with_map, with_scan, device_increase, map_mib, scan_mib);
}

/** \brief rate_hz: calls per second, 0 for as fast as possible, negative for no calls (idle GPU). */
static void measureUtilization(std::size_t num_voxels, std::size_t num_points, double rate_hz, double seconds) {
  Cloud::Ptr map, scan;
  makeClouds(num_voxels, num_points, map, scan);

  Ndt ndt;
  ndt.setBuildTargetKdTree(false);
  ndt.setResolution(1.0);
  ndt.setInputTarget(map);
  ndt.setInputSource(scan);

  // The driver averages utilization over up to one second: the uploads must not show up in the first sample.
  Cloud aligned;
  ndt.align(aligned);
  std::this_thread::sleep_for(std::chrono::seconds(2));

  std::atomic<bool> stop(false);
  std::vector<int> gpu_samples, memory_samples;
  std::thread sampler([&] {
    while (!stop) {
      int gpu = 0, memory = 0;
      const std::string output = runCommand("nvidia-smi -i 0 --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits");
      if (std::sscanf(output.c_str(), "%d, %d", &gpu, &memory) == 2) {
        gpu_samples.push_back(gpu);
        memory_samples.push_back(memory);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
  });

  double align_ms = 0.0;
  int calls = 0;
  const auto start = Clock::now();
  auto next = start;
  while (elapsedMs(start) < seconds * 1000.0) {
    if (rate_hz < 0.0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }
    const auto t0 = Clock::now();
    ndt.align(aligned);
    align_ms += elapsedMs(t0);
    calls++;
    if (rate_hz > 0.0) {
      next += std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / rate_hz));
      std::this_thread::sleep_until(next);
    }
  }
  const double total_ms = elapsedMs(start);
  stop = true;
  sampler.join();

  const auto mean = [](const std::vector<int>& v) {
    double sum = 0.0;
    for (const int x : v) sum += x;
    return v.empty() ? 0.0 : sum / v.size();
  };
  char rate[16];
  std::snprintf(rate, sizeof(rate), rate_hz > 0.0 ? "%.0f Hz" : rate_hz < 0.0 ? "idle" : "max", rate_hz);
  std::printf("%10zu %10zu %8s %7.1f %10.2f %9.1f %9.1f %8d %10.1f %8zu\n", ndt.getNumTargetVoxels(), scan->size(), rate, calls * 1000.0 / total_ms, calls ? align_ms / calls : 0.0, 100.0 * align_ms / total_ms, mean(gpu_samples),
              gpu_samples.empty() ? 0 : *std::max_element(gpu_samples.begin(), gpu_samples.end()), mean(memory_samples), gpu_samples.size());
}

/** \brief Runs a measurement in a new process. CUDA is never initialized in the parent, so this fork is safe. */
static void inChildProcess(const std::function<void()>& measurement) {
  std::fflush(stdout);
  const pid_t pid = fork();
  if (pid == 0) {
    measurement();
    std::fflush(stdout);
    _exit(0);
  }
  int status = 0;
  waitpid(pid, &status, 0);
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::fprintf(stderr, "a measurement failed\n");
    std::exit(1);
  }
}

int main(int argc, char** argv) {
  g_data_dir = argc > 1 ? argv[1] : NDT_CUDA_DATA_DIR;

  if (runCommand("nvidia-smi --query-gpu=name --format=csv,noheader").empty()) {
    std::fprintf(stderr, "nvidia-smi is needed for this measurement\n");
    return 1;
  }
  std::printf("GPU: %s", runCommand("nvidia-smi -i 0 --query-gpu=name,memory.total,driver_version --format=csv,noheader").c_str());
  std::printf("Used by other programs: %s\n", runCommand("nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader").c_str());

  std::printf("[memory] video memory of the process in MiB; device = increase on the whole GPU; exp. = calculated size of the buffers\n");
  std::printf("%10s %10s %10s %9s %9s %9s %9s %11s %11s\n", "map points", "voxels", "scan pts", "context", "+ map", "+ scan", "device", "exp. map", "exp. scan");
  const std::size_t memory_cases[][2] = {{0, 0}, {100000, 100000}, {1000000, 100000}, {5000000, 100000}, {1000000, 1000000}, {1000000, 5000000}};
  for (const auto& c : memory_cases) {
    inChildProcess([&] { measureMemory(c[0], c[1]); });
  }

  std::printf("\n[utilization] percentages from nvidia-smi, sampled every 250 ms; busy = share of time spent inside align(), CPU part included\n");
  std::printf("%10s %10s %8s %7s %10s %9s %9s %8s %10s %8s\n", "voxels", "scan pts", "rate", "calls/s", "align ms", "busy %", "gpu %", "gpu max", "memory %", "samples");
  inChildProcess([&] { measureUtilization(0, 0, -1.0, 30.0); });
  const std::size_t utilization_cases[][2] = {{0, 0}, {0, 100000}, {1000000, 100000}};
  for (const auto& c : utilization_cases) {
    for (const double rate : {10.0, 20.0, 0.0}) {
      inChildProcess([&] { measureUtilization(c[0], c[1], rate, 30.0); });
    }
  }
  return 0;
}
