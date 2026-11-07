// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Melike
//
// Produces the measurements of docs/benchmark.md: repeated timings on the sample data and on synthetic large clouds,
// ndt_omp thread scaling, agreement with ndt_omp, determinism, getFitnessScore() and the cost of a map change.
//
// The full run takes about 10 minutes, nearly all of it in ndt_omp on the largest clouds.
//
// usage: ndt_cuda_benchmark_report [data_dir] [max_source_points]   (max_source_points defaults to 5000000)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <omp.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pclomp/ndt_omp.h>
#include <ndt_cuda/ndt_cuda.hpp>

using PointT = pcl::PointXYZ;
using Cloud = pcl::PointCloud<PointT>;
using NdtGpu = ndt_cuda::NormalDistributionsTransformCUDA<PointT, PointT>;
using NdtOmp = pclomp::NormalDistributionsTransform<PointT, PointT>;

static const char* const kMethods[] = {"KDTREE", "DIRECT26", "DIRECT7", "DIRECT1"};

struct Timing {
  double median, p10, p90;
};

static double nowMs() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static Cloud::Ptr loadDownsampled(const std::string& path) {
  Cloud::Ptr cloud(new Cloud), filtered(new Cloud);
  if (pcl::io::loadPCDFile(path, *cloud)) {
    return nullptr;
  }
  pcl::VoxelGrid<PointT> voxelgrid;
  voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);
  voxelgrid.setInputCloud(cloud);
  voxelgrid.filter(*filtered);
  return filtered;
}

/** \brief Synthetic cloud of n points: repeats the input with 5 mm of Gaussian noise on each axis. */
static Cloud::Ptr repeatWithNoise(const Cloud& input, std::size_t n) {
  Cloud::Ptr output(new Cloud);
  std::mt19937 rng(1);
  std::normal_distribution<float> noise(0.0f, 0.005f);
  output->reserve(n);
  while (output->size() < n) {
    for (const auto& pt : input) {
      if (output->size() >= n) break;
      PointT q = pt;
      q.x += noise(rng);
      q.y += noise(rng);
      q.z += noise(rng);
      output->push_back(q);
    }
  }
  return output;
}

static void transformDiff(const Eigen::Matrix4f& a, const Eigen::Matrix4f& b, double& dt, double& dr_deg) {
  const Eigen::Matrix4d d = a.cast<double>().inverse() * b.cast<double>();
  dt = d.block<3, 1>(0, 3).norm();
  const double c = std::max(-1.0, std::min(1.0, (d.block<3, 3>(0, 0).trace() - 1.0) / 2.0));
  dr_deg = std::acos(c) * 180.0 / M_PI;
}

template <typename Registration>
static Timing timeAlign(Registration& reg, const Cloud::Ptr& source, int warmup, int repeats) {
  Cloud aligned;
  reg.setInputSource(source);
  for (int i = 0; i < warmup; i++) {
    reg.align(aligned);
  }
  std::vector<double> times;
  for (int i = 0; i < repeats; i++) {
    const double t0 = nowMs();
    reg.align(aligned);
    times.push_back(nowMs() - t0);
  }
  std::sort(times.begin(), times.end());
  return {times[times.size() / 2], times[times.size() / 10], times[times.size() * 9 / 10]};
}

static void configure(NdtOmp& omp, NdtGpu& gpu, int mode, const Cloud::Ptr& target) {
  omp.setNumThreads(omp_get_max_threads());
  omp.setResolution(1.0);
  gpu.setResolution(1.0);
  omp.setNeighborhoodSearchMethod(static_cast<pclomp::NeighborSearchMethod>(mode));
  gpu.setNeighborhoodSearchMethod(static_cast<ndt_cuda::NeighborSearchMethod>(mode));
  omp.setInputTarget(target);
  gpu.setInputTarget(target);
}

static void measureScaling(const Cloud::Ptr& target, const Cloud::Ptr& source, std::size_t max_points) {
  std::printf("\n[timing] align() in ms, median (p10-p90)\n");
  std::printf("%-9s %-9s %28s %28s %9s %6s %10s %9s\n", "points", "method", "ndt_omp", "ndt_cuda", "speedup", "iters", "dt[um]", "dr[deg]");

  for (const std::size_t n : {std::size_t(0), std::size_t(100000), std::size_t(300000), std::size_t(1000000), std::size_t(5000000)}) {
    if (n > max_points) continue;
    const Cloud::Ptr cloud = n ? repeatWithNoise(*source, n) : source;
    const int repeats = n >= 5000000 ? 7 : n >= 1000000 ? 15 : n ? 31 : 101;

    for (int mode = 0; mode < 4; mode++) {
      NdtOmp omp;
      NdtGpu gpu;
      configure(omp, gpu, mode, target);
      // DIRECT26 is by far the slowest method in ndt_omp.
      const Timing t_omp = timeAlign(omp, cloud, 2, (mode == pclomp::DIRECT26 && n >= 1000000) ? 5 : repeats);
      const Timing t_gpu = timeAlign(gpu, cloud, 5, repeats);

      double dt, dr;
      transformDiff(omp.getFinalTransformation(), gpu.getFinalTransformation(), dt, dr);
      std::printf("%-9zu %-9s %10.2f (%7.2f-%7.2f) %10.2f (%7.2f-%7.2f) %8.1fx %3d/%-2d %10.1f %9.5f\n", cloud->size(), kMethods[mode], t_omp.median, t_omp.p10, t_omp.p90, t_gpu.median, t_gpu.p10, t_gpu.p90,
                  t_omp.median / t_gpu.median, omp.getFinalNumIteration(), gpu.getFinalNumIteration(), dt * 1e6, dr);
      std::fflush(stdout);
    }
  }
}

static void measureThreads(const Cloud::Ptr& target, const Cloud::Ptr& source) {
  std::printf("\n[threads] ndt_omp DIRECT7 on the sample data\n");
  std::vector<int> counts = {1, 4, 8, 16, omp_get_max_threads()};
  std::sort(counts.begin(), counts.end());
  counts.erase(std::unique(counts.begin(), counts.end()), counts.end());
  for (const int threads : counts) {
    if (threads > omp_get_max_threads()) continue;
    NdtOmp omp;
    omp.setNumThreads(threads);
    omp.setResolution(1.0);
    omp.setNeighborhoodSearchMethod(pclomp::DIRECT7);
    omp.setInputTarget(target);
    const Timing t = timeAlign(omp, source, 2, 51);
    std::printf("%2d threads: %.2f ms (%.2f-%.2f)\n", threads, t.median, t.p10, t.p90);
  }
}

static void measureDeterminism(const Cloud::Ptr& target, const Cloud::Ptr& source) {
  std::printf("\n[determinism] 200 repeats against the first result\n");
  for (int mode = 0; mode < 4; mode++) {
    NdtGpu gpu;
    gpu.setResolution(1.0);
    gpu.setNeighborhoodSearchMethod(static_cast<ndt_cuda::NeighborSearchMethod>(mode));
    gpu.setInputTarget(target);
    gpu.setInputSource(source);
    Cloud aligned;
    gpu.align(aligned);
    const Eigen::Matrix4f first = gpu.getFinalTransformation();
    const double probability = gpu.getTransformationProbability();
    int different = 0;
    for (int i = 0; i < 200; i++) {
      gpu.align(aligned);
      const Eigen::Matrix4f current = gpu.getFinalTransformation();
      different += std::memcmp(first.data(), current.data(), sizeof(float) * 16) != 0 || probability != gpu.getTransformationProbability();
    }
    std::printf("%-9s differing results: %d\n", kMethods[mode], different);
  }
}

static void measurePerturbedGuess(const Cloud::Ptr& target, const Cloud::Ptr& source) {
  std::printf("\n[perturbed guess] about 0.95 m and 2.9 deg\n");
  Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();
  guess.block<3, 3>(0, 0) = Eigen::AngleAxisf(0.05f, Eigen::Vector3f::UnitZ()).toRotationMatrix();
  guess.block<3, 1>(0, 3) = Eigen::Vector3f(0.8f, -0.5f, 0.1f);
  for (int mode = 0; mode < 4; mode++) {
    NdtOmp omp;
    NdtGpu gpu;
    configure(omp, gpu, mode, target);
    omp.setInputSource(source);
    gpu.setInputSource(source);
    Cloud aligned;
    omp.align(aligned, guess);
    gpu.align(aligned, guess);
    double dt, dr;
    transformDiff(omp.getFinalTransformation(), gpu.getFinalTransformation(), dt, dr);
    std::printf("%-9s iters %d/%d  dt %.1f um  dr %.5f deg  fitness %.6f/%.6f\n", kMethods[mode], omp.getFinalNumIteration(), gpu.getFinalNumIteration(), dt * 1e6, dr, omp.getFitnessScore(), gpu.getFitnessScore());
  }
}

template <typename Registration>
static void measureMapChange(Registration& reg, const char* name, const Cloud::Ptr& map, const Cloud::Ptr& source) {
  reg.setResolution(1.0);
  const double t0 = nowMs();
  reg.setInputTarget(map);
  const double t1 = nowMs();
  reg.setInputSource(source);
  Cloud aligned;
  const double t2 = nowMs();
  reg.align(aligned);
  const double t3 = nowMs();
  reg.align(aligned);
  const double t4 = nowMs();
  std::printf("%-9zu %-9s %18.1f %14.2f %15.2f\n", map->size(), name, t1 - t0, t3 - t2, t4 - t3);
}

static void measureMapChanges(const Cloud::Ptr& target, const Cloud::Ptr& source) {
  std::printf("\n[map change] DIRECT7, times in ms, single measurements\n");
  std::printf("%-9s %-9s %18s %14s %15s\n", "points", "impl", "setInputTarget", "first align", "second align");
  for (const std::size_t n : {std::size_t(1000000), std::size_t(3000000)}) {
    const Cloud::Ptr map = repeatWithNoise(*target, n);
    NdtGpu gpu;
    measureMapChange(gpu, "ndt_cuda", map, source);
    NdtGpu gpu_no_kdtree;
    gpu_no_kdtree.setBuildTargetKdTree(false);
    measureMapChange(gpu_no_kdtree, "no_kdtree", map, source);
    NdtOmp omp;
    omp.setNumThreads(omp_get_max_threads());
    measureMapChange(omp, "ndt_omp", map, source);
    std::printf("%-9s voxels: %zu\n", "", gpu.getNumTargetVoxels());
  }
}

int main(int argc, char** argv) {
  const std::string data_dir = argc > 1 ? argv[1] : NDT_CUDA_DATA_DIR;
  const std::size_t max_points = argc > 2 ? std::stoull(argv[2]) : 5000000;

  const Cloud::Ptr target = loadDownsampled(data_dir + "/251370668.pcd");
  const Cloud::Ptr source = loadDownsampled(data_dir + "/251371071.pcd");
  if (!target || !source) {
    std::fprintf(stderr, "failed to load the sample clouds from %s\n", data_dir.c_str());
    return 1;
  }
  if (ndt_cuda::GpuBackend::deviceCount() <= 0) {
    std::fprintf(stderr, "no CUDA device found\n");
    return 1;
  }

  std::printf("GPU          : %s\n", ndt_cuda::GpuBackend::deviceName(0).c_str());
  std::printf("CPU threads  : %d\n", omp_get_max_threads());
  std::printf("target/source: %zu / %zu points, resolution 1 m\n", target->size(), source->size());

  measureScaling(target, source, max_points);
  measureThreads(target, source);
  measureDeterminism(target, source);
  measurePerturbedGuess(target, source);
  measureMapChanges(target, source);
  return 0;
}
