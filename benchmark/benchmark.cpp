// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Melike
//
// Compares pclomp::NormalDistributionsTransform (koide3/ndt_omp, OpenMP) with ndt_cuda::NormalDistributionsTransformCUDA:
// processing time, iterations, score and the difference between the estimated transformations.
//
// usage: ndt_cuda_benchmark [target.pcd source.pcd] [resolution]   (defaults to the scans in data/)

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <omp.h>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pclomp/ndt_omp.h>
#include <ndt_cuda/ndt_cuda.hpp>

using PointT = pcl::PointXYZ;
using Cloud = pcl::PointCloud<PointT>;

struct RunResult {
  Eigen::Matrix4f transform;
  double first_ms;
  double avg_ms;
  int iterations;
  double trans_probability;
  bool converged;
};

template <typename Registration>
RunResult run(Registration& reg, const Cloud::Ptr& target, const Cloud::Ptr& source, int repeats) {
  reg.setInputTarget(target);
  reg.setInputSource(source);
  Cloud aligned;

  auto t0 = std::chrono::steady_clock::now();
  reg.align(aligned);
  auto t1 = std::chrono::steady_clock::now();
  for (int i = 0; i < repeats; i++) {
    reg.align(aligned);
  }
  auto t2 = std::chrono::steady_clock::now();

  RunResult r;
  r.transform = reg.getFinalTransformation();
  r.first_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  r.avg_ms = std::chrono::duration<double, std::milli>(t2 - t1).count() / repeats;
  r.iterations = reg.getFinalNumIteration();
  r.trans_probability = reg.getTransformationProbability();
  r.converged = reg.hasConverged();
  return r;
}

void transformDiff(const Eigen::Matrix4f& a, const Eigen::Matrix4f& b, double& dt, double& dr_deg) {
  const Eigen::Matrix4d d = a.cast<double>().inverse() * b.cast<double>();
  dt = d.block<3, 1>(0, 3).norm();
  const double c = std::max(-1.0, std::min(1.0, (d.block<3, 3>(0, 0).trace() - 1.0) / 2.0));
  dr_deg = std::acos(c) * 180.0 / M_PI;
}

int main(int argc, char** argv) {
  std::string target_pcd = NDT_CUDA_DATA_DIR "/251370668.pcd";
  std::string source_pcd = NDT_CUDA_DATA_DIR "/251371071.pcd";
  float resolution = 1.0f;
  if (argc >= 3) {
    target_pcd = argv[1];
    source_pcd = argv[2];
  }
  if (argc >= 4) {
    resolution = std::stof(argv[3]);
  }

  Cloud::Ptr target(new Cloud), source(new Cloud);
  if (pcl::io::loadPCDFile(target_pcd, *target) || pcl::io::loadPCDFile(source_pcd, *source)) {
    std::cerr << "failed to load " << target_pcd << " / " << source_pcd << std::endl;
    std::cerr << "usage: ndt_cuda_benchmark [target.pcd source.pcd] [resolution]" << std::endl;
    return 1;
  }

  // downsampling (same as apps/align.cpp)
  pcl::VoxelGrid<PointT> voxelgrid;
  voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);
  Cloud::Ptr tmp(new Cloud);
  voxelgrid.setInputCloud(target);
  voxelgrid.filter(*tmp);
  *target = *tmp;
  voxelgrid.setInputCloud(source);
  voxelgrid.filter(*tmp);
  *source = *tmp;

  const int devices = ndt_cuda::GpuBackend::deviceCount();
  if (devices <= 0) {
    std::cerr << "no CUDA device found" << std::endl;
    return 1;
  }
  std::cout << "GPU          : " << ndt_cuda::GpuBackend::deviceName(0) << std::endl;
  std::cout << "CPU threads  : " << omp_get_max_threads() << std::endl;
  std::cout << "target/source: " << target->size() << " / " << source->size() << " points, resolution " << resolution << std::endl << std::endl;

  const std::vector<std::pair<std::string, pclomp::NeighborSearchMethod>> methods = {
    {"KDTREE", pclomp::KDTREE},
    {"DIRECT26", pclomp::DIRECT26},
    {"DIRECT7", pclomp::DIRECT7},
    {"DIRECT1", pclomp::DIRECT1},
  };

  pclomp::NormalDistributionsTransform<PointT, PointT> ndt_omp;
  ndt_cuda::NormalDistributionsTransformCUDA<PointT, PointT> ndt_gpu;
  ndt_omp.setResolution(resolution);
  ndt_gpu.setResolution(resolution);
  ndt_omp.setNumThreads(omp_get_max_threads());

  const int repeats = 10;
  bool all_ok = true;

  std::printf("%-9s %-5s %10s %10s %6s %12s %10s %10s\n", "method", "impl", "first[ms]", "avg[ms]", "iters", "trans_prob", "dt[m]", "dr[deg]");
  for (const auto& m : methods) {
    ndt_omp.setNeighborhoodSearchMethod(m.second);
    ndt_gpu.setNeighborhoodSearchMethod(m.second);

    const RunResult cpu = run(ndt_omp, target, source, repeats);
    const RunResult gpu = run(ndt_gpu, target, source, repeats);

    double dt, dr;
    transformDiff(cpu.transform, gpu.transform, dt, dr);
    std::printf("%-9s %-5s %10.2f %10.2f %6d %12.5f %10s %10s\n", m.first.c_str(), "omp", cpu.first_ms, cpu.avg_ms, cpu.iterations, cpu.trans_probability, "-", "-");
    std::printf("%-9s %-5s %10.2f %10.2f %6d %12.5f %10.5f %10.5f   speedup x%.1f\n", m.first.c_str(), "cuda", gpu.first_ms, gpu.avg_ms, gpu.iterations, gpu.trans_probability, dt, dr, cpu.avg_ms / gpu.avg_ms);

    // Both implementations should land on the same optimum (they differ only in float summation order and
    // in the corrected sign of one hessian term, which affects the path but not the optimum).
    const bool ok = dt < 0.02 && dr < 0.1;
    all_ok = all_ok && ok;

    // calculateScore() must agree closely, it does not depend on the optimizer at all.
    Cloud aligned;
    pcl::transformPointCloud(*source, aligned, cpu.transform);
    const double s_cpu = ndt_omp.calculateScore(aligned);
    const double s_gpu = ndt_gpu.calculateScore(aligned);
    std::printf("%-9s calculateScore omp=%.6f cuda=%.6f %s\n\n", m.first.c_str(), s_cpu, s_gpu, ok ? "[OK]" : "[TRANSFORM MISMATCH]");
  }

  std::cout << (all_ok ? "All methods agree." : "Some methods disagree, see above.") << std::endl;
  return all_ok ? 0 : 2;
}
