// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Melike
//
// Validation of the real CUDA kernels and reductions, which the GPU-less ndt_cuda_test does not execute. It checks:
//   1. align() against ndt_omp, for every neighbor search method,
//   2. the same with more source points than the kernels have threads (several points per thread),
//   3. repeated runs are bit-identical,
//   4. target handling (empty target, no kd-tree build on the map).
//
// usage: ndt_cuda_gpu_test [data_dir]      exit code 77 (reported as skipped by ctest) if no GPU is available

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pclomp/ndt_omp.h>

#include <ndt_cuda/ndt_cuda.hpp>

using PointT = pcl::PointXYZ;
using Cloud = pcl::PointCloud<PointT>;
using NDT = ndt_cuda::NormalDistributionsTransformCUDA<PointT, PointT>;

static int g_failures = 0;
static const char* const kMethods[] = {"KDTREE", "DIRECT26", "DIRECT7", "DIRECT1"};

static void check(bool ok, const char* what, double value, double tol) {
  std::printf("  %-58s %12.3e  (tol %.0e)  %s\n", what, value, tol, ok ? "OK" : "FAIL");
  if (!ok) g_failures++;
}

static void transformDiff(const Eigen::Matrix4f& a, const Eigen::Matrix4f& b, double& dt, double& dr_deg) {
  const Eigen::Matrix4d d = a.cast<double>().inverse() * b.cast<double>();
  dt = d.block<3, 1>(0, 3).norm();
  const double c = std::max(-1.0, std::min(1.0, (d.block<3, 3>(0, 0).trace() - 1.0) / 2.0));
  dr_deg = std::acos(c) * 180.0 / M_PI;
}

static void compareWithOmp(const Cloud::Ptr& target, const Cloud::Ptr& source, const char* name) {
  for (int mode = 0; mode < 4; mode++) {
    pclomp::NormalDistributionsTransform<PointT, PointT> omp;
    NDT gpu;
    omp.setResolution(1.0);
    gpu.setResolution(1.0);
    omp.setNeighborhoodSearchMethod(static_cast<pclomp::NeighborSearchMethod>(mode));
    gpu.setNeighborhoodSearchMethod(static_cast<pclomp::NeighborSearchMethod>(mode));
    omp.setInputTarget(target);
    gpu.setInputTarget(target);
    omp.setInputSource(source);
    gpu.setInputSource(source);
    Cloud out;
    omp.align(out);
    gpu.align(out);

    double dt, dr;
    transformDiff(omp.getFinalTransformation(), gpu.getFinalTransformation(), dt, dr);
    char label[128];
    std::snprintf(label, sizeof(label), "%s %s translation diff [m]", name, kMethods[mode]);
    check(dt < 1e-2, label, dt, 1e-2);
    std::snprintf(label, sizeof(label), "%s %s rotation diff [deg]", name, kMethods[mode]);
    check(dr < 5e-2, label, dr, 5e-2);
    std::snprintf(label, sizeof(label), "%s %s converged", name, kMethods[mode]);
    check(gpu.hasConverged(), label, gpu.hasConverged() ? 0.0 : 1.0, 0.0);
  }
}

int main(int argc, char** argv) {
  const std::string data_dir = argc > 1 ? argv[1] : NDT_CUDA_DATA_DIR;
  if (ndt_cuda::GpuBackend::deviceCount() <= 0) {
    std::printf("no CUDA device found, skipping\n");
    return 77;
  }

  Cloud::Ptr target(new Cloud), source(new Cloud);
  if (pcl::io::loadPCDFile(data_dir + "/251370668.pcd", *target) || pcl::io::loadPCDFile(data_dir + "/251371071.pcd", *source)) {
    std::fprintf(stderr, "failed to load the test clouds from %s\n", data_dir.c_str());
    return 1;
  }
  pcl::VoxelGrid<PointT> voxelgrid;
  voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);
  Cloud::Ptr tmp(new Cloud);
  voxelgrid.setInputCloud(target);
  voxelgrid.filter(*tmp);
  *target = *tmp;
  voxelgrid.setInputCloud(source);
  voxelgrid.filter(*tmp);
  *source = *tmp;
  std::printf("GPU: %s\n", ndt_cuda::GpuBackend::deviceName(0).c_str());

  std::printf("\n[1] align() vs ndt_omp\n");
  compareWithOmp(target, source, "sample");

  // 1024 blocks of 256 threads: larger clouds put several points on one thread.
  std::printf("\n[2] align() vs ndt_omp, 300000 source points\n");
  Cloud::Ptr many(new Cloud);
  std::mt19937 rng(1);
  std::normal_distribution<float> noise(0.0f, 0.005f);
  many->reserve(300000);
  while (many->size() < 300000) {
    for (const auto& pt : *source) {
      if (many->size() == 300000) break;
      PointT q = pt;
      q.x += noise(rng);
      q.y += noise(rng);
      q.z += noise(rng);
      many->push_back(q);
    }
  }
  compareWithOmp(target, many, "many");

  std::printf("\n[3] repeated runs\n");
  for (int mode = 0; mode < 4; mode++) {
    NDT gpu;
    gpu.setNeighborhoodSearchMethod(static_cast<ndt_cuda::NeighborSearchMethod>(mode));
    gpu.setInputTarget(target);
    gpu.setInputSource(source);
    Cloud out;
    gpu.align(out);
    const Eigen::Matrix4f first = gpu.getFinalTransformation();
    const double probability = gpu.getTransformationProbability();
    int different = 0;
    for (int i = 0; i < 20; i++) {
      gpu.align(out);
      const Eigen::Matrix4f current = gpu.getFinalTransformation();
      different += std::memcmp(first.data(), current.data(), sizeof(float) * 16) != 0 || probability != gpu.getTransformationProbability();
    }
    char label[128];
    std::snprintf(label, sizeof(label), "%s runs that differ from the first", kMethods[mode]);
    check(different == 0, label, different, 0.0);
  }

  std::printf("\n[4] target handling\n");
  NDT empty;
  empty.setInputTarget(Cloud::Ptr(new Cloud));
  empty.setInputSource(source);
  Cloud out;
  empty.align(out);
  check(!empty.hasConverged(), "empty target as first target: align() does not converge", empty.hasConverged() ? 1.0 : 0.0, 0.0);

  // initCompute() builds its kd-tree only while this flag is set.
  struct Probe : NDT {
    bool kdtreePending() const { return this->target_cloud_updated_; }
  };
  Probe probe;
  probe.setInputTarget(target);
  check(!probe.kdtreePending(), "no kd-tree pending after setInputTarget()", probe.kdtreePending() ? 1.0 : 0.0, 0.0);

  std::printf("\n%s (%d failure%s)\n", g_failures ? "FAILED" : "PASSED", g_failures, g_failures == 1 ? "" : "s");
  return g_failures ? 1 : 0;
}
