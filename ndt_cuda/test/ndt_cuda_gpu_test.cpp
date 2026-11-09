// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Melike
//
// Validation of the real CUDA kernels and reductions, which the GPU-less ndt_cuda_test does not execute. It checks:
//   1. align() against ndt_omp, for every neighbor search method,
//   2. the same with more source points than the kernels have threads (several points per thread),
//   3. repeated runs are bit-identical,
//   4. target handling (empty and invalid targets, getFitnessScore(), optional kd-tree, copies).
//
// usage: ndt_cuda_gpu_test [data_dir]      exit code 77 (reported as skipped by ctest) if no GPU is available

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pclomp/ndt_omp.h>

#include <ndt_cuda/ndt_cuda.hpp>
#include <ndt_cuda/voxel_grid_covariance.hpp>

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

  // pcl::Registration::getFitnessScore() needs the kd-tree built by the first align(). A guess 3 m off gives a clearly nonzero score.
  Eigen::Matrix4f off = Eigen::Matrix4f::Identity();
  off(0, 3) = 3.0f;
  for (const int iterations : {1, 35}) {
    pclomp::NormalDistributionsTransform<PointT, PointT> omp;
    NDT gpu;
    pcl::Registration<PointT, PointT>* base = &gpu;
    for (pcl::Registration<PointT, PointT>* reg : std::vector<pcl::Registration<PointT, PointT>*>{&omp, &gpu}) {
      reg->setMaximumIterations(iterations);
      reg->setInputTarget(target);
      reg->setInputSource(source);
      reg->align(out, off);
    }
    const double fitness_omp = omp.getFitnessScore();
    const double error = std::fabs(base->getFitnessScore() - fitness_omp) / fitness_omp;
    char label[128];
    std::snprintf(label, sizeof(label), "getFitnessScore() vs ndt_omp, %d iterations, rel. error", iterations);
    check(fitness_omp > 0.1 && error < 1e-2, label, error, 1e-2);
  }

  // The kd-tree can be turned off for faster map changes.
  struct Probe : NDT {
    bool kdtreePending() const { return this->target_cloud_updated_; }
  };
  Probe probe;
  probe.setInputTarget(target);
  check(probe.kdtreePending(), "kd-tree pending after setInputTarget() by default", probe.kdtreePending() ? 0.0 : 1.0, 0.0);
  probe.setBuildTargetKdTree(false);
  check(!probe.kdtreePending(), "no kd-tree pending once turned off", probe.kdtreePending() ? 1.0 : 0.0, 0.0);
  probe.setInputTarget(target);
  check(!probe.kdtreePending(), "no kd-tree pending after setInputTarget() while off", probe.kdtreePending() ? 1.0 : 0.0, 0.0);
  probe.setInputSource(source);
  probe.align(out);
  check(probe.hasConverged(), "align() without the kd-tree converged", probe.hasConverged() ? 0.0 : 1.0, 0.0);
  probe.setBuildTargetKdTree(true);
  check(probe.kdtreePending(), "kd-tree pending again once turned back on", probe.kdtreePending() ? 0.0 : 1.0, 0.0);

  // Non-finite points are ignored even if the cloud is wrongly marked as dense.
  Cloud::Ptr with_nan(new Cloud(*target));
  PointT nan_point;
  nan_point.x = nan_point.y = nan_point.z = std::numeric_limits<float>::quiet_NaN();
  with_nan->insert(with_nan->end(), 10, nan_point);
  with_nan->is_dense = true;
  NDT clean, nan_target;
  clean.setInputTarget(target);
  clean.setInputSource(source);
  clean.align(out);
  nan_target.setInputTarget(with_nan);
  nan_target.setInputSource(source);
  nan_target.align(out);
  const float nan_diff = (clean.getFinalTransformation() - nan_target.getFinalTransformation()).cwiseAbs().maxCoeff();
  check(nan_target.hasConverged() && nan_diff == 0.0f, "NaN points in a dense target: same result as without them", nan_diff, 0.0);

  // A copy has its own GPU state and gets the map on its own.
  NDT copied(clean);
  copied.setInputSource(source);
  copied.align(out);
  const float copy_diff = (clean.getFinalTransformation() - copied.getFinalTransformation()).cwiseAbs().maxCoeff();
  check(copied.hasConverged() && copy_diff == 0.0f, "copy: same result as the original", copy_diff, 0.0);
  NDT assigned;
  assigned = clean;
  assigned.setInputSource(source);
  assigned.align(out);
  const float assign_diff = (clean.getFinalTransformation() - assigned.getFinalTransformation()).cwiseAbs().maxCoeff();
  check(assigned.hasConverged() && assign_diff == 0.0f, "assignment: same result as the original", assign_diff, 0.0);

  // A target without usable voxels must not be reported as a successful alignment.
  Cloud::Ptr only_nan(new Cloud);
  only_nan->insert(only_nan->end(), 10, nan_point);
  only_nan->is_dense = true;
  NDT no_voxels;
  no_voxels.setInputTarget(only_nan);
  no_voxels.setInputSource(source);
  no_voxels.align(out);
  check(no_voxels.getNumTargetVoxels() == 0 && !no_voxels.hasConverged(), "only NaN points in the target: align() does not converge", no_voxels.hasConverged() ? 1.0 : 0.0, 0.0);

  // setResolution() after setInputTarget() rebuilds the grid.
  NDT resolution_after, resolution_before;
  resolution_after.setInputTarget(target);
  resolution_after.setResolution(2.0);
  resolution_before.setResolution(2.0);
  resolution_before.setInputTarget(target);
  check(resolution_after.getNumTargetVoxels() == resolution_before.getNumTargetVoxels(), "setResolution() after setInputTarget() rebuilds the grid", double(resolution_after.getNumTargetVoxels()) - double(resolution_before.getNumTargetVoxels()), 0.0);

  // A failed upload must not leave the old map in use.
  ndt_cuda::VoxelGridCovariance<PointT> cells;
  cells.setLeafSize(1.0f);
  cells.build(*target);
  ndt_cuda::VoxelGridHost grid;
  cells.exportTo(grid);
  ndt_cuda::GpuBackend backend;
  backend.setTarget(grid);
  const std::size_t voxels_before = backend.numTargetVoxels();
  grid.flags.pop_back();
  bool threw = false;
  try {
    backend.setTarget(grid);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  check(voxels_before > 0 && threw && backend.numTargetVoxels() == 0, "failed setTarget() leaves no target", double(backend.numTargetVoxels()), 0.0);

  std::printf("\n%s (%d failure%s)\n", g_failures ? "FAILED" : "PASSED", g_failures, g_failures == 1 ? "" : "s");
  return g_failures ? 1 : 0;
}
