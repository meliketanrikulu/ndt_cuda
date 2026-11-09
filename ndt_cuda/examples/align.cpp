// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Melike
//
// Minimal example: align two point clouds with ndt_cuda::NormalDistributionsTransformCUDA.
//
// usage: ndt_cuda_align target.pcd source.pcd [resolution] [DIRECT7|DIRECT1|DIRECT26|KDTREE]

#include <chrono>
#include <iostream>
#include <string>

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <ndt_cuda/ndt_cuda.hpp>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: ndt_cuda_align target.pcd source.pcd [resolution] [DIRECT7|DIRECT1|DIRECT26|KDTREE]" << std::endl;
    return 1;
  }
  using PointT = pcl::PointXYZ;
  using Cloud = pcl::PointCloud<PointT>;

  Cloud::Ptr target(new Cloud), source(new Cloud);
  if (pcl::io::loadPCDFile(argv[1], *target) || pcl::io::loadPCDFile(argv[2], *source)) {
    std::cerr << "failed to load the input clouds" << std::endl;
    return 1;
  }
  const float resolution = argc > 3 ? std::stof(argv[3]) : 1.0f;
  const std::string method = argc > 4 ? argv[4] : "DIRECT7";

  // Downsample the clouds (0.1 m), as is usual before NDT
  pcl::VoxelGrid<PointT> voxelgrid;
  voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);
  Cloud::Ptr tmp(new Cloud);
  voxelgrid.setInputCloud(target);
  voxelgrid.filter(*tmp);
  target.swap(tmp);
  tmp.reset(new Cloud);
  voxelgrid.setInputCloud(source);
  voxelgrid.filter(*tmp);
  source.swap(tmp);

  ndt_cuda::NormalDistributionsTransformCUDA<PointT, PointT> ndt;
  ndt.setResolution(resolution);
  ndt.setNeighborhoodSearchMethod(method == "DIRECT1"    ? ndt_cuda::DIRECT1
                                  : method == "DIRECT26" ? ndt_cuda::DIRECT26
                                  : method == "KDTREE"   ? ndt_cuda::KDTREE
                                                         : ndt_cuda::DIRECT7);

  std::cout << "GPU     : " << ndt_cuda::GpuBackend::deviceName(ndt.getDevice()) << std::endl;

  auto t0 = std::chrono::steady_clock::now();
  ndt.setInputTarget(target);
  auto t1 = std::chrono::steady_clock::now();
  ndt.setInputSource(source);

  Cloud aligned;
  ndt.align(aligned);  // first call includes CUDA context warm-up
  auto t2 = std::chrono::steady_clock::now();
  const int repeats = 10;
  for (int i = 0; i < repeats; i++) ndt.align(aligned);
  auto t3 = std::chrono::steady_clock::now();

  auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
  std::cout << "points  : target " << target->size() << ", source " << source->size() << ", voxels " << ndt.getNumTargetVoxels() << std::endl;
  std::cout << "target  : " << ms(t0, t1) << " ms (voxel grid + upload)" << std::endl;
  std::cout << "align   : first " << ms(t1, t2) << " ms, average " << ms(t2, t3) / repeats << " ms" << std::endl;
  std::cout << "iters   : " << ndt.getFinalNumIteration() << ", converged " << ndt.hasConverged() << ", trans_prob " << ndt.getTransformationProbability() << std::endl;
  std::cout << "result  :\n" << ndt.getFinalTransformation() << std::endl;
  return 0;
}
