// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Melike
//
// Compiles <ndt_cuda/pclomp_compat.hpp> in a translation unit that does not use ndt_omp: the pclomp names
// (NormalDistributionsTransformCUDA, NeighborSearchMethod, DIRECT7, ...) must then come from ndt_cuda.

#include <ndt_cuda/pclomp_compat.hpp>

int compatStandaloneDirect7() {
  using NDT = pclomp::NormalDistributionsTransformCUDA<pcl::PointXYZ, pcl::PointXYZ>;
  static_assert(std::is_same_v<NDT, ndt_cuda::NormalDistributionsTransformCUDA<pcl::PointXYZ, pcl::PointXYZ>>);
  pclomp::NeighborSearchMethod method = pclomp::DIRECT7;
  return static_cast<int>(method);
}
