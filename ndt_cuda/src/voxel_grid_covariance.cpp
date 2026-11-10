// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Melike

#include <ndt_cuda/impl/voxel_grid_covariance.hpp>

template class ndt_cuda::VoxelGridCovariance<pcl::PointXYZ>;
template class ndt_cuda::VoxelGridCovariance<pcl::PointXYZI>;
template class ndt_cuda::VoxelGridCovariance<pcl::PointXYZRGB>;
