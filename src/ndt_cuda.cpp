// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Melike

#include <ndt_cuda/impl/multigrid_ndt_cuda.hpp>
#include <ndt_cuda/impl/ndt_cuda.hpp>

template class ndt_cuda::NormalDistributionsTransformCUDA<pcl::PointXYZ, pcl::PointXYZ>;
template class ndt_cuda::NormalDistributionsTransformCUDA<pcl::PointXYZI, pcl::PointXYZI>;
template class ndt_cuda::NormalDistributionsTransformCUDA<pcl::PointXYZRGB, pcl::PointXYZRGB>;

template class ndt_cuda::MultiGridNormalDistributionsTransformCUDA<pcl::PointXYZ, pcl::PointXYZ>;
template class ndt_cuda::MultiGridNormalDistributionsTransformCUDA<pcl::PointXYZI, pcl::PointXYZI>;
template class ndt_cuda::MultiGridNormalDistributionsTransformCUDA<pcl::PointXYZRGB, pcl::PointXYZRGB>;
