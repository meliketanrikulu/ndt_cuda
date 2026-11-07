// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Melike

#pragma once

// Drop-in compatibility for code written against koide3/ndt_omp:
//
//   #include <ndt_cuda/pclomp_compat.hpp>
//   pclomp::NormalDistributionsTransformCUDA<pcl::PointXYZ, pcl::PointXYZ>::Ptr ndt(...);
//   ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
//
// pclomp::NormalDistributionsTransformCUDA is an alias of ndt_cuda::NormalDistributionsTransformCUDA.
//
// If ndt_omp's <pclomp/ndt_omp.h> is NOT used in the same translation unit, this header also provides
// pclomp::NeighborSearchMethod and pclomp::KDTREE / DIRECT26 / DIRECT7 / DIRECT1. If you use both libraries,
// include <pclomp/ndt_omp.h> BEFORE this header; ndt_omp's own enum is then used, which
// setNeighborhoodSearchMethod() also accepts.

#include <ndt_cuda/ndt_cuda.hpp>

namespace pclomp {

template <typename PointSource, typename PointTarget>
using NormalDistributionsTransformCUDA = ndt_cuda::NormalDistributionsTransformCUDA<PointSource, PointTarget>;

#ifndef PCL_REGISTRATION_NDT_OMP_H_  // include guard of <pclomp/ndt_omp.h>
using ndt_cuda::DIRECT1;
using ndt_cuda::DIRECT26;
using ndt_cuda::DIRECT7;
using ndt_cuda::KDTREE;
using ndt_cuda::NeighborSearchMethod;
#endif

}  // namespace pclomp
