// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Melike

#pragma once

// Drop-in compatibility for code written against koide3/ndt_omp or tier4/ndt_omp:
//
//   #include <ndt_cuda/pclomp_compat.hpp>
//   pclomp::NormalDistributionsTransformCUDA<pcl::PointXYZ, pcl::PointXYZ>::Ptr ndt(...);
//   ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
//
// pclomp::NormalDistributionsTransformCUDA and pclomp::MultiGridNormalDistributionsTransformCUDA are aliases of the
// ndt_cuda classes.
//
// If ndt_omp's <pclomp/ndt_omp.h> is NOT used in the same translation unit, this header also provides
// pclomp::NeighborSearchMethod, pclomp::KDTREE / DIRECT26 / DIRECT7 / DIRECT1, pclomp::NdtParams and
// pclomp::NdtResult. If you use both libraries,
// include <pclomp/ndt_omp.h> BEFORE this header; ndt_omp's own enum is then used, which
// setNeighborhoodSearchMethod() also accepts.

#include <ndt_cuda/multigrid_ndt_cuda.hpp>
#include <ndt_cuda/ndt_cuda.hpp>

namespace pclomp {

template <typename PointSource, typename PointTarget>
using NormalDistributionsTransformCUDA = ndt_cuda::NormalDistributionsTransformCUDA<PointSource, PointTarget>;

template <typename PointSource, typename PointTarget>
using MultiGridNormalDistributionsTransformCUDA = ndt_cuda::MultiGridNormalDistributionsTransformCUDA<PointSource, PointTarget>;

#if !defined(PCL_REGISTRATION_NDT_OMP_H_) && !defined(PCLOMP_NDT_STRUCT_HPP_)  // include guards of <pclomp/ndt_omp.h>, <pclomp/ndt_struct.hpp>
using ndt_cuda::DIRECT1;
using ndt_cuda::DIRECT26;
using ndt_cuda::DIRECT7;
using ndt_cuda::KDTREE;
using ndt_cuda::NeighborSearchMethod;
#endif

#ifndef PCLOMP_NDT_STRUCT_HPP_
using ndt_cuda::NdtParams;
using ndt_cuda::NdtResult;
#endif

}  // namespace pclomp
