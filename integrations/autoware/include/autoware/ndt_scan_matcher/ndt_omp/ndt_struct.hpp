// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Melike
//
// Replaces Autoware's copy of tier4/ndt_omp: makes the types of pclomp:: those of ndt_cuda.

#ifndef AUTOWARE__NDT_SCAN_MATCHER__NDT_OMP__NDT_STRUCT_HPP_
#define AUTOWARE__NDT_SCAN_MATCHER__NDT_OMP__NDT_STRUCT_HPP_

#include <ndt_cuda/ndt_struct.hpp>

namespace pclomp
{
using ndt_cuda::DIRECT1;
using ndt_cuda::DIRECT26;
using ndt_cuda::DIRECT7;
using ndt_cuda::KDTREE;
using ndt_cuda::NdtParams;
using ndt_cuda::NdtResult;
using ndt_cuda::NeighborSearchMethod;
}  // namespace pclomp

#endif  // AUTOWARE__NDT_SCAN_MATCHER__NDT_OMP__NDT_STRUCT_HPP_
