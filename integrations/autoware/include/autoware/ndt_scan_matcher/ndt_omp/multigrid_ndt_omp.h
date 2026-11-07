// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Melike
//
// Replaces Autoware's copy of tier4/ndt_omp: ndt_scan_matcher then runs ndt_cuda without further changes.

#ifndef AUTOWARE__NDT_SCAN_MATCHER__NDT_OMP__MULTIGRID_NDT_OMP_H_
#define AUTOWARE__NDT_SCAN_MATCHER__NDT_OMP__MULTIGRID_NDT_OMP_H_

#include "ndt_struct.hpp"

#include <ndt_cuda/multigrid_ndt_cuda.hpp>

namespace pclomp
{
template <typename PointSource, typename PointTarget>
class MultiGridNormalDistributionsTransform
: public ndt_cuda::MultiGridNormalDistributionsTransformCUDA<PointSource, PointTarget>
{
public:
  MultiGridNormalDistributionsTransform() { this->setUseAutowareCovariance(true); }
};
}  // namespace pclomp

#endif  // AUTOWARE__NDT_SCAN_MATCHER__NDT_OMP__MULTIGRID_NDT_OMP_H_
