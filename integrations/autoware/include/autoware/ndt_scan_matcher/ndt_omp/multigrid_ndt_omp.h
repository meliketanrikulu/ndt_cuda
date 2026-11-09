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

  // Newer Autoware asks whether map tiles are loaded.
  bool hasTarget() const { return !this->getCurrentMapIDs().empty(); }

  // Newer Autoware passes the scan to align() instead of calling setInputSource().
  using ndt_cuda::MultiGridNormalDistributionsTransformCUDA<PointSource, PointTarget>::align;
  void align(
    pcl::PointCloud<PointSource> & output, const Eigen::Matrix4f & guess,
    const typename pcl::PointCloud<PointSource>::ConstPtr & source)
  {
    this->setInputSource(source);
    this->align(output, guess);
  }
};
}  // namespace pclomp

#endif  // AUTOWARE__NDT_SCAN_MATCHER__NDT_OMP__MULTIGRID_NDT_OMP_H_
