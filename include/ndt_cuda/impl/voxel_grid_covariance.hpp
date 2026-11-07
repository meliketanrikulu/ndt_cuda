/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *  Modifications Copyright (c) 2026, Melike
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Derived from pcl::VoxelGridCovariance::applyFilter() and pclomp::VoxelGridCovariance::applyFilter()
 *  (koide3/ndt_omp, BSD 2-Clause, Copyright (c) 2019, k.koide).
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

#include <Eigen/Eigenvalues>
#include <pcl/console/print.h>

#include <ndt_cuda/voxel_grid_covariance.hpp>

template <typename PointT>
bool ndt_cuda::VoxelGridCovariance<PointT>::build(const pcl::PointCloud<PointT>& cloud) {
  leaves_.clear();
  min_b_.setZero();
  max_b_.setZero();
  divb_mul_.setZero();

  if (cloud.empty()) {
    return false;
  }

  // Same float arithmetic as pcl::VoxelGrid (inverse_leaf_size_ is a float array).
  const float inverse_leaf_size = 1.0f / leaf_size_;

  // Non-finite points are skipped even if the cloud is marked as dense: they have no meaningful voxel key.
  const auto is_finite = [](const PointT& pt) { return std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z); };

  Eigen::Array3f min_p = Eigen::Array3f::Constant(std::numeric_limits<float>::max());
  Eigen::Array3f max_p = Eigen::Array3f::Constant(std::numeric_limits<float>::lowest());
  bool has_finite_point = false;
  for (const auto& pt : cloud.points) {
    if (!is_finite(pt)) continue;
    const Eigen::Array3f p(pt.x, pt.y, pt.z);
    min_p = min_p.min(p);
    max_p = max_p.max(p);
    has_finite_point = true;
  }
  if (!has_finite_point) {
    return false;
  }

  // Check that the leaf size is not too small, given the size of the data
  const std::int64_t dx = static_cast<std::int64_t>((max_p[0] - min_p[0]) * inverse_leaf_size) + 1;
  const std::int64_t dy = static_cast<std::int64_t>((max_p[1] - min_p[1]) * inverse_leaf_size) + 1;
  const std::int64_t dz = static_cast<std::int64_t>((max_p[2] - min_p[2]) * inverse_leaf_size) + 1;
  if (dx * dy * dz > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
    PCL_WARN("[ndt_cuda::VoxelGridCovariance] Leaf size is too small for the input dataset. Integer indices would overflow.\n");
    return false;
  }

  // Bounding box in voxel coordinates
  for (int d = 0; d < 3; d++) {
    min_b_[d] = static_cast<int>(std::floor(min_p[d] * inverse_leaf_size));
    max_b_[d] = static_cast<int>(std::floor(max_p[d] * inverse_leaf_size));
  }
  const Eigen::Vector3i div_b = max_b_ - min_b_ + Eigen::Vector3i::Ones();
  divb_mul_ = Eigen::Vector3i(1, div_b[0], div_b[0] * div_b[1]);

  // First pass: accumulate point sums and second moments per voxel
  leaves_.reserve(cloud.size() / 4 + 1);
  for (const auto& pt : cloud.points) {
    if (!is_finite(pt)) continue;
    const int ijk0 = static_cast<int>(std::floor(pt.x * inverse_leaf_size) - static_cast<float>(min_b_[0]));
    const int ijk1 = static_cast<int>(std::floor(pt.y * inverse_leaf_size) - static_cast<float>(min_b_[1]));
    const int ijk2 = static_cast<int>(std::floor(pt.z * inverse_leaf_size) - static_cast<float>(min_b_[2]));
    const int idx = ijk0 * divb_mul_[0] + ijk1 * divb_mul_[1] + ijk2 * divb_mul_[2];

    // Note: as in PCL and ndt_omp, the second moment accumulator (cov) starts from the identity set by Leaf's
    // constructor rather than from zero. This adds I * (n - 1) / n^2 to the covariance (a small regularization);
    // it is kept so that the grid is identical to ndt_omp's.
    Leaf& leaf = leaves_[idx];
    const Eigen::Vector3d pt3d(pt.x, pt.y, pt.z);
    leaf.mean += pt3d;
    leaf.cov += pt3d * pt3d.transpose();
    ++leaf.nr_points;
  }

  // Second pass: mean, covariance, eigenvalue inflation and inverse covariance
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver;
  for (auto& [key, leaf] : leaves_) {
    (void)key;
    const Eigen::Vector3d pt_sum = leaf.mean;
    leaf.mean /= leaf.nr_points;

    // Voxels with too few points cannot be approximated by a normal distribution.
    if (leaf.nr_points < min_points_per_voxel_) {
      continue;
    }

    // Single pass covariance calculation
    if (use_autoware_covariance_) {
      leaf.cov = (leaf.cov - pt_sum * leaf.mean.transpose()) / (leaf.nr_points - 1);
    } else {
      leaf.cov = (leaf.cov - 2 * (pt_sum * leaf.mean.transpose())) / leaf.nr_points + leaf.mean * leaf.mean.transpose();
      leaf.cov *= (leaf.nr_points - 1.0) / leaf.nr_points;
    }

    eigensolver.compute(leaf.cov);
    Eigen::Matrix3d eigen_val = eigensolver.eigenvalues().asDiagonal();
    leaf.evecs = eigensolver.eigenvectors();

    if (eigen_val(0, 0) < 0 || eigen_val(1, 1) < 0 || eigen_val(2, 2) <= 0) {
      leaf.nr_points = -1;
      continue;
    }

    // Avoids matrices near singularities (eq 6.11)[Magnusson 2009]
    const double min_covar_eigvalue = min_covar_eigvalue_mult_ * eigen_val(2, 2);
    if (eigen_val(0, 0) < min_covar_eigvalue) {
      eigen_val(0, 0) = min_covar_eigvalue;
      if (eigen_val(1, 1) < min_covar_eigvalue) {
        eigen_val(1, 1) = min_covar_eigvalue;
      }
      leaf.cov = leaf.evecs * eigen_val * leaf.evecs.inverse();
    }
    leaf.evals = eigen_val.diagonal();

    leaf.icov = leaf.cov.inverse();
    if (leaf.icov.maxCoeff() == std::numeric_limits<float>::infinity() || leaf.icov.minCoeff() == -std::numeric_limits<float>::infinity()) {
      leaf.nr_points = -1;
    }
  }

  return true;
}

template <typename PointT>
std::size_t ndt_cuda::VoxelGridCovariance<PointT>::getNumValidVoxels() const {
  std::size_t n = 0;
  for (const auto& [key, leaf] : leaves_) {
    (void)key;
    n += leaf.nr_points >= min_points_per_voxel_ ? 1 : 0;
  }
  return n;
}

template <typename PointT>
void ndt_cuda::VoxelGridCovariance<PointT>::exportTo(VoxelGridHost& grid) const {
  grid.clear();
  for (int d = 0; d < 3; d++) {
    grid.leaf_size[d] = leaf_size_;
    grid.min_b[d] = min_b_[d];
    grid.max_b[d] = max_b_[d];
    grid.divb_mul[d] = divb_mul_[d];
  }

  grid.keys.reserve(leaves_.size());
  grid.means.reserve(leaves_.size() * 3);
  grid.icovs.reserve(leaves_.size() * 6);
  grid.flags.reserve(leaves_.size());

  for (const auto& [key, leaf] : leaves_) {
    // nr_points == -1: enough points but a degenerate covariance. Such voxels are skipped by the DIRECT* searches but
    // are still part of the centroid cloud searched by ndt_omp's KDTREE method.
    const bool direct = leaf.nr_points >= min_points_per_voxel_;
    const bool centroid = direct || leaf.nr_points == -1;
    if (!centroid) continue;

    grid.keys.push_back(static_cast<std::int32_t>(key));
    grid.means.insert(grid.means.end(), {leaf.mean(0), leaf.mean(1), leaf.mean(2)});
    grid.icovs.insert(grid.icovs.end(), {leaf.icov(0, 0), leaf.icov(0, 1), leaf.icov(0, 2), leaf.icov(1, 1), leaf.icov(1, 2), leaf.icov(2, 2)});
    grid.flags.push_back(static_cast<std::uint8_t>((direct ? VoxelGridHost::kFlagDirect : 0) | VoxelGridHost::kFlagCentroid));
  }
}
