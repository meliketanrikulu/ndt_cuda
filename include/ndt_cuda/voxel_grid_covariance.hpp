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
 *  Derived from pcl::VoxelGridCovariance and pclomp::VoxelGridCovariance (koide3/ndt_omp, BSD 2-Clause,
 *  Copyright (c) 2019, k.koide). Rewritten without the pcl::VoxelGrid base class; the voxel statistics
 *  are computed exactly as in ndt_omp so that both produce identical grids.
 */

#pragma once

#include <cstdint>
#include <unordered_map>

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <ndt_cuda/gpu_backend.hpp>

namespace ndt_cuda {

/** \brief Voxel grid holding the mean and inverse covariance of the points in each voxel (NDT target representation). */
template <typename PointT>
class VoxelGridCovariance {
public:
  struct Leaf {
    /** \brief Number of points, or -1 if the voxel had enough points but a degenerate covariance. */
    int nr_points = 0;
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    Eigen::Matrix3d cov = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d icov = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d evecs = Eigen::Matrix3d::Identity();
    Eigen::Vector3d evals = Eigen::Vector3d::Zero();
  };
  using LeafMap = std::unordered_map<int, Leaf>;

  /** \brief Voxel side length (the NDT resolution). */
  void setLeafSize(float leaf_size) { leaf_size_ = leaf_size; }
  float getLeafSize() const { return leaf_size_; }

  /** \brief Minimum number of points for a voxel to be used (default 6, at least 3). */
  void setMinPointPerVoxel(int min_points) { min_points_per_voxel_ = min_points > 2 ? min_points : 3; }
  int getMinPointPerVoxel() const { return min_points_per_voxel_; }

  /** \brief Eigenvalues smaller than this fraction of the largest one are inflated (default 0.01, Equation 6.11 [Magnusson 2009]). */
  void setMinCovarEigvalueMult(double mult) { min_covar_eigvalue_mult_ = mult; }
  double getMinCovarEigvalueMult() const { return min_covar_eigvalue_mult_; }

  /** \brief Use the covariance formula of Autoware's copy of the voxel grid (default false: the formula of ndt_omp).
   * Autoware divides the scatter matrix by n - 1. ndt_omp multiplies it by (n - 1) / n^2, which gives smaller covariances
   * and so lower scores. Both add a small multiple of the identity to it.
   */
  void setUseAutowareCovariance(bool use) { use_autoware_covariance_ = use; }
  bool getUseAutowareCovariance() const { return use_autoware_covariance_; }

  /** \brief Builds the grid from a cloud.
   * \return false if the cloud is empty or the leaf size is too small for the extent of the cloud (the grid is then empty).
   */
  bool build(const pcl::PointCloud<PointT>& cloud);

  const LeafMap& getLeaves() const { return leaves_; }
  const Eigen::Vector3i& getMinBoxCoordinates() const { return min_b_; }
  const Eigen::Vector3i& getMaxBoxCoordinates() const { return max_b_; }
  const Eigen::Vector3i& getDivisionMultiplier() const { return divb_mul_; }

  /** \brief Number of voxels usable by the DIRECT* searches. */
  std::size_t getNumValidVoxels() const;

  /** \brief Flattens the grid into the format uploaded to the GPU. */
  void exportTo(VoxelGridHost& grid) const;

private:
  float leaf_size_ = 1.0f;
  int min_points_per_voxel_ = 6;
  double min_covar_eigvalue_mult_ = 0.01;
  bool use_autoware_covariance_ = false;

  Eigen::Vector3i min_b_ = Eigen::Vector3i::Zero();
  Eigen::Vector3i max_b_ = Eigen::Vector3i::Zero();
  Eigen::Vector3i divb_mul_ = Eigen::Vector3i::Zero();
  LeafMap leaves_;
};

}  // namespace ndt_cuda

extern template class ndt_cuda::VoxelGridCovariance<pcl::PointXYZ>;
extern template class ndt_cuda::VoxelGridCovariance<pcl::PointXYZI>;
extern template class ndt_cuda::VoxelGridCovariance<pcl::PointXYZRGB>;
