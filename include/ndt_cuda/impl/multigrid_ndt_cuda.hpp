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
 *
 *  Derived from pclomp::MultiGridNormalDistributionsTransform (tier4/ndt_omp, BSD 2-Clause, Copyright (c) 2019, k.koide),
 *  which derives from pcl::NormalDistributionsTransform.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <ndt_cuda/impl/voxel_grid_covariance.hpp>
#include <ndt_cuda/multigrid_ndt_cuda.hpp>

template <typename PointSource, typename PointTarget>
ndt_cuda::MultiGridNormalDistributionsTransformCUDA<PointSource, PointTarget>::MultiGridNormalDistributionsTransformCUDA(int device) : Base(device) {
  this->reg_name_ = "MultiGridNormalDistributionsTransformCUDA";
  this->search_method = KDTREE;
  this->use_line_search_ = false;
  this->build_target_kdtree_ = false;
}

template <typename PointSource, typename PointTarget>
void ndt_cuda::MultiGridNormalDistributionsTransformCUDA<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  addTarget(cloud, "default");
  createVoxelKdtree();
}

template <typename PointSource, typename PointTarget>
void ndt_cuda::MultiGridNormalDistributionsTransformCUDA<PointSource, PointTarget>::addTarget(const PointCloudTargetConstPtr& cloud, const std::string& target_id) {
  pcl::Registration<PointSource, PointTarget>::setInputTarget(cloud);

  TargetGrid grid;
  grid.setLeafSize(resolution_);
  grid.setMinPointPerVoxel(target_cells_.getMinPointPerVoxel());
  grid.build(*cloud);

  VoxelGridHost exported;
  grid.exportTo(exported);

  auto tile = std::make_shared<Tile>();
  tile->leaf_size = resolution_;
  tile->cells.reserve(exported.size() * 3);
  for (const int32_t key : exported.keys) {
    // Inverse of key = sum_d (ijk_d - min_b_d) * divb_mul_d
    const int nx = exported.divb_mul[1];
    const int nxy = exported.divb_mul[2];
    tile->cells.push_back(key % nx + exported.min_b[0]);
    tile->cells.push_back((key % nxy) / nx + exported.min_b[1]);
    tile->cells.push_back(key / nxy + exported.min_b[2]);
  }
  tile->means = std::move(exported.means);
  tile->icovs = std::move(exported.icovs);
  tile->flags = std::move(exported.flags);
  tiles_[target_id] = std::move(tile);
}

template <typename PointSource, typename PointTarget>
void ndt_cuda::MultiGridNormalDistributionsTransformCUDA<PointSource, PointTarget>::createVoxelKdtree() {
  target_cloud_updated_ = false;

  voxel_means_.clear();
  for (const auto& [id, tile] : tiles_) {
    (void)id;
    for (std::size_t i = 0; i < tile->flags.size(); i++) {
      PointTarget point;
      point.x = static_cast<float>(tile->means[3 * i]);
      point.y = static_cast<float>(tile->means[3 * i + 1]);
      point.z = static_cast<float>(tile->means[3 * i + 2]);
      voxel_means_.push_back(point);
    }
  }

  uploadTarget();
  target_upload_pending_ = false;
}

template <typename PointSource, typename PointTarget>
void ndt_cuda::MultiGridNormalDistributionsTransformCUDA<PointSource, PointTarget>::mergeTiles(VoxelGridHost& grid) const {
  grid.clear();

  std::size_t num_voxels = 0;
  int min_b[3] = {std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), std::numeric_limits<int>::max()};
  int max_b[3] = {std::numeric_limits<int>::lowest(), std::numeric_limits<int>::lowest(), std::numeric_limits<int>::lowest()};
  const float leaf_size = tiles_.empty() ? resolution_ : tiles_.begin()->second->leaf_size;
  for (const auto& [id, tile] : tiles_) {
    (void)id;
    if (tile->leaf_size != leaf_size) {
      throw std::invalid_argument("[ndt_cuda] the loaded map tiles have different resolutions");
    }
    num_voxels += tile->flags.size();
    for (std::size_t i = 0; i < tile->cells.size(); i++) {
      min_b[i % 3] = std::min(min_b[i % 3], tile->cells[i]);
      max_b[i % 3] = std::max(max_b[i % 3], tile->cells[i]);
    }
  }
  if (num_voxels == 0) {
    return;
  }

  std::int64_t divisions = 1;
  std::int64_t divb_mul[3];
  for (int d = 0; d < 3; d++) {
    grid.leaf_size[d] = leaf_size;
    grid.min_b[d] = min_b[d];
    grid.max_b[d] = max_b[d];
    divb_mul[d] = divisions;
    divisions *= static_cast<std::int64_t>(max_b[d]) - min_b[d] + 1;
    if (divisions > std::numeric_limits<std::int32_t>::max()) {
      throw std::invalid_argument("[ndt_cuda] the loaded map tiles are too far apart for the voxel keys");
    }
    grid.divb_mul[d] = static_cast<int>(divb_mul[d]);
  }

  grid.keys.reserve(num_voxels);
  grid.means.reserve(num_voxels * 3);
  grid.icovs.reserve(num_voxels * 6);
  grid.flags.reserve(num_voxels);
  for (const auto& [id, tile] : tiles_) {
    (void)id;
    for (std::size_t i = 0; i < tile->flags.size(); i++) {
      grid.keys.push_back(static_cast<std::int32_t>(
        (tile->cells[3 * i] - min_b[0]) * divb_mul[0] + (tile->cells[3 * i + 1] - min_b[1]) * divb_mul[1] + (tile->cells[3 * i + 2] - min_b[2]) * divb_mul[2]));
    }
    grid.means.insert(grid.means.end(), tile->means.begin(), tile->means.end());
    grid.icovs.insert(grid.icovs.end(), tile->icovs.begin(), tile->icovs.end());
    grid.flags.insert(grid.flags.end(), tile->flags.begin(), tile->flags.end());
  }
}

template <typename PointSource, typename PointTarget>
void ndt_cuda::MultiGridNormalDistributionsTransformCUDA<PointSource, PointTarget>::uploadTarget() const {
  VoxelGridHost grid;
  mergeTiles(grid);
  backend_->setTarget(grid);
}

template <typename PointSource, typename PointTarget>
bool ndt_cuda::MultiGridNormalDistributionsTransformCUDA<PointSource, PointTarget>::finishIteration(double step_length) {
  nr_iterations_++;
  return nr_iterations_ >= max_iterations_ || std::fabs(step_length) < transformation_epsilon_;
}

template <typename PointSource, typename PointTarget>
std::vector<std::string> ndt_cuda::MultiGridNormalDistributionsTransformCUDA<PointSource, PointTarget>::getCurrentMapIDs() const {
  std::vector<std::string> ids;
  ids.reserve(tiles_.size());
  for (const auto& [id, tile] : tiles_) {
    (void)tile;
    ids.push_back(id);
  }
  return ids;
}
