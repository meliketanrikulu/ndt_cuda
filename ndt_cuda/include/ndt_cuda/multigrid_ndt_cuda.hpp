// Copyright 2022 TIER IV, Inc.
// Modifications Copyright (c) 2026, Melike: port to the GPU backend of ndt_cuda.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
 *  Derived from pclomp::MultiGridNormalDistributionsTransform of tier4/ndt_omp (Apache License 2.0, see above), which
 *  derives from pcl::NormalDistributionsTransform (BSD License, this notice).
 */

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <ndt_cuda/ndt_cuda.hpp>

namespace ndt_cuda {

/** \brief NDT whose map is made of tiles that can be added and removed while running, on an NVIDIA GPU.
 *
 * Same interface and optimization as pclomp::MultiGridNormalDistributionsTransform (tier4/ndt_omp), which Autoware's
 * ndt_scan_matcher uses. Each tile gets its own voxels on the CPU. Changes take effect in createVoxelKdtree(), which
 * uploads the loaded tiles to the GPU again.
 *
 * \warning As in tier4/ndt_omp, no kd-tree is built on the map, so pcl::Registration::getFitnessScore() must not be used.
 * \note setResolution() and setParams() do not change the tiles already added, as in tier4/ndt_omp. createVoxelKdtree()
 *       throws if tiles with different resolutions are loaded.
 */
template <typename PointSource, typename PointTarget>
class MultiGridNormalDistributionsTransformCUDA : public NormalDistributionsTransformCUDA<PointSource, PointTarget> {
  using Base = NormalDistributionsTransformCUDA<PointSource, PointTarget>;

protected:
  using PointCloudSourceConstPtr = typename Base::PointCloudSourceConstPtr;
  using PointCloudTargetConstPtr = typename Base::PointCloudTargetConstPtr;
  using TargetGrid = typename Base::TargetGrid;

public:
  using Ptr = pcl::shared_ptr<MultiGridNormalDistributionsTransformCUDA<PointSource, PointTarget>>;
  using ConstPtr = pcl::shared_ptr<const MultiGridNormalDistributionsTransformCUDA<PointSource, PointTarget>>;

  /** \brief Constructor. Uses the defaults of tier4/ndt_omp, including the KDTREE search and no line search.
   * \param[in] device CUDA device ordinal; -1 uses the current device.
   * \throws std::runtime_error if no CUDA capable device is available.
   */
  explicit MultiGridNormalDistributionsTransformCUDA(int device = -1);

  /** \brief Adds or replaces the tile "default" and calls createVoxelKdtree(), as in tier4/ndt_omp. */
  void setInputTarget(const PointCloudTargetConstPtr& cloud) override;

  /** \brief Builds the voxels of a tile on the CPU. A tile with the same id is replaced. */
  void addTarget(const PointCloudTargetConstPtr& cloud, const std::string& target_id);

  void removeTarget(const std::string& target_id) { tiles_.erase(target_id); }

  /** \brief Makes the added and removed tiles take effect. The name is that of tier4/ndt_omp: no kd-tree is built.
   * \throws std::invalid_argument if the loaded tiles have different resolutions or are too far apart for the voxel keys.
   */
  void createVoxelKdtree();

  /** \brief The voxel means of the map, as of the last createVoxelKdtree(). */
  pcl::PointCloud<PointTarget> getVoxelPCD() const { return voxel_means_; }

  /** \brief The ids of the currently added tiles. */
  std::vector<std::string> getCurrentMapIDs() const;

protected:
  using Base::backend_;
  using Base::max_iterations_;
  using Base::nr_iterations_;
  using Base::resolution_;
  using Base::target_cells_;
  using Base::target_cloud_updated_;
  using Base::target_upload_pending_;
  using Base::transformation_epsilon_;

  /** \brief Voxels of a tile, with their cell coordinates. They do not change once built, so copies share them. */
  struct Tile {
    float leaf_size = 1.0f;
    std::vector<int> cells;  // 3 per voxel
    std::vector<double> means;
    std::vector<double> icovs;
    std::vector<uint8_t> flags;
  };

  /** \brief The map is built tile by tile: changing a parameter does not rebuild it. */
  void init() override {}
  void uploadTarget() const override;
  bool finishIteration(double step_length) override;

  /** \brief Merges the tiles for the GPU. */
  void mergeTiles(VoxelGridHost& grid) const;

  std::map<std::string, std::shared_ptr<const Tile>> tiles_;
  pcl::PointCloud<PointTarget> voxel_means_;

public:
  PCL_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace ndt_cuda

extern template class ndt_cuda::MultiGridNormalDistributionsTransformCUDA<pcl::PointXYZ, pcl::PointXYZ>;
extern template class ndt_cuda::MultiGridNormalDistributionsTransformCUDA<pcl::PointXYZI, pcl::PointXYZI>;
extern template class ndt_cuda::MultiGridNormalDistributionsTransformCUDA<pcl::PointXYZRGB, pcl::PointXYZRGB>;
