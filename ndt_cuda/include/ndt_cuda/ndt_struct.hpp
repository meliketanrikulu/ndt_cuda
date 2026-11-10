// Copyright 2024 TIER IV, Inc.
// Modifications Copyright (c) 2026, Melike: moved to the namespace ndt_cuda.
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
//
// pclomp::NdtParams and pclomp::NdtResult of tier4/ndt_omp (include/pclomp/ndt_struct.hpp), with the same fields.

#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ndt_cuda/search_method.hpp>

namespace ndt_cuda {

/** \brief Everything an alignment produced. */
struct NdtResult {
  Eigen::Matrix4f pose;
  float transform_probability;
  float nearest_voxel_transformation_likelihood;
  int iteration_num;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> transformation_array;
  Eigen::Matrix<double, 6, 6> hessian;
  std::vector<float> transform_probability_array;
  std::vector<float> nearest_voxel_transformation_likelihood_array;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/** \brief Parameters of an alignment, for setParams() and getParams(). */
struct NdtParams {
  double trans_epsilon{};
  double step_size{};
  double resolution{};
  int max_iterations{};
  NeighborSearchMethod search_method{};
  /** \brief Has no effect on the GPU. */
  int num_threads{};
  float regularization_scale_factor{};
  bool use_line_search = false;
};

}  // namespace ndt_cuda
