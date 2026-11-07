// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Melike
//
// Same fields as pclomp::NdtParams and pclomp::NdtResult of tier4/ndt_omp (BSD 2-Clause, Copyright (c) 2019, k.koide).

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
