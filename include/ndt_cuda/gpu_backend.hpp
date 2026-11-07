// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Melike

#pragma once

// Plain C++ interface of the CUDA kernels used by ndt_cuda::NormalDistributionsTransformCUDA.
// It does not include CUDA, Eigen or PCL headers, so it can be used from any host translation unit.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ndt_cuda {

/** \brief Flattened voxel grid uploaded to the GPU (see VoxelGridCovariance::exportTo()). */
struct VoxelGridHost {
  /** \brief Voxel side lengths. */
  float leaf_size[3] = {1.0f, 1.0f, 1.0f};
  /** \brief Bounding box of the grid in voxel coordinates. */
  int min_b[3] = {0, 0, 0};
  int max_b[3] = {0, 0, 0};
  /** \brief Multipliers that linearize voxel coordinates: key = sum_d (ijk_d - min_b_d) * divb_mul_d. */
  int divb_mul[3] = {0, 0, 0};

  /** \brief Linear voxel index (key) of each voxel. */
  std::vector<int32_t> keys;
  /** \brief Voxel means, 3 doubles per voxel. */
  std::vector<double> means;
  /** \brief Inverse covariances, upper triangle (xx xy xz yy yz zz), 6 doubles per voxel. */
  std::vector<double> icovs;
  /** \brief Per-voxel flags, see kFlagDirect / kFlagCentroid. */
  std::vector<uint8_t> flags;

  /** \brief Voxel has enough points and a valid covariance (used by the DIRECT* searches). */
  static constexpr uint8_t kFlagDirect = 1;
  /** \brief Voxel had enough points, possibly with a degenerate covariance (used by the KDTREE search, as in ndt_omp). */
  static constexpr uint8_t kFlagCentroid = 2;

  void clear() {
    keys.clear();
    means.clear();
    icovs.clear();
    flags.clear();
  }
  std::size_t size() const { return keys.size(); }
};

/** \brief Inputs of one score / gradient / hessian evaluation. */
struct DerivativeParams {
  /** \brief Absolute 4x4 transformation (row-major) applied to the source cloud. */
  float transform[16];
  /** \brief Angular jacobian rows, Equation 6.19 [Magnusson 2009]. */
  float j_ang[8][3];
  /** \brief Angular hessian rows, Equation 6.21 [Magnusson 2009]. */
  float h_ang[15][3];
  /** \brief Gaussian fitting parameters, Equation 6.8 [Magnusson 2009]. */
  double gauss_d1;
  double gauss_d2;
  /** \brief Search radius for the KDTREE search (the NDT resolution). */
  float radius;
  /** \brief NeighborSearchMethod value. */
  int search_method;
  bool compute_hessian;
};

/** \brief Outputs of one evaluation. The hessian is row-major 6x6. */
struct DerivativeResult {
  double score;
  double gradient[6];
  double hessian[36];
};

/** \brief Owns the GPU state: voxel hash table, source points, scratch buffers and a CUDA stream.
 *
 * One instance must not be used from several threads at the same time; separate instances are independent.
 */
class GpuBackend {
public:
  /** \param device CUDA device ordinal, or -1 for the device that is current at construction. */
  explicit GpuBackend(int device = -1);
  ~GpuBackend();

  GpuBackend(const GpuBackend&) = delete;
  GpuBackend& operator=(const GpuBackend&) = delete;

  /** \brief Uploads the voxel grid and builds the GPU hash table. */
  void setTarget(const VoxelGridHost& grid);

  /** \brief Uploads the source points, packed as x, y, z, (unused) per point. */
  void setSource(const float* xyzw, std::size_t num_points);

  std::size_t numTargetVoxels() const;
  std::size_t numSourcePoints() const;

  /** \brief Score, gradient and (optionally) hessian of the source cloud transformed by params.transform. */
  DerivativeResult computeDerivatives(const DerivativeParams& params);

  /** \brief Mean negative log likelihood of an already transformed cloud (packed x, y, z, unused). */
  double calculateScore(const float* xyzw, std::size_t num_points, double gauss_d1, double gauss_d2, double gauss_d3, float radius, int search_method);

  /** \brief Device ordinal used by this backend. */
  int device() const;

  /** \brief Number of CUDA capable devices, 0 if no driver / device is available. */
  static int deviceCount();
  /** \brief Device name, e.g. "NVIDIA GeForce RTX 4090". */
  static std::string deviceName(int device);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ndt_cuda
