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
 *  Derived from pcl::NormalDistributionsTransform (pcl/registration/ndt.h, Brian Okorn, Todor Stoyanov) and
 *  pclomp::NormalDistributionsTransform (koide3/ndt_omp, BSD 2-Clause, Copyright (c) 2019, k.koide).
 *  The per-point computations run on the GPU; the optimizer is unchanged.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/registration.h>

#include <ndt_cuda/gpu_backend.hpp>
#include <ndt_cuda/search_method.hpp>
#include <ndt_cuda/voxel_grid_covariance.hpp>

namespace ndt_cuda {

/** \brief Normal Distributions Transform registration running on an NVIDIA GPU.
 *
 * Same interface and results as pclomp::NormalDistributionsTransform (koide3/ndt_omp): same parameters and defaults,
 * same neighbor search methods and the same Newton / More-Thuente optimization. Every score / gradient / hessian
 * evaluation (transforming the source points, finding their neighbor voxels and accumulating Equations 6.9-6.13
 * [Magnusson 2009]) runs on the GPU. The target voxel grid is built on the CPU when the target or the resolution
 * changes and uploaded once.
 *
 * \note One instance must not be used from several threads at the same time.
 */
template <typename PointSource, typename PointTarget>
class NormalDistributionsTransformCUDA : public pcl::Registration<PointSource, PointTarget> {
protected:
  using PointCloudSource = typename pcl::Registration<PointSource, PointTarget>::PointCloudSource;
  using PointCloudSourcePtr = typename PointCloudSource::Ptr;
  using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

  using PointCloudTarget = typename pcl::Registration<PointSource, PointTarget>::PointCloudTarget;
  using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
  using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

  using TargetGrid = VoxelGridCovariance<PointTarget>;

public:
  using Ptr = pcl::shared_ptr<NormalDistributionsTransformCUDA<PointSource, PointTarget>>;
  using ConstPtr = pcl::shared_ptr<const NormalDistributionsTransformCUDA<PointSource, PointTarget>>;

  /** \brief Constructor.
   * \param[in] device CUDA device ordinal; -1 uses the current device.
   * \throws std::runtime_error if no CUDA capable device is available.
   */
  explicit NormalDistributionsTransformCUDA(int device = -1);

  ~NormalDistributionsTransformCUDA() override;

  /** \brief Not copyable: an instance owns its GPU state. */
  NormalDistributionsTransformCUDA(const NormalDistributionsTransformCUDA&) = delete;
  NormalDistributionsTransformCUDA& operator=(const NormalDistributionsTransformCUDA&) = delete;

  /** \brief Kept for API compatibility with pclomp::NormalDistributionsTransform; has no effect. */
  void setNumThreads(int) {}

  /** \brief Provide a pointer to the input target (the map). Builds and uploads the voxel grid. */
  void setInputTarget(const PointCloudTargetConstPtr& cloud) override {
    pcl::Registration<PointSource, PointTarget>::setInputTarget(cloud);
    // pcl::Registration rejects an empty cloud and leaves target_ unchanged, so it may still be null.
    if (target_) init();
    // The neighbor search runs on the GPU: keeps initCompute() from building its unused kd-tree over the whole map.
    target_cloud_updated_ = false;
  }

  /** \brief Set/change the voxel grid resolution (side length of the voxels). */
  void setResolution(float resolution) {
    if (resolution_ != resolution) {
      resolution_ = resolution;
      if (target_) init();
    }
  }
  float getResolution() const { return resolution_; }

  /** \brief Set/change the newton line search maximum step length. */
  void setStepSize(double step_size) { step_size_ = step_size; }
  double getStepSize() const { return step_size_; }

  /** \brief Set/change the point cloud outlier ratio. */
  void setOutlierRatio(double outlier_ratio) { outlier_ratio_ = outlier_ratio; }
  double getOutlierRatio() const { return outlier_ratio_; }

  /** \brief Minimum number of points for a voxel to be used (default 6). */
  void setMinPointPerVoxel(int min_points) {
    target_cells_.setMinPointPerVoxel(min_points);
    if (target_) init();
  }
  int getMinPointPerVoxel() const { return target_cells_.getMinPointPerVoxel(); }

  void setNeighborhoodSearchMethod(NeighborSearchMethod method) { search_method = method; }

  /** \brief Accepts pclomp::NeighborSearchMethod (or any enum with the same values) for drop-in compatibility. */
  template <typename Enum, std::enable_if_t<std::is_enum_v<Enum> && !std::is_same_v<Enum, NeighborSearchMethod>, int> = 0>
  void setNeighborhoodSearchMethod(Enum method) {
    search_method = static_cast<NeighborSearchMethod>(static_cast<int>(method));
  }

  NeighborSearchMethod getNeighborhoodSearchMethod() const { return search_method; }

  /** \brief Registration alignment probability (score / number of source points). */
  double getTransformationProbability() const { return trans_probability_; }

  /** \brief Number of iterations of the last alignment. */
  int getFinalNumIteration() const { return nr_iterations_; }

  /** \brief Number of voxels uploaded to the GPU. */
  std::size_t getNumTargetVoxels() const;

  /** \brief CUDA device ordinal used by this instance. */
  int getDevice() const;

  /** \brief The voxel grid built from the target cloud. */
  const TargetGrid& getTargetCells() const { return target_cells_; }

  /** \brief Convert a [x, y, z, roll, pitch, yaw] vector to an affine transformation. */
  static void convertTransform(const Eigen::Matrix<double, 6, 1>& x, Eigen::Affine3f& trans) {
    trans = Eigen::Translation<float, 3>(float(x(0)), float(x(1)), float(x(2))) * Eigen::AngleAxis<float>(float(x(3)), Eigen::Vector3f::UnitX()) *
            Eigen::AngleAxis<float>(float(x(4)), Eigen::Vector3f::UnitY()) * Eigen::AngleAxis<float>(float(x(5)), Eigen::Vector3f::UnitZ());
  }

  /** \brief Convert a [x, y, z, roll, pitch, yaw] vector to a 4x4 transformation matrix. */
  static void convertTransform(const Eigen::Matrix<double, 6, 1>& x, Eigen::Matrix4f& trans) {
    Eigen::Affine3f affine;
    convertTransform(x, affine);
    trans = affine.matrix();
  }

  /** \brief Mean negative log likelihood of an already transformed cloud (lower is better), as in ndt_omp. */
  double calculateScore(const PointCloudSource& cloud) const;

  /** \brief Neighbor search method (public for compatibility with pclomp::NormalDistributionsTransform). */
  NeighborSearchMethod search_method;

protected:
  using pcl::Registration<PointSource, PointTarget>::reg_name_;
  using pcl::Registration<PointSource, PointTarget>::getClassName;
  using pcl::Registration<PointSource, PointTarget>::input_;
  using pcl::Registration<PointSource, PointTarget>::indices_;
  using pcl::Registration<PointSource, PointTarget>::target_;
  using pcl::Registration<PointSource, PointTarget>::target_cloud_updated_;
  using pcl::Registration<PointSource, PointTarget>::nr_iterations_;
  using pcl::Registration<PointSource, PointTarget>::max_iterations_;
  using pcl::Registration<PointSource, PointTarget>::previous_transformation_;
  using pcl::Registration<PointSource, PointTarget>::final_transformation_;
  using pcl::Registration<PointSource, PointTarget>::transformation_;
  using pcl::Registration<PointSource, PointTarget>::transformation_epsilon_;
  using pcl::Registration<PointSource, PointTarget>::converged_;
  using pcl::Registration<PointSource, PointTarget>::update_visualizer_;

  virtual void computeTransformation(PointCloudSource& output) { computeTransformation(output, Eigen::Matrix4f::Identity()); }
  void computeTransformation(PointCloudSource& output, const Eigen::Matrix4f& guess) override;

  /** \brief Builds the voxel grid on the CPU and uploads it to the GPU. */
  void init();

  /** \brief Score, gradient and hessian of the source cloud transformed by final_transformation_ (on the GPU). */
  double computeDerivatives(Eigen::Matrix<double, 6, 1>& score_gradient, Eigen::Matrix<double, 6, 6>& hessian, const Eigen::Matrix<double, 6, 1>& p, bool compute_hessian = true);

  /** \brief Line search with guaranteed sufficient decrease [More, Thuente 1994]. */
  double computeStepLengthMT(
    const Eigen::Matrix<double, 6, 1>& x,
    Eigen::Matrix<double, 6, 1>& step_dir,
    double step_init,
    double step_max,
    double step_min,
    double& score,
    Eigen::Matrix<double, 6, 1>& score_gradient,
    Eigen::Matrix<double, 6, 6>& hessian);

  bool updateIntervalMT(double& a_l, double& f_l, double& g_l, double& a_u, double& f_u, double& g_u, double a_t, double f_t, double g_t);
  double trialValueSelectionMT(double a_l, double f_l, double g_l, double a_u, double f_u, double g_u, double a_t, double f_t, double g_t);

  static double auxiliaryFunction_PsiMT(double a, double f_a, double f_0, double g_0, double mu = 1.e-4) { return f_a - f_0 - mu * g_0 * a; }
  static double auxiliaryFunction_dPsiMT(double g_a, double g_0, double mu = 1.e-4) { return g_a - mu * g_0; }

  /** \brief Gaussian fitting parameters, Equation 6.8 [Magnusson 2009]. */
  void updateGaussParams();

  /** \brief Packs a cloud as x, y, z, 0 floats for upload. */
  static void packPoints(const PointCloudSource& cloud, std::vector<float>& xyzw);

  /** \brief Transformation vector to matrix, as in ndt_omp. */
  static Eigen::Matrix4f vectorToMatrix(const Eigen::Matrix<double, 6, 1>& x);

  TargetGrid target_cells_;
  float resolution_ = 1.0f;
  double step_size_ = 0.1;
  double outlier_ratio_ = 0.55;
  double gauss_d1_ = 0.0, gauss_d2_ = 0.0, gauss_d3_ = 0.0;
  double trans_probability_ = 0.0;

  float j_ang_[8][3] = {};
  float h_ang_[15][3] = {};

  /** \brief GPU state. */
  std::unique_ptr<GpuBackend> backend_;
  mutable std::vector<float> packed_;

public:
  PCL_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace ndt_cuda

extern template class ndt_cuda::NormalDistributionsTransformCUDA<pcl::PointXYZ, pcl::PointXYZ>;
extern template class ndt_cuda::NormalDistributionsTransformCUDA<pcl::PointXYZI, pcl::PointXYZI>;
extern template class ndt_cuda::NormalDistributionsTransformCUDA<pcl::PointXYZRGB, pcl::PointXYZRGB>;
