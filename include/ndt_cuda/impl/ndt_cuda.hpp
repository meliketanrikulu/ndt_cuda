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
 *  Derived from pcl/registration/impl/ndt.hpp and pclomp/ndt_omp_impl.hpp (koide3/ndt_omp, BSD 2-Clause,
 *  Copyright (c) 2019, k.koide). The Newton / More-Thuente optimization is unchanged; computeDerivatives()
 *  and calculateScore() delegate the per-point work to the GPU.
 */

#pragma once

#include <cmath>
#include <stdexcept>

#include <pcl/common/transforms.h>

#include <ndt_cuda/angle_derivatives.hpp>
#include <ndt_cuda/impl/voxel_grid_covariance.hpp>
#include <ndt_cuda/ndt_cuda.hpp>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
ndt_cuda::NormalDistributionsTransformCUDA<PointSource, PointTarget>::NormalDistributionsTransformCUDA(int device) : search_method(DIRECT7) {
  reg_name_ = "NormalDistributionsTransformCUDA";
  transformation_epsilon_ = 0.1;
  max_iterations_ = 35;
  updateGaussParams();

  if (GpuBackend::deviceCount() <= 0) {
    throw std::runtime_error("[ndt_cuda::NormalDistributionsTransformCUDA] no CUDA capable device found");
  }
  backend_ = std::make_shared<GpuBackend>(device);
}

template <typename PointSource, typename PointTarget>
ndt_cuda::NormalDistributionsTransformCUDA<PointSource, PointTarget>::~NormalDistributionsTransformCUDA() = default;

template <typename PointSource, typename PointTarget>
std::size_t ndt_cuda::NormalDistributionsTransformCUDA<PointSource, PointTarget>::getNumTargetVoxels() const {
  return backend_->numTargetVoxels();
}

template <typename PointSource, typename PointTarget>
int ndt_cuda::NormalDistributionsTransformCUDA<PointSource, PointTarget>::getDevice() const {
  return backend_->device();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
void ndt_cuda::NormalDistributionsTransformCUDA<PointSource, PointTarget>::updateGaussParams() {
  // Initializes the gaussian fitting parameters (eq. 6.8) [Magnusson 2009]
  const double gauss_c1 = 10.0 * (1 - outlier_ratio_);
  const double gauss_c2 = outlier_ratio_ / std::pow(resolution_, 3);
  gauss_d3_ = -std::log(gauss_c2);
  gauss_d1_ = -std::log(gauss_c1 + gauss_c2) - gauss_d3_;
  gauss_d2_ = -2 * std::log((-std::log(gauss_c1 * std::exp(-0.5) + gauss_c2) - gauss_d3_) / gauss_d1_);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
void ndt_cuda::NormalDistributionsTransformCUDA<PointSource, PointTarget>::packPoints(const PointCloudSource& cloud, std::vector<float>& xyzw) {
  xyzw.resize(cloud.points.size() * 4);
  float* dst = xyzw.data();
  for (const auto& pt : cloud.points) {
    dst[0] = pt.x;
    dst[1] = pt.y;
    dst[2] = pt.z;
    dst[3] = 0.0f;
    dst += 4;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
Eigen::Matrix4f ndt_cuda::NormalDistributionsTransformCUDA<PointSource, PointTarget>::vectorToMatrix(const Eigen::Matrix<double, 6, 1>& x) {
  Eigen::Matrix4f trans;
  convertTransform(x, trans);
  return trans;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
void ndt_cuda::NormalDistributionsTransformCUDA<PointSource, PointTarget>::init() {
  target_cells_.setLeafSize(resolution_);
  target_cells_.build(*target_);

  VoxelGridHost grid;
  target_cells_.exportTo(grid);
  backend_->setTarget(grid);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
void ndt_cuda::NormalDistributionsTransformCUDA<PointSource, PointTarget>::computeTransformation(PointCloudSource& output, const Eigen::Matrix4f& guess) {
  nr_iterations_ = 0;
  converged_ = false;

  updateGaussParams();

  // The source points stay on the GPU for the whole optimization.
  packPoints(*input_, packed_);
  backend_->setSource(packed_.data(), input_->points.size());

  if (guess != Eigen::Matrix4f::Identity()) {
    // Initialise final transformation to the guessed one
    final_transformation_ = guess;
  }

  Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor> eig_transformation;
  eig_transformation.matrix() = final_transformation_;

  // Convert initial guess matrix to 6 element transformation vector
  Eigen::Matrix<double, 6, 1> p, delta_p, score_gradient;
  const Eigen::Vector3f init_translation = eig_transformation.translation();
  const Eigen::Vector3f init_rotation = eig_transformation.rotation().eulerAngles(0, 1, 2);
  p << init_translation(0), init_translation(1), init_translation(2), init_rotation(0), init_rotation(1), init_rotation(2);

  Eigen::Matrix<double, 6, 6> hessian;

  // Calculate derivatives of initial transform vector, subsequent derivative calculations are done in the step length determination.
  double score = computeDerivatives(score_gradient, hessian, p);

  while (!converged_) {
    // Store previous transformation
    previous_transformation_ = transformation_;

    // Solve for decent direction using newton method, line 23 in Algorithm 2 [Magnusson 2009]
    Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> sv(hessian, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // Negative for maximization as opposed to minimization
    delta_p = sv.solve(-score_gradient);

    // Calculate step length with guaranteed sufficient decrease [More, Thuente 1994]
    double delta_p_norm = delta_p.norm();

    if (delta_p_norm == 0 || delta_p_norm != delta_p_norm) {
      trans_probability_ = score / static_cast<double>(input_->points.size());
      converged_ = delta_p_norm == delta_p_norm;
      pcl::transformPointCloud(*input_, output, final_transformation_);
      return;
    }

    delta_p.normalize();
    delta_p_norm = computeStepLengthMT(p, delta_p, delta_p_norm, step_size_, transformation_epsilon_ / 2, score, score_gradient, hessian);
    delta_p *= delta_p_norm;

    transformation_ = vectorToMatrix(delta_p);

    p = p + delta_p;

    // Update Visualizer (untested)
    if (update_visualizer_) {
      pcl::transformPointCloud(*input_, output, final_transformation_);
      update_visualizer_(output, pcl::Indices(), *target_, pcl::Indices());
    }

    if (nr_iterations_ > max_iterations_ || (nr_iterations_ && (std::fabs(delta_p_norm) < transformation_epsilon_))) {
      converged_ = true;
    }

    nr_iterations_++;
  }

  // The aligned cloud is only needed once, at the end.
  pcl::transformPointCloud(*input_, output, final_transformation_);

  // Store transformation probability. The relative differences within each scan registration are accurate
  // but the normalization constants need to be modified for it to be globally accurate
  trans_probability_ = score / static_cast<double>(input_->points.size());
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
double ndt_cuda::NormalDistributionsTransformCUDA<PointSource, PointTarget>::computeDerivatives(
  Eigen::Matrix<double, 6, 1>& score_gradient,
  Eigen::Matrix<double, 6, 6>& hessian,
  const Eigen::Matrix<double, 6, 1>& p,
  bool compute_hessian) {
  // Precompute Angular Derivatives (eq. 6.19 and 6.21)[Magnusson 2009]
  computeAngleDerivatives(p, compute_hessian, j_ang_, h_ang_);

  DerivativeParams params;
  for (int r = 0; r < 4; r++)
    for (int c = 0; c < 4; c++) params.transform[r * 4 + c] = final_transformation_(r, c);
  for (int r = 0; r < 8; r++)
    for (int c = 0; c < 3; c++) params.j_ang[r][c] = j_ang_[r][c];
  for (int r = 0; r < 15; r++)
    for (int c = 0; c < 3; c++) params.h_ang[r][c] = h_ang_[r][c];
  params.gauss_d1 = gauss_d1_;
  params.gauss_d2 = gauss_d2_;
  params.radius = resolution_;
  params.search_method = static_cast<int>(search_method);
  params.compute_hessian = compute_hessian;

  const DerivativeResult result = backend_->computeDerivatives(params);

  for (int i = 0; i < 6; i++) score_gradient(i) = result.gradient[i];
  hessian.setZero();
  if (compute_hessian) {
    for (int i = 0; i < 6; i++)
      for (int j = 0; j < 6; j++) hessian(i, j) = result.hessian[i * 6 + j];
  }
  return result.score;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
bool ndt_cuda::NormalDistributionsTransformCUDA<PointSource, PointTarget>::updateIntervalMT(
  double& a_l,
  double& f_l,
  double& g_l,
  double& a_u,
  double& f_u,
  double& g_u,
  double a_t,
  double f_t,
  double g_t) {
  // Case U1 in Update Algorithm and Case a in Modified Update Algorithm [More, Thuente 1994]
  if (f_t > f_l) {
    a_u = a_t;
    f_u = f_t;
    g_u = g_t;
    return (false);
  }
  // Case U2 in Update Algorithm and Case b in Modified Update Algorithm [More, Thuente 1994]
  else if (g_t * (a_l - a_t) > 0) {
    a_l = a_t;
    f_l = f_t;
    g_l = g_t;
    return (false);
  }
  // Case U3 in Update Algorithm and Case c in Modified Update Algorithm [More, Thuente 1994]
  else if (g_t * (a_l - a_t) < 0) {
    a_u = a_l;
    f_u = f_l;
    g_u = g_l;

    a_l = a_t;
    f_l = f_t;
    g_l = g_t;
    return (false);
  }
  // Interval Converged
  else
    return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
double ndt_cuda::NormalDistributionsTransformCUDA<PointSource, PointTarget>::trialValueSelectionMT(
  double a_l,
  double f_l,
  double g_l,
  double a_u,
  double f_u,
  double g_u,
  double a_t,
  double f_t,
  double g_t) {
  // Case 1 in Trial Value Selection [More, Thuente 1994]
  if (f_t > f_l) {
    // Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
    // Equation 2.4.52 [Sun, Yuan 2006]
    double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
    double w = std::sqrt(z * z - g_t * g_l);
    // Equation 2.4.56 [Sun, Yuan 2006]
    double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

    // Calculate the minimizer of the quadratic that interpolates f_l, f_t and g_l
    // Equation 2.4.2 [Sun, Yuan 2006]
    double a_q = a_l - 0.5 * (a_l - a_t) * g_l / (g_l - (f_l - f_t) / (a_l - a_t));

    if (std::fabs(a_c - a_l) < std::fabs(a_q - a_l))
      return (a_c);
    else
      return (0.5 * (a_q + a_c));
  }
  // Case 2 in Trial Value Selection [More, Thuente 1994]
  else if (g_t * g_l < 0) {
    // Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
    // Equation 2.4.52 [Sun, Yuan 2006]
    double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
    double w = std::sqrt(z * z - g_t * g_l);
    // Equation 2.4.56 [Sun, Yuan 2006]
    double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

    // Calculate the minimizer of the quadratic that interpolates f_l, g_l and g_t
    // Equation 2.4.5 [Sun, Yuan 2006]
    double a_s = a_l - (a_l - a_t) / (g_l - g_t) * g_l;

    if (std::fabs(a_c - a_t) >= std::fabs(a_s - a_t))
      return (a_c);
    else
      return (a_s);
  }
  // Case 3 in Trial Value Selection [More, Thuente 1994]
  else if (std::fabs(g_t) <= std::fabs(g_l)) {
    // Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
    // Equation 2.4.52 [Sun, Yuan 2006]
    double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
    double w = std::sqrt(z * z - g_t * g_l);
    double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

    // Calculate the minimizer of the quadratic that interpolates g_l and g_t
    // Equation 2.4.5 [Sun, Yuan 2006]
    double a_s = a_l - (a_l - a_t) / (g_l - g_t) * g_l;

    double a_t_next;

    if (std::fabs(a_c - a_t) < std::fabs(a_s - a_t))
      a_t_next = a_c;
    else
      a_t_next = a_s;

    if (a_t > a_l)
      return (std::min(a_t + 0.66 * (a_u - a_t), a_t_next));
    else
      return (std::max(a_t + 0.66 * (a_u - a_t), a_t_next));
  }
  // Case 4 in Trial Value Selection [More, Thuente 1994]
  else {
    // Calculate the minimizer of the cubic that interpolates f_u, f_t, g_u and g_t
    // Equation 2.4.52 [Sun, Yuan 2006]
    double z = 3 * (f_t - f_u) / (a_t - a_u) - g_t - g_u;
    double w = std::sqrt(z * z - g_t * g_u);
    // Equation 2.4.56 [Sun, Yuan 2006]
    return (a_u + (a_t - a_u) * (w - g_u - z) / (g_t - g_u + 2 * w));
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
double ndt_cuda::NormalDistributionsTransformCUDA<PointSource, PointTarget>::computeStepLengthMT(
  const Eigen::Matrix<double, 6, 1>& x,
  Eigen::Matrix<double, 6, 1>& step_dir,
  double step_init,
  double step_max,
  double step_min,
  double& score,
  Eigen::Matrix<double, 6, 1>& score_gradient,
  Eigen::Matrix<double, 6, 6>& hessian) {
  // Set the value of phi(0), Equation 1.3 [More, Thuente 1994]
  double phi_0 = -score;
  // Set the value of phi'(0), Equation 1.3 [More, Thuente 1994]
  double d_phi_0 = -(score_gradient.dot(step_dir));

  Eigen::Matrix<double, 6, 1> x_t;

  if (d_phi_0 >= 0) {
    // Not a decent direction
    if (d_phi_0 == 0)
      return 0;
    else {
      // Reverse step direction and calculate optimal step.
      d_phi_0 *= -1;
      step_dir *= -1;
    }
  }

  // The Search Algorithm for T(mu) [More, Thuente 1994]

  int max_step_iterations = 10;
  int step_iterations = 0;

  // Sufficient decrease constant, Equation 1.1 [More, Thuete 1994]
  double mu = 1.e-4;
  // Curvature condition constant, Equation 1.2 [More, Thuete 1994]
  double nu = 0.9;

  // Initial endpoints of Interval I,
  double a_l = 0, a_u = 0;

  // Auxiliary function psi is used until I is determined ot be a closed interval, Equation 2.1 [More, Thuente 1994]
  double f_l = auxiliaryFunction_PsiMT(a_l, phi_0, phi_0, d_phi_0, mu);
  double g_l = auxiliaryFunction_dPsiMT(d_phi_0, d_phi_0, mu);

  double f_u = auxiliaryFunction_PsiMT(a_u, phi_0, phi_0, d_phi_0, mu);
  double g_u = auxiliaryFunction_dPsiMT(d_phi_0, d_phi_0, mu);

  // Check used to allow More-Thuente step length calculation to be skipped by making step_min == step_max
  bool interval_converged = (step_max - step_min) < 0, open_interval = true;

  double a_t = step_init;
  a_t = std::min(a_t, step_max);
  a_t = std::max(a_t, step_min);

  x_t = x + step_dir * a_t;

  final_transformation_ = vectorToMatrix(x_t);

  // Updates score, gradient and hessian.  Hessian calculation is unnecessary but testing showed that most step calculations use the
  // initial step suggestion and recalculation the reusable portions of the hessian would intail more computation time.
  score = computeDerivatives(score_gradient, hessian, x_t, true);

  // Calculate phi(alpha_t)
  double phi_t = -score;
  // Calculate phi'(alpha_t)
  double d_phi_t = -(score_gradient.dot(step_dir));

  // Calculate psi(alpha_t)
  double psi_t = auxiliaryFunction_PsiMT(a_t, phi_t, phi_0, d_phi_0, mu);
  // Calculate psi'(alpha_t)
  double d_psi_t = auxiliaryFunction_dPsiMT(d_phi_t, d_phi_0, mu);

  // Iterate until max number of iterations, interval convergence or a value satisfies the sufficient decrease, Equation 1.1, and curvature condition, Equation 1.2 [More, Thuente 1994]
  while (!interval_converged && step_iterations < max_step_iterations && !(psi_t <= 0 /*Sufficient Decrease*/ && d_phi_t <= -nu * d_phi_0 /*Curvature Condition*/)) {
    // Use auxiliary function if interval I is not closed
    if (open_interval) {
      a_t = trialValueSelectionMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, psi_t, d_psi_t);
    } else {
      a_t = trialValueSelectionMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, phi_t, d_phi_t);
    }

    a_t = std::min(a_t, step_max);
    a_t = std::max(a_t, step_min);

    x_t = x + step_dir * a_t;

    final_transformation_ = vectorToMatrix(x_t);

    // Updates score, gradient. Values stored to prevent wasted computation.
    score = computeDerivatives(score_gradient, hessian, x_t, false);

    // Calculate phi(alpha_t+)
    phi_t = -score;
    // Calculate phi'(alpha_t+)
    d_phi_t = -(score_gradient.dot(step_dir));

    // Calculate psi(alpha_t+)
    psi_t = auxiliaryFunction_PsiMT(a_t, phi_t, phi_0, d_phi_0, mu);
    // Calculate psi'(alpha_t+)
    d_psi_t = auxiliaryFunction_dPsiMT(d_phi_t, d_phi_0, mu);

    // Check if I is now a closed interval
    if (open_interval && (psi_t <= 0 && d_psi_t >= 0)) {
      open_interval = false;

      // Converts f_l and g_l from psi to phi
      f_l = f_l + phi_0 - mu * d_phi_0 * a_l;
      g_l = g_l + mu * d_phi_0;

      // Converts f_u and g_u from psi to phi
      f_u = f_u + phi_0 - mu * d_phi_0 * a_u;
      g_u = g_u + mu * d_phi_0;
    }

    if (open_interval) {
      // Update interval end points using Updating Algorithm [More, Thuente 1994]
      interval_converged = updateIntervalMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, psi_t, d_psi_t);
    } else {
      // Update interval end points using Modified Updating Algorithm [More, Thuente 1994]
      interval_converged = updateIntervalMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, phi_t, d_phi_t);
    }

    step_iterations++;
  }

  // If inner loop was run then hessian needs to be calculated.
  // Hessian is unnecessary for step length determination but gradients are required
  // so derivative and transform data is stored for the next iteration.
  if (step_iterations) {
    Eigen::Matrix<double, 6, 1> unused_gradient;
    computeDerivatives(unused_gradient, hessian, x_t, true);
  }

  return (a_t);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
double ndt_cuda::NormalDistributionsTransformCUDA<PointSource, PointTarget>::calculateScore(const PointCloudSource& trans_cloud) const {
  if (trans_cloud.points.empty()) {
    return 0.0;
  }
  packPoints(trans_cloud, packed_);
  return backend_->calculateScore(packed_.data(), trans_cloud.points.size(), gauss_d1_, gauss_d2_, gauss_d3_, resolution_, static_cast<int>(search_method));
}
