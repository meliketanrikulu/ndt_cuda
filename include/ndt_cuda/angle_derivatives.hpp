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
 *  Derived from pcl::NormalDistributionsTransform::computeAngleDerivatives() and
 *  pclomp::NormalDistributionsTransform::computeAngleDerivatives() (koide3/ndt_omp, BSD 2-Clause,
 *  Copyright (c) 2019, k.koide).
 */

#pragma once

#include <cmath>

#include <Eigen/Core>

namespace ndt_cuda {

/** \brief Angular components of the point derivatives, Equation 6.19 and 6.21 [Magnusson 2009].
 *
 * Same tables as j_ang / h_ang in ndt_omp_impl.hpp, with one difference: row 6 of h_ang (d1, the second derivative
 * w.r.t. pitch) uses -sin(pitch) as in ndt_omp's double precision path (h_ang_d1_). ndt_omp's float path uses
 * +sin(pitch) there, which is a sign error that only affects the (pitch, pitch) entry of the hessian.
 */
inline void computeAngleDerivatives(const Eigen::Matrix<double, 6, 1>& p, bool compute_hessian, float j_ang[8][3], float h_ang[15][3]) {
  // Simplified math for near 0 angles
  double cx, cy, cz, sx, sy, sz;
  if (std::fabs(p(3)) < 10e-5) {
    cx = 1.0;
    sx = 0.0;
  } else {
    cx = std::cos(p(3));
    sx = std::sin(p(3));
  }
  if (std::fabs(p(4)) < 10e-5) {
    cy = 1.0;
    sy = 0.0;
  } else {
    cy = std::cos(p(4));
    sy = std::sin(p(4));
  }
  if (std::fabs(p(5)) < 10e-5) {
    cz = 1.0;
    sz = 0.0;
  } else {
    cz = std::cos(p(5));
    sz = std::sin(p(5));
  }

  // Precomputed angular gradient components. Letters correspond to Equation 6.19 [Magnusson 2009]
  const double j[8][3] = {
    {(-sx * sz + cx * sy * cz), (-sx * cz - cx * sy * sz), (-cx * cy)},  // a
    {(cx * sz + sx * sy * cz), (cx * cz - sx * sy * sz), (-sx * cy)},    // b
    {(-sy * cz), sy * sz, cy},                                           // c
    {sx * cy * cz, (-sx * cy * sz), sx * sy},                            // d
    {(-cx * cy * cz), cx * cy * sz, (-cx * sy)},                         // e
    {(-cy * sz), (-cy * cz), 0},                                         // f
    {(cx * cz - sx * sy * sz), (-cx * sz - sx * sy * cz), 0},            // g
    {(sx * cz + cx * sy * sz), (cx * sy * cz - sx * sz), 0}};            // h
  for (int r = 0; r < 8; r++)
    for (int c = 0; c < 3; c++) j_ang[r][c] = static_cast<float>(j[r][c]);

  if (compute_hessian) {
    // Precomputed angular hessian components. Letters correspond to Equation 6.21 and numbers correspond to row index [Magnusson 2009]
    // The sign of 'sx * sz' in c2 is incorrect in [Magnusson 2009], and it is fixed here (as in ndt_omp).
    const double h[15][3] = {
      {(-cx * sz - sx * sy * cz), (-cx * cz + sx * sy * sz), sx * cy},     // a2
      {(-sx * sz + cx * sy * cz), (-cx * sy * sz - sx * cz), (-cx * cy)},  // a3
      {(cx * cy * cz), (-cx * cy * sz), (cx * sy)},                        // b2
      {(sx * cy * cz), (-sx * cy * sz), (sx * sy)},                        // b3
      {(-sx * cz - cx * sy * sz), (sx * sz - cx * sy * cz), 0},            // c2
      {(cx * cz - sx * sy * sz), (-sx * sy * cz - cx * sz), 0},            // c3
      {(-cy * cz), (cy * sz), (-sy)},                                      // d1 (see the note above)
      {(-sx * sy * cz), (sx * sy * sz), (sx * cy)},                        // d2
      {(cx * sy * cz), (-cx * sy * sz), (-cx * cy)},                       // d3
      {(sy * sz), (sy * cz), 0},                                           // e1
      {(-sx * cy * sz), (-sx * cy * cz), 0},                               // e2
      {(cx * cy * sz), (cx * cy * cz), 0},                                 // e3
      {(-cy * cz), (cy * sz), 0},                                          // f1
      {(-cx * sz - sx * sy * cz), (-cx * cz + sx * sy * sz), 0},           // f2
      {(-sx * sz + cx * sy * cz), (-cx * sy * sz - sx * cz), 0}};          // f3
    for (int r = 0; r < 15; r++)
      for (int c = 0; c < 3; c++) h_ang[r][c] = static_cast<float>(h[r][c]);
  }
}

}  // namespace ndt_cuda
