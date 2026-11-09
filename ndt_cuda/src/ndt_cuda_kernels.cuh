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
 *  Per-point NDT math, derived from pclomp::NormalDistributionsTransform::computePointDerivatives() and
 *  updateDerivatives() (koide3/ndt_omp, BSD 2-Clause, Copyright (c) 2019, k.koide), which derive from
 *  pcl::NormalDistributionsTransform. Rewritten for CUDA (__host__ __device__) with an origin-relative layout.
 */

// Shared by the CUDA kernels (gpu_backend.cu) and the host-side emulation used by the tests
// (test/gpu_backend_emulated.cpp). Everything here compiles both with nvcc and with a plain C++ compiler, so the
// exact kernel arithmetic can be validated on a machine without a GPU.
//
// The arithmetic mirrors the float code path of ndt_omp (Equations 6.9 - 6.13 and 6.17 - 6.21 [Magnusson 2009]).

#pragma once

#include <math.h>
#include <stdint.h>

#include <cstddef>

#ifdef __CUDACC__
#define NDT_HD __host__ __device__ __forceinline__
#else
#define NDT_HD inline
struct float4 {
  float x, y, z, w;
};
inline float4 make_float4(float x, float y, float z, float w) {
  float4 v;
  v.x = x, v.y = y, v.z = z, v.w = w;
  return v;
}
#endif

#if defined(__CUDA_ARCH__)
#define NDT_LDG(ptr) __ldg(ptr)
#define NDT_ISFINITE(x) isfinite(x)
#else
#include <cmath>
#define NDT_LDG(ptr) (*(ptr))
#define NDT_ISFINITE(x) std::isfinite(x)
#endif

namespace ndt_cuda {
namespace kernels {

// score, gradient (6), upper triangle of the hessian (21), nearest voxel score, points with neighbors, neighbors
constexpr int kNumDerivativeValues = 31;
constexpr int kNumScoreValues = 1;
// sum of the voxel scores, nearest voxel score, points with neighbors
constexpr int kNumProbabilityValues = 3;

constexpr int32_t kEmptySlot = -1;
constexpr uint32_t kProbeDone = 0xFFFFFFFFu;
constexpr uint32_t kIndexMask = 0x3FFFFFFFu;
constexpr int kFlagShift = 30;

constexpr uint32_t kFlagDirect = 1;    // == VoxelGridHost::kFlagDirect
constexpr uint32_t kFlagCentroid = 2;  // == VoxelGridHost::kFlagCentroid

constexpr int kModeKdtree = 0;    // == KDTREE
constexpr int kModeDirect26 = 1;  // == DIRECT26
constexpr int kModeDirect7 = 2;   // == DIRECT7
constexpr int kModeDirect1 = 3;   // == DIRECT1

// Neighbor offsets:
// [0, 27): all 27 cells, center first. Used by KDTREE (radius search == distance-filtered 3x3x3 block).
// [1, 27): the 26 surrounding cells, same set as pcl::getAllNeighborCellIndices() (DIRECT26).
// [0, 7) : center and the 6 face neighbors, same order as getNeighborhoodAtPoint7() (DIRECT7).
// [0, 1) : center only (DIRECT1).
#define NDT_CUDA_NEIGHBOR_OFFSETS                                                                                        \
  {                                                                                                                      \
    {0, 0, 0}, {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1},                                     \
    {-1, -1, -1}, {-1, -1, 0}, {-1, -1, 1}, {-1, 0, -1}, {-1, 0, 1}, {-1, 1, -1}, {-1, 1, 0}, {-1, 1, 1},               \
    {0, -1, -1}, {0, -1, 1}, {0, 1, -1}, {0, 1, 1},                                                                      \
    {1, -1, -1}, {1, -1, 0}, {1, -1, 1}, {1, 0, -1}, {1, 0, 1}, {1, 1, -1}, {1, 1, 0}, {1, 1, 1}                        \
  }

#ifdef __CUDACC__
__constant__ int c_offsets[27][3] = NDT_CUDA_NEIGHBOR_OFFSETS;
#endif
static const int h_offsets[27][3] = NDT_CUDA_NEIGHBOR_OFFSETS;

NDT_HD int neighborOffset(int n, int d) {
#if defined(__CUDA_ARCH__)
  return c_offsets[n][d];
#else
  return h_offsets[n][d];
#endif
}

template <int MODE>
struct NeighborRange;
template <>
struct NeighborRange<kModeKdtree> {
  static constexpr int begin = 0, end = 27;
  static constexpr uint32_t flag = kFlagCentroid;
};
template <>
struct NeighborRange<kModeDirect26> {
  static constexpr int begin = 1, end = 27;
  static constexpr uint32_t flag = kFlagDirect;
};
template <>
struct NeighborRange<kModeDirect7> {
  static constexpr int begin = 0, end = 7;
  static constexpr uint32_t flag = kFlagDirect;
};
template <>
struct NeighborRange<kModeDirect1> {
  static constexpr int begin = 0, end = 1;
  static constexpr uint32_t flag = kFlagDirect;
};

/** \brief Voxel stored relative to the grid origin: mean (xyz) + inverse covariance (6). */
struct alignas(16) DeviceVoxel {
  float4 a;  // mean.x, mean.y, mean.z, icov_xx
  float4 b;  // icov_xy, icov_xz, icov_yy, icov_yz
  float4 c;  // icov_zz, -, -, -
};

/** \brief Grid geometry and hash table description (passed by value to kernels). */
struct GridView {
  const int32_t* slot_keys;
  const uint32_t* slot_values;  // voxel index | flags << 30
  const DeviceVoxel* voxels;
  uint32_t slot_mask;
  bool unique_keys;  // false if map tiles put several voxels in one cell
  int min_b[3];
  int max_b[3];
  int divb_mul[3];
  int origin_cell[3];  // grid origin in voxel coordinates
  float leaf_size[3];
};

/** \brief Per-evaluation parameters (passed by value to kernels). */
struct KernelParams {
  float r[9];  // rotation, row-major
  float t[3];  // translation relative to the grid origin
  float j_ang[8][3];
  float h_ang[15][3];
  float gauss_d1;
  float gauss_d2;
  float gauss_d3;
  float radius2;
};

NDT_HD uint32_t hashKey(int32_t key) {
  // Knuth multiplicative hash
  return static_cast<uint32_t>(key) * 2654435761u;
}

NDT_HD uint32_t firstSlot(const GridView& grid, int32_t key) {
  return hashKey(key) & grid.slot_mask;
}

/** \brief Next voxel of a cell with the required flag, or -1. Start with slot = firstSlot(). */
NDT_HD int nextVoxel(const GridView& grid, int32_t key, uint32_t required_flag, uint32_t& slot) {
  while (slot != kProbeDone) {
    const int32_t k = NDT_LDG(grid.slot_keys + slot);
    if (k == kEmptySlot) {
      slot = kProbeDone;
      return -1;
    }
    const uint32_t current = slot;
    slot = (slot + 1) & grid.slot_mask;
    if (k == key) {
      if (grid.unique_keys) slot = kProbeDone;
      const uint32_t v = NDT_LDG(grid.slot_values + current);
      if ((v >> kFlagShift) & required_flag) return static_cast<int>(v & kIndexMask);
    }
  }
  return -1;
}

/** \brief Linear leaf index of cell (ijk + offset n), or false if it lies outside the grid
 *  (same test as VoxelGridCovariance::getNeighborhoodAtPoint()). */
NDT_HD bool cellKey(const GridView& grid, const int ijk[3], int n, int32_t& key) {
  int idx = 0;
#ifdef __CUDACC__
#pragma unroll
#endif
  for (int d = 0; d < 3; d++) {
    const int c = ijk[d] + neighborOffset(n, d);
    if (c < grid.min_b[d] || c > grid.max_b[d]) {
      return false;
    }
    idx += (c - grid.min_b[d]) * grid.divb_mul[d];
  }
  key = idx;
  return true;
}

NDT_HD void loadVoxel(const GridView& grid, int v, float mean[3], float c[6]) {
  const DeviceVoxel* voxel = grid.voxels + v;
  const float4 a = NDT_LDG(&voxel->a);
  const float4 b = NDT_LDG(&voxel->b);
  const float4 cc = NDT_LDG(&voxel->c);
  mean[0] = a.x;
  mean[1] = a.y;
  mean[2] = a.z;
  c[0] = a.w;   // xx
  c[1] = b.x;   // xy
  c[2] = b.y;   // xz
  c[3] = b.z;   // yy
  c[4] = b.w;   // yz
  c[5] = cc.x;  // zz
}

NDT_HD void symMul(const float c[6], const float v[3], float out[3]) {
  out[0] = c[0] * v[0] + c[1] * v[1] + c[2] * v[2];
  out[1] = c[1] * v[0] + c[3] * v[1] + c[4] * v[2];
  out[2] = c[2] * v[0] + c[4] * v[1] + c[5] * v[2];
}

NDT_HD float dot3(const float a[3], const float b[3]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/** \brief Transforms a source point (result relative to the grid origin) and finds its voxel coordinates.
 *  \return false for non-finite points, which never contribute. */
NDT_HD bool transformAndLocate(const GridView& grid, const KernelParams& p, const float4 src, float x_trans[3], int ijk[3]) {
  x_trans[0] = p.r[0] * src.x + p.r[1] * src.y + p.r[2] * src.z + p.t[0];
  x_trans[1] = p.r[3] * src.x + p.r[4] * src.y + p.r[5] * src.z + p.t[1];
  x_trans[2] = p.r[6] * src.x + p.r[7] * src.y + p.r[8] * src.z + p.t[2];
  if (!NDT_ISFINITE(x_trans[0] + x_trans[1] + x_trans[2])) {
    return false;
  }
  // floor(x_abs / leaf) == floor(x_rel / leaf) + origin_cell because the origin is snapped to a voxel corner.
  for (int d = 0; d < 3; d++) {
    ijk[d] = static_cast<int>(floorf(x_trans[d] / grid.leaf_size[d])) + grid.origin_cell[d];
  }
  return true;
}

/** \brief Adds the score, gradient and (if HESSIAN) hessian contributions of one source point to acc.
 *  acc layout: [0] score, [1..6] gradient, [7..27] hessian upper triangle (row-major, i <= j),
 *  [28] nearest voxel score, [29] 1 if the point has neighbors, [30] number of neighbors. */
template <int MODE, bool HESSIAN>
NDT_HD void accumulateDerivatives(const GridView& grid, const KernelParams& p, const float4 src, float acc[kNumDerivativeValues]) {
  float x_trans[3];
  int ijk[3];
  if (!transformAndLocate(grid, p, src, x_trans, ijk)) {
    return;
  }

  const float x[3] = {src.x, src.y, src.z};
  const float d1 = p.gauss_d1;
  const float d2 = p.gauss_d2;

  // Point gradient J_E, Equation 6.18 and 6.19 [Magnusson 2009].
  // Columns 0-2 are the identity; pg[i] holds column i + 3.
  float x_j_ang[8];
  for (int r = 0; r < 8; r++) {
    x_j_ang[r] = dot3(p.j_ang[r], x);
  }
  const float pg[3][3] = {{0.0f, x_j_ang[0], x_j_ang[1]}, {x_j_ang[2], x_j_ang[3], x_j_ang[4]}, {x_j_ang[5], x_j_ang[6], x_j_ang[7]}};

  // Point hessian H_E, Equation 6.20 and 6.21 [Magnusson 2009]. Only blocks (i, j) with i, j >= 3 are non-zero:
  // (3,3)=a (3,4)=b (3,5)=c (4,4)=d (4,5)=e (5,5)=f
  float ph[6][3] = {{0.0f}};
  if (HESSIAN) {
    float x_h_ang[15];
    for (int r = 0; r < 15; r++) {
      x_h_ang[r] = dot3(p.h_ang[r], x);
    }
    ph[0][0] = 0.0f, ph[0][1] = x_h_ang[0], ph[0][2] = x_h_ang[1];           // a
    ph[1][0] = 0.0f, ph[1][1] = x_h_ang[2], ph[1][2] = x_h_ang[3];           // b
    ph[2][0] = 0.0f, ph[2][1] = x_h_ang[4], ph[2][2] = x_h_ang[5];           // c
    ph[3][0] = x_h_ang[6], ph[3][1] = x_h_ang[7], ph[3][2] = x_h_ang[8];     // d
    ph[4][0] = x_h_ang[9], ph[4][1] = x_h_ang[10], ph[4][2] = x_h_ang[11];   // e
    ph[5][0] = x_h_ang[12], ph[5][1] = x_h_ang[13], ph[5][2] = x_h_ang[14];  // f
  }

  float nearest_score = 0.0f;
  int num_neighbors = 0;

  for (int n = NeighborRange<MODE>::begin; n < NeighborRange<MODE>::end; n++) {
    int32_t key;
    if (!cellKey(grid, ijk, n, key)) {
      continue;
    }
    uint32_t slot = firstSlot(grid, key);
    int v;
    while ((v = nextVoxel(grid, key, NeighborRange<MODE>::flag, slot)) >= 0) {
      float mean[3], c[6];
      loadVoxel(grid, v, mean, c);

      // Denorm point, x_k' in Equations 6.12 and 6.13 [Magnusson 2009]
      const float xt[3] = {x_trans[0] - mean[0], x_trans[1] - mean[1], x_trans[2] - mean[2]};

      if (MODE == kModeKdtree) {
        if (dot3(xt, xt) >= p.radius2) {
          continue;
        }
      }
      num_neighbors++;

      float cx[3];
      symMul(c, xt, cx);

      // e^(-d_2/2 * (x_k - mu_k)^T Sigma_k^-1 (x_k - mu_k)) Equation 6.9 [Magnusson 2009]
      float e_x_cov_x = expf(-d2 * dot3(xt, cx) * 0.5f);
      // Probability of transformed points existence, Equation 6.9 [Magnusson 2009]
      const float score_inc = -d1 * e_x_cov_x;

      e_x_cov_x = d2 * e_x_cov_x;
      // Error checking for invalid values (the negated form also rejects NaN).
      if (!(e_x_cov_x <= 1.0f && e_x_cov_x >= 0.0f)) {
        continue;
      }
      // Reusable portion of Equation 6.12 and 6.13 [Magnusson 2009]
      e_x_cov_x *= d1;

      // g_i = x'^T Sigma^-1 dT/dp_i. Sigma^-1 is symmetric, so x'^T C pg_i == (C x')^T pg_i.
      float g[6];
      g[0] = cx[0];
      g[1] = cx[1];
      g[2] = cx[2];
      for (int i = 0; i < 3; i++) {
        g[3 + i] = dot3(cx, pg[i]);
      }

      if (score_inc > nearest_score) {
        nearest_score = score_inc;
      }

      // Update score and gradient, Equation 6.10 and 6.12 [Magnusson 2009]
      acc[0] += score_inc;
      for (int i = 0; i < 6; i++) {
        acc[1 + i] += e_x_cov_x * g[i];
      }

      if (HESSIAN) {
        // C * pg_i for all 6 columns of J_E (the first three are the columns of C)
        float cpg[6][3];
        cpg[0][0] = c[0], cpg[0][1] = c[1], cpg[0][2] = c[2];
        cpg[1][0] = c[1], cpg[1][1] = c[3], cpg[1][2] = c[4];
        cpg[2][0] = c[2], cpg[2][1] = c[4], cpg[2][2] = c[5];
        for (int i = 0; i < 3; i++) {
          symMul(c, pg[i], cpg[3 + i]);
        }

        // Update hessian, Equation 6.13 [Magnusson 2009]
        int h = 7;
        for (int i = 0; i < 6; i++) {
          for (int j = i; j < 6; j++, h++) {
            // pg_j^T C pg_i
            const float jcj = (j < 3) ? cpg[i][j] : dot3(pg[j - 3], cpg[i]);
            // x'^T C H_E(i, j)
            float xch = 0.0f;
            if (i >= 3) {
              const int hb = (i == 3) ? (j - 3) : (i == 4 ? (j == 4 ? 3 : 4) : 5);
              xch = dot3(cx, ph[hb]);
            }
            acc[h] += e_x_cov_x * (-d2 * g[i] * g[j] + xch + jcj);
          }
        }
      }
    }
  }

  if (num_neighbors > 0) {
    acc[28] += nearest_score;
    acc[29] += 1.0f;
    acc[30] += static_cast<float>(num_neighbors);
  }
}

/** \brief calculateScore() contribution of one (already transformed) point:
 *  mean over its neighbor voxels of -d1 * exp(-d2/2 * x'^T C x') - d3. */
template <int MODE>
NDT_HD void accumulateScore(const GridView& grid, const KernelParams& p, const float4 src, float acc[kNumScoreValues]) {
  float x_trans[3];
  int ijk[3];
  if (!transformAndLocate(grid, p, src, x_trans, ijk)) {
    return;
  }

  float sum = 0.0f;
  int count = 0;
  for (int n = NeighborRange<MODE>::begin; n < NeighborRange<MODE>::end; n++) {
    int32_t key;
    if (!cellKey(grid, ijk, n, key)) {
      continue;
    }
    uint32_t slot = firstSlot(grid, key);
    int v;
    while ((v = nextVoxel(grid, key, NeighborRange<MODE>::flag, slot)) >= 0) {
      float mean[3], c[6];
      loadVoxel(grid, v, mean, c);
      const float xt[3] = {x_trans[0] - mean[0], x_trans[1] - mean[1], x_trans[2] - mean[2]};
      if (MODE == kModeKdtree) {
        if (dot3(xt, xt) >= p.radius2) {
          continue;
        }
      }
      float cx[3];
      symMul(c, xt, cx);
      const float e_x_cov_x = expf(-p.gauss_d2 * dot3(xt, cx) * 0.5f);
      sum += -p.gauss_d1 * e_x_cov_x - p.gauss_d3;
      count++;
    }
  }
  if (count > 0) {
    acc[0] += sum / static_cast<float>(count);
  }
}

/** \brief Sum of -d1 * exp(-d2/2 * x'^T C x') over the neighbor voxels of one point and the largest of these values.
 *  \return false if the point has no neighbor voxel. As in tier4/ndt_omp. */
template <int MODE>
NDT_HD bool pointProbability(const GridView& grid, const KernelParams& p, const float4 src, float& sum, float& nearest) {
  float x_trans[3];
  int ijk[3];
  sum = 0.0f;
  nearest = 0.0f;
  if (!transformAndLocate(grid, p, src, x_trans, ijk)) {
    return false;
  }

  bool found = false;
  for (int n = NeighborRange<MODE>::begin; n < NeighborRange<MODE>::end; n++) {
    int32_t key;
    if (!cellKey(grid, ijk, n, key)) {
      continue;
    }
    uint32_t slot = firstSlot(grid, key);
    int v;
    while ((v = nextVoxel(grid, key, NeighborRange<MODE>::flag, slot)) >= 0) {
      float mean[3], c[6];
      loadVoxel(grid, v, mean, c);
      const float xt[3] = {x_trans[0] - mean[0], x_trans[1] - mean[1], x_trans[2] - mean[2]};
      if (MODE == kModeKdtree) {
        if (dot3(xt, xt) >= p.radius2) {
          continue;
        }
      }
      float cx[3];
      symMul(c, xt, cx);
      const float score = -p.gauss_d1 * expf(-p.gauss_d2 * dot3(xt, cx) * 0.5f);
      sum += score;
      if (score > nearest) {
        nearest = score;
      }
      found = true;
    }
  }
  return found;
}

/** \brief acc layout: [0] sum of the voxel scores, [1] nearest voxel score, [2] 1 if the point has neighbors. */
template <int MODE>
NDT_HD void accumulateProbability(const GridView& grid, const KernelParams& p, const float4 src, float acc[kNumProbabilityValues]) {
  float sum, nearest;
  if (pointProbability<MODE>(grid, p, src, sum, nearest)) {
    acc[0] += sum;
    acc[1] += nearest;
    acc[2] += 1.0f;
  }
}

// ---------------------------------------------------------------------------------------------
// Host-side construction of the hash table and parameters (shared by the backend and the test)
// ---------------------------------------------------------------------------------------------

/** \brief Origin (in voxel coordinates) used to keep coordinates small: a voxel corner near the grid center. */
inline void chooseOrigin(const int min_b[3], const int max_b[3], const float leaf_size[3], int origin_cell[3], double origin[3]) {
  for (int d = 0; d < 3; d++) {
    origin_cell[d] = static_cast<int>((static_cast<long long>(min_b[d]) + max_b[d]) / 2);
    origin[d] = static_cast<double>(origin_cell[d]) * static_cast<double>(leaf_size[d]);
  }
}

/** \brief Fills the rotation / translation part of KernelParams from an absolute row-major 4x4 transform. */
inline void setTransform(KernelParams& p, const float transform[16], const double origin[3]) {
  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < 3; c++) {
      p.r[r * 3 + c] = transform[r * 4 + c];
    }
    // Translation relative to the grid origin, computed in double to keep precision with large map coordinates.
    p.t[r] = static_cast<float>(static_cast<double>(transform[r * 4 + 3]) - origin[r]);
  }
}

/** \brief Open addressing hash table (linear probing, load factor <= 0.5). Returns the capacity (power of two).
 *  A key may occur several times. Containers must provide resize() and operator[]. */
template <typename KeyVector, typename ValueVector, typename FlagVector, typename InKeyVector>
std::size_t buildHashTable(const InKeyVector& keys, const FlagVector& flags, KeyVector& slot_keys, ValueVector& slot_values, bool& unique_keys) {
  const std::size_t n = keys.size();
  std::size_t capacity = 1024;
  while (capacity < 2 * n) capacity <<= 1;
  slot_keys.assign(capacity, kEmptySlot);
  slot_values.assign(capacity, 0u);
  const uint32_t mask = static_cast<uint32_t>(capacity - 1);
  unique_keys = true;
  for (std::size_t i = 0; i < n; i++) {
    const int32_t key = keys[i];
    uint32_t slot = hashKey(key) & mask;
    while (slot_keys[slot] != kEmptySlot) {
      unique_keys = unique_keys && slot_keys[slot] != key;
      slot = (slot + 1) & mask;
    }
    slot_keys[slot] = key;
    slot_values[slot] = static_cast<uint32_t>(i) | (static_cast<uint32_t>(flags[i] & 0x3u) << kFlagShift);
  }
  return capacity;
}

}  // namespace kernels
}  // namespace ndt_cuda

