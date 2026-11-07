// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Melike
//
// CPU emulation of ndt_cuda::GpuBackend, used only by the tests.
//
// It runs the exact per-point code of the CUDA kernels (src/ndt_cuda_kernels.cuh) on the CPU and reproduces the
// GPU data layout (origin-relative float voxels, hash table, float per-point accumulation, double reduction).
// Linking the tests against this file instead of gpu_backend.cu lets ndt_cuda::NormalDistributionsTransformCUDA
// run end-to-end on a machine without a GPU.

#include <ndt_cuda/gpu_backend.hpp>
#include <ndt_cuda/search_method.hpp>

#include <cstring>
#include <stdexcept>
#include <vector>

#include "ndt_cuda_kernels.cuh"

namespace ndt_cuda {

using namespace kernels;

struct GpuBackend::Impl {
  std::vector<int32_t> slot_keys;
  std::vector<uint32_t> slot_values;
  std::vector<DeviceVoxel> voxels;
  GridView grid{};
  double origin[3] = {0.0, 0.0, 0.0};
  bool has_target = false;
  std::vector<float4> source;

  template <int MODE, bool HESSIAN>
  void derivatives(const KernelParams& p, double out[kNumDerivativeValues]) const {
    for (int k = 0; k < kNumDerivativeValues; k++) out[k] = 0.0;
    for (const float4& pt : source) {
      float acc[kNumDerivativeValues] = {0.0f};
      accumulateDerivatives<MODE, HESSIAN>(grid, p, pt, acc);
      for (int k = 0; k < kNumDerivativeValues; k++) out[k] += acc[k];
    }
  }

  template <int MODE>
  double score(const KernelParams& p, const std::vector<float4>& points) const {
    double sum = 0.0;
    for (const float4& pt : points) {
      float acc[kNumScoreValues] = {0.0f};
      accumulateScore<MODE>(grid, p, pt, acc);
      sum += acc[0];
    }
    return sum;
  }
};

GpuBackend::GpuBackend(int) : impl_(new Impl) {}
GpuBackend::~GpuBackend() {}

int GpuBackend::device() const {
  return 0;
}
std::size_t GpuBackend::numTargetVoxels() const {
  return impl_->voxels.size();
}
std::size_t GpuBackend::numSourcePoints() const {
  return impl_->source.size();
}

void GpuBackend::setTarget(const VoxelGridHost& g) {
  Impl& m = *impl_;
  const std::size_t n = g.size();
  if (g.means.size() != 3 * n || g.icovs.size() != 6 * n || g.flags.size() != n) {
    throw std::invalid_argument("inconsistent VoxelGridHost array sizes");
  }
  chooseOrigin(g.min_b, g.max_b, g.leaf_size, m.grid.origin_cell, m.origin);
  for (int d = 0; d < 3; d++) {
    m.grid.min_b[d] = g.min_b[d];
    m.grid.max_b[d] = g.max_b[d];
    m.grid.divb_mul[d] = g.divb_mul[d];
    m.grid.leaf_size[d] = g.leaf_size[d];
  }
  m.voxels.resize(n);
  for (std::size_t i = 0; i < n; i++) {
    const double* mu = &g.means[3 * i];
    const double* ic = &g.icovs[6 * i];
    m.voxels[i].a = make_float4(float(mu[0] - m.origin[0]), float(mu[1] - m.origin[1]), float(mu[2] - m.origin[2]), float(ic[0]));
    m.voxels[i].b = make_float4(float(ic[1]), float(ic[2]), float(ic[3]), float(ic[4]));
    m.voxels[i].c = make_float4(float(ic[5]), 0.0f, 0.0f, 0.0f);
  }
  const std::size_t capacity = buildHashTable(g.keys, g.flags, m.slot_keys, m.slot_values);
  m.grid.slot_keys = m.slot_keys.data();
  m.grid.slot_values = m.slot_values.data();
  m.grid.voxels = m.voxels.data();
  m.grid.slot_mask = static_cast<uint32_t>(capacity - 1);
  m.has_target = true;
}

void GpuBackend::setSource(const float* xyzw, std::size_t num_points) {
  impl_->source.resize(num_points);
  if (num_points) std::memcpy(impl_->source.data(), xyzw, num_points * sizeof(float4));
}

DerivativeResult GpuBackend::computeDerivatives(const DerivativeParams& params) {
  Impl& m = *impl_;
  DerivativeResult result;
  std::memset(&result, 0, sizeof(result));
  if (!m.has_target || m.voxels.empty() || m.source.empty()) return result;

  KernelParams p;
  std::memset(&p, 0, sizeof(p));
  setTransform(p, params.transform, m.origin);
  std::memcpy(p.j_ang, params.j_ang, sizeof(p.j_ang));
  std::memcpy(p.h_ang, params.h_ang, sizeof(p.h_ang));
  p.gauss_d1 = static_cast<float>(params.gauss_d1);
  p.gauss_d2 = static_cast<float>(params.gauss_d2);
  p.radius2 = params.radius * params.radius;

  double r[kNumDerivativeValues];
  const bool h = params.compute_hessian;
  switch (params.search_method) {
    case KDTREE:
      h ? m.derivatives<kModeKdtree, true>(p, r) : m.derivatives<kModeKdtree, false>(p, r);
      break;
    case DIRECT26:
      h ? m.derivatives<kModeDirect26, true>(p, r) : m.derivatives<kModeDirect26, false>(p, r);
      break;
    case DIRECT1:
      h ? m.derivatives<kModeDirect1, true>(p, r) : m.derivatives<kModeDirect1, false>(p, r);
      break;
    default:
      h ? m.derivatives<kModeDirect7, true>(p, r) : m.derivatives<kModeDirect7, false>(p, r);
      break;
  }

  result.score = r[0];
  for (int i = 0; i < 6; i++) result.gradient[i] = r[1 + i];
  int k = 7;
  for (int i = 0; i < 6; i++) {
    for (int j = i; j < 6; j++, k++) {
      result.hessian[i * 6 + j] = r[k];
      result.hessian[j * 6 + i] = r[k];
    }
  }
  return result;
}

double GpuBackend::calculateScore(const float* xyzw, std::size_t num_points, double gauss_d1, double gauss_d2, double gauss_d3, float radius, int search_method) {
  Impl& m = *impl_;
  if (!m.has_target || m.voxels.empty() || num_points == 0) return 0.0;
  std::vector<float4> points(num_points);
  std::memcpy(points.data(), xyzw, num_points * sizeof(float4));

  const float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  KernelParams p;
  std::memset(&p, 0, sizeof(p));
  setTransform(p, identity, m.origin);
  p.gauss_d1 = static_cast<float>(gauss_d1);
  p.gauss_d2 = static_cast<float>(gauss_d2);
  p.gauss_d3 = static_cast<float>(gauss_d3);
  p.radius2 = radius * radius;

  double sum;
  switch (search_method) {
    case KDTREE:
      sum = m.score<kModeKdtree>(p, points);
      break;
    case DIRECT26:
      sum = m.score<kModeDirect26>(p, points);
      break;
    case DIRECT1:
      sum = m.score<kModeDirect1>(p, points);
      break;
    default:
      sum = m.score<kModeDirect7>(p, points);
      break;
  }
  return sum / static_cast<double>(num_points);
}

int GpuBackend::deviceCount() {
  return 1;
}
std::string GpuBackend::deviceName(int) {
  return "CPU emulation";
}

}  // namespace ndt_cuda
