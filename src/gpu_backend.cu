// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Melike
//
// CUDA kernels, deterministic reductions and GPU resource management of ndt_cuda.
// The per-point NDT math lives in ndt_cuda_kernels.cuh (shared with the host-side test emulation).
//
// Reductions are done in double precision with a fixed grid size and a fixed reduction tree,
// so repeated runs on the same device give bit-identical results.

#include <ndt_cuda/gpu_backend.hpp>
#include <ndt_cuda/search_method.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "ndt_cuda_kernels.cuh"

namespace ndt_cuda {

namespace {

using namespace kernels;

#define NDT_CUDA_CHECK(expr)                                                                       \
  do {                                                                                             \
    cudaError_t err__ = (expr);                                                                    \
    if (err__ != cudaSuccess) {                                                                    \
      cudaGetLastError(); /* a failed call must not be reported again by the next check */        \
      std::ostringstream oss__;                                                                    \
      oss__ << "[ndt_cuda] " << #expr << " failed at " << __FILE__ << ":" << __LINE__ << " : " \
            << cudaGetErrorString(err__);                                                          \
      throw std::runtime_error(oss__.str());                                                       \
    }                                                                                              \
  } while (0)

static_assert(kModeKdtree == KDTREE && kModeDirect26 == DIRECT26 && kModeDirect7 == DIRECT7 && kModeDirect1 == DIRECT1, "search mode mismatch");
static_assert(kFlagDirect == VoxelGridHost::kFlagDirect && kFlagCentroid == VoxelGridHost::kFlagCentroid, "voxel flag mismatch");
static_assert(sizeof(DeviceVoxel) == 48, "unexpected DeviceVoxel layout");

constexpr int kBlockSize = 256;
constexpr int kWarpsPerBlock = kBlockSize / 32;
constexpr int kMaxBlocks = 1024;

// ---------------------------------------------------------------------------------------------
// Reductions (deterministic: fixed grid size and fixed reduction tree)
// ---------------------------------------------------------------------------------------------

/** \brief Reduces NV per-thread values over a block and writes one partial per value. */
template <int NV>
__device__ __forceinline__ void blockReduceToPartials(const float (&acc)[NV], double* partials) {
  __shared__ double s_warp[kWarpsPerBlock][NV];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

#pragma unroll
  for (int k = 0; k < NV; k++) {
    double v = static_cast<double>(acc[k]);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
    }
    if (lane == 0) {
      s_warp[warp][k] = v;
    }
  }
  __syncthreads();

  if (threadIdx.x < NV) {
    double sum = 0.0;
#pragma unroll
    for (int w = 0; w < kWarpsPerBlock; w++) {
      sum += s_warp[w][threadIdx.x];
    }
    partials[threadIdx.x * gridDim.x + blockIdx.x] = sum;
  }
}

/** \brief One block per value: sums the per-block partials of that value. */
__global__ void finalReduceKernel(const double* __restrict__ partials, int num_blocks, double* __restrict__ out) {
  __shared__ double s_data[kBlockSize];
  const int k = blockIdx.x;

  double sum = 0.0;
  for (int b = threadIdx.x; b < num_blocks; b += kBlockSize) {
    sum += partials[k * num_blocks + b];
  }
  s_data[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_data[threadIdx.x] += s_data[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    out[k] = s_data[0];
  }
}

// ---------------------------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------------------------

template <int MODE, bool HESSIAN>
__global__ void __launch_bounds__(kBlockSize) derivativesKernel(const float4* __restrict__ points, int num_points, GridView grid, KernelParams p, double* __restrict__ partials) {
  float acc[kNumDerivativeValues];
#pragma unroll
  for (int k = 0; k < kNumDerivativeValues; k++) {
    acc[k] = 0.0f;
  }

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_points; idx += blockDim.x * gridDim.x) {
    accumulateDerivatives<MODE, HESSIAN>(grid, p, __ldg(points + idx), acc);
  }

  blockReduceToPartials<kNumDerivativeValues>(acc, partials);
}

template <int MODE>
__global__ void __launch_bounds__(kBlockSize) scoreKernel(const float4* __restrict__ points, int num_points, GridView grid, KernelParams p, double* __restrict__ partials) {
  float acc[kNumScoreValues] = {0.0f};

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_points; idx += blockDim.x * gridDim.x) {
    accumulateScore<MODE>(grid, p, __ldg(points + idx), acc);
  }

  blockReduceToPartials<kNumScoreValues>(acc, partials);
}

// ---------------------------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------------------------

/** \brief Grow-only device buffer. */
template <typename T>
class DeviceBuffer {
public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { release(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void reserve(std::size_t n) {
    if (n <= capacity_) return;
    release();
    NDT_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), n * sizeof(T)));
    capacity_ = n;
  }
  void release() {
    if (ptr_) cudaFree(ptr_);
    ptr_ = nullptr;
    capacity_ = 0;
  }
  T* get() const { return ptr_; }

private:
  T* ptr_ = nullptr;
  std::size_t capacity_ = 0;
};

/** \brief Grow-only pinned host buffer. */
template <typename T>
class PinnedBuffer {
public:
  PinnedBuffer() = default;
  ~PinnedBuffer() { release(); }
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;

  void reserve(std::size_t n) {
    if (n <= capacity_) return;
    release();
    NDT_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&ptr_), n * sizeof(T)));
    capacity_ = n;
  }
  void release() {
    if (ptr_) cudaFreeHost(ptr_);
    ptr_ = nullptr;
    capacity_ = 0;
  }
  T* get() const { return ptr_; }

private:
  T* ptr_ = nullptr;
  std::size_t capacity_ = 0;
};

/** \brief Makes `device` current for the lifetime of the guard. */
class DeviceGuard {
public:
  explicit DeviceGuard(int device) {
    NDT_CUDA_CHECK(cudaGetDevice(&previous_));
    if (previous_ != device) {
      NDT_CUDA_CHECK(cudaSetDevice(device));
      switched_ = true;
    }
  }
  ~DeviceGuard() {
    if (switched_) cudaSetDevice(previous_);
  }

private:
  int previous_ = 0;
  bool switched_ = false;
};

int numBlocksFor(int num_points) {
  const int blocks = (num_points + kBlockSize - 1) / kBlockSize;
  return std::max(1, std::min(blocks, kMaxBlocks));
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// GpuBackend
// ---------------------------------------------------------------------------------------------

struct GpuBackend::Impl {
  int device = 0;
  cudaStream_t stream = nullptr;

  // target
  DeviceBuffer<int32_t> slot_keys;
  DeviceBuffer<uint32_t> slot_values;
  DeviceBuffer<DeviceVoxel> voxels;
  std::size_t num_voxels = 0;
  GridView grid{};
  double origin[3] = {0.0, 0.0, 0.0};
  bool has_target = false;

  // source
  DeviceBuffer<float4> source;
  std::size_t num_source = 0;

  // scratch
  DeviceBuffer<float4> score_points;
  DeviceBuffer<double> partials;
  DeviceBuffer<double> results;
  PinnedBuffer<double> host_results;
  PinnedBuffer<float4> host_staging;

  void ensureScratch(int num_blocks, int num_values) {
    partials.reserve(static_cast<std::size_t>(num_blocks) * num_values);
    results.reserve(num_values);
    host_results.reserve(num_values);
  }

  /** \brief Copies points to the device through the pinned staging buffer and waits for the copy. */
  void upload(DeviceBuffer<float4>& dst, const float* xyzw, std::size_t n) {
    dst.reserve(n);
    host_staging.reserve(n);
    std::memcpy(host_staging.get(), xyzw, n * sizeof(float4));
    NDT_CUDA_CHECK(cudaMemcpyAsync(dst.get(), host_staging.get(), n * sizeof(float4), cudaMemcpyHostToDevice, stream));
    // host_staging is reused by the next upload, so the copy must be finished before returning.
    NDT_CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void finishReduction(int num_blocks, int num_values) {
    finalReduceKernel<<<num_values, kBlockSize, 0, stream>>>(partials.get(), num_blocks, results.get());
    NDT_CUDA_CHECK(cudaGetLastError());
    NDT_CUDA_CHECK(cudaMemcpyAsync(host_results.get(), results.get(), num_values * sizeof(double), cudaMemcpyDeviceToHost, stream));
    NDT_CUDA_CHECK(cudaStreamSynchronize(stream));
  }
};

GpuBackend::GpuBackend(int device) : impl_(new Impl) {
  if (device < 0) {
    NDT_CUDA_CHECK(cudaGetDevice(&device));
  }
  impl_->device = device;
  DeviceGuard guard(device);
  NDT_CUDA_CHECK(cudaStreamCreateWithFlags(&impl_->stream, cudaStreamNonBlocking));
}

GpuBackend::~GpuBackend() {
  if (!impl_) return;
  int previous = -1;
  const bool switched = cudaGetDevice(&previous) == cudaSuccess && previous != impl_->device && cudaSetDevice(impl_->device) == cudaSuccess;
  if (impl_->stream) {
    cudaStreamSynchronize(impl_->stream);
    cudaStreamDestroy(impl_->stream);
  }
  impl_.reset();  // frees device buffers while the owning device is current
  if (switched) cudaSetDevice(previous);
}

int GpuBackend::device() const {
  return impl_->device;
}
std::size_t GpuBackend::numTargetVoxels() const {
  return impl_->num_voxels;
}
std::size_t GpuBackend::numSourcePoints() const {
  return impl_->num_source;
}

void GpuBackend::setTarget(const VoxelGridHost& grid) {
  Impl& m = *impl_;
  // Stays cleared if anything below fails: the old buffers may already be freed or overwritten.
  m.has_target = false;
  m.num_voxels = 0;
  DeviceGuard guard(m.device);
  const std::size_t n = grid.size();
  if (grid.means.size() != 3 * n || grid.icovs.size() != 6 * n || grid.flags.size() != n) {
    throw std::invalid_argument("[ndt_cuda] inconsistent VoxelGridHost array sizes");
  }
  if (n > kIndexMask) {
    throw std::invalid_argument("[ndt_cuda] too many voxels");
  }
  for (std::size_t i = 0; i < n; i++) {
    if (grid.keys[i] < 0) throw std::invalid_argument("[ndt_cuda] negative voxel key");
  }

  chooseOrigin(grid.min_b, grid.max_b, grid.leaf_size, m.grid.origin_cell, m.origin);
  for (int d = 0; d < 3; d++) {
    m.grid.min_b[d] = grid.min_b[d];
    m.grid.max_b[d] = grid.max_b[d];
    m.grid.divb_mul[d] = grid.divb_mul[d];
    m.grid.leaf_size[d] = grid.leaf_size[d];
  }

  // Voxel payload, relative to the origin
  std::vector<DeviceVoxel> voxels(n);
  for (std::size_t i = 0; i < n; i++) {
    const double* mu = &grid.means[3 * i];
    const double* ic = &grid.icovs[6 * i];
    DeviceVoxel& v = voxels[i];
    v.a = make_float4(static_cast<float>(mu[0] - m.origin[0]), static_cast<float>(mu[1] - m.origin[1]), static_cast<float>(mu[2] - m.origin[2]), static_cast<float>(ic[0]));
    v.b = make_float4(static_cast<float>(ic[1]), static_cast<float>(ic[2]), static_cast<float>(ic[3]), static_cast<float>(ic[4]));
    v.c = make_float4(static_cast<float>(ic[5]), 0.0f, 0.0f, 0.0f);
  }

  std::vector<int32_t> slot_keys;
  std::vector<uint32_t> slot_values;
  const std::size_t capacity = buildHashTable(grid.keys, grid.flags, slot_keys, slot_values);

  m.slot_keys.reserve(capacity);
  m.slot_values.reserve(capacity);
  m.voxels.reserve(std::max<std::size_t>(n, 1));
  NDT_CUDA_CHECK(cudaMemcpyAsync(m.slot_keys.get(), slot_keys.data(), capacity * sizeof(int32_t), cudaMemcpyHostToDevice, m.stream));
  NDT_CUDA_CHECK(cudaMemcpyAsync(m.slot_values.get(), slot_values.data(), capacity * sizeof(uint32_t), cudaMemcpyHostToDevice, m.stream));
  if (n > 0) {
    NDT_CUDA_CHECK(cudaMemcpyAsync(m.voxels.get(), voxels.data(), n * sizeof(DeviceVoxel), cudaMemcpyHostToDevice, m.stream));
  }
  // The host vectors go out of scope at return, so wait for the copies.
  NDT_CUDA_CHECK(cudaStreamSynchronize(m.stream));

  m.grid.slot_keys = m.slot_keys.get();
  m.grid.slot_values = m.slot_values.get();
  m.grid.voxels = m.voxels.get();
  m.grid.slot_mask = static_cast<uint32_t>(capacity - 1);
  m.num_voxels = n;
  m.has_target = true;
}

void GpuBackend::setSource(const float* xyzw, std::size_t num_points) {
  DeviceGuard guard(impl_->device);
  Impl& m = *impl_;
  if (num_points > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("[ndt_cuda] too many source points");
  }
  m.num_source = 0;
  if (num_points > 0) {
    m.upload(m.source, xyzw, num_points);
  }
  m.num_source = num_points;
}

DerivativeResult GpuBackend::computeDerivatives(const DerivativeParams& params) {
  DeviceGuard guard(impl_->device);
  Impl& m = *impl_;

  DerivativeResult result;
  std::memset(&result, 0, sizeof(result));
  if (!m.has_target || m.num_voxels == 0 || m.num_source == 0) {
    return result;
  }

  KernelParams p;
  std::memset(&p, 0, sizeof(p));
  setTransform(p, params.transform, m.origin);
  std::memcpy(p.j_ang, params.j_ang, sizeof(p.j_ang));
  std::memcpy(p.h_ang, params.h_ang, sizeof(p.h_ang));
  p.gauss_d1 = static_cast<float>(params.gauss_d1);
  p.gauss_d2 = static_cast<float>(params.gauss_d2);
  p.radius2 = params.radius * params.radius;

  const int num_points = static_cast<int>(m.num_source);
  const int num_blocks = numBlocksFor(num_points);
  m.ensureScratch(num_blocks, kNumDerivativeValues);

#define NDT_CUDA_LAUNCH_DERIVATIVES(MODE)                                                                                           \
  if (params.compute_hessian)                                                                                                       \
    derivativesKernel<MODE, true><<<num_blocks, kBlockSize, 0, m.stream>>>(m.source.get(), num_points, m.grid, p, m.partials.get());  \
  else                                                                                                                              \
    derivativesKernel<MODE, false><<<num_blocks, kBlockSize, 0, m.stream>>>(m.source.get(), num_points, m.grid, p, m.partials.get());

  switch (params.search_method) {
    case KDTREE:
      NDT_CUDA_LAUNCH_DERIVATIVES(kModeKdtree);
      break;
    case DIRECT26:
      NDT_CUDA_LAUNCH_DERIVATIVES(kModeDirect26);
      break;
    case DIRECT1:
      NDT_CUDA_LAUNCH_DERIVATIVES(kModeDirect1);
      break;
    case DIRECT7:
    default:
      NDT_CUDA_LAUNCH_DERIVATIVES(kModeDirect7);
      break;
  }
#undef NDT_CUDA_LAUNCH_DERIVATIVES
  NDT_CUDA_CHECK(cudaGetLastError());

  m.finishReduction(num_blocks, kNumDerivativeValues);

  const double* r = m.host_results.get();
  result.score = r[0];
  for (int i = 0; i < 6; i++) {
    result.gradient[i] = r[1 + i];
  }
  int h = 7;
  for (int i = 0; i < 6; i++) {
    for (int j = i; j < 6; j++, h++) {
      result.hessian[i * 6 + j] = r[h];
      result.hessian[j * 6 + i] = r[h];
    }
  }
  return result;
}

double GpuBackend::calculateScore(const float* xyzw, std::size_t num_points, double gauss_d1, double gauss_d2, double gauss_d3, float radius, int search_method) {
  DeviceGuard guard(impl_->device);
  Impl& m = *impl_;
  if (!m.has_target || m.num_voxels == 0 || num_points == 0) {
    return 0.0;
  }
  if (num_points > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("[ndt_cuda] too many points");
  }

  m.upload(m.score_points, xyzw, num_points);

  const float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  KernelParams p;
  std::memset(&p, 0, sizeof(p));
  setTransform(p, identity, m.origin);
  p.gauss_d1 = static_cast<float>(gauss_d1);
  p.gauss_d2 = static_cast<float>(gauss_d2);
  p.gauss_d3 = static_cast<float>(gauss_d3);
  p.radius2 = radius * radius;

  const int n = static_cast<int>(num_points);
  const int num_blocks = numBlocksFor(n);
  m.ensureScratch(num_blocks, kNumScoreValues);

  switch (search_method) {
    case KDTREE:
      scoreKernel<kModeKdtree><<<num_blocks, kBlockSize, 0, m.stream>>>(m.score_points.get(), n, m.grid, p, m.partials.get());
      break;
    case DIRECT26:
      scoreKernel<kModeDirect26><<<num_blocks, kBlockSize, 0, m.stream>>>(m.score_points.get(), n, m.grid, p, m.partials.get());
      break;
    case DIRECT1:
      scoreKernel<kModeDirect1><<<num_blocks, kBlockSize, 0, m.stream>>>(m.score_points.get(), n, m.grid, p, m.partials.get());
      break;
    case DIRECT7:
    default:
      scoreKernel<kModeDirect7><<<num_blocks, kBlockSize, 0, m.stream>>>(m.score_points.get(), n, m.grid, p, m.partials.get());
      break;
  }
  NDT_CUDA_CHECK(cudaGetLastError());

  m.finishReduction(num_blocks, kNumScoreValues);
  return m.host_results.get()[0] / static_cast<double>(num_points);
}

int GpuBackend::deviceCount() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    cudaGetLastError();  // clear the error state
    return 0;
  }
  return count;
}

std::string GpuBackend::deviceName(int device) {
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
    cudaGetLastError();
    return "unknown";
  }
  return prop.name;
}

}  // namespace ndt_cuda
