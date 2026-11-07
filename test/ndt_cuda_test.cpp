// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Melike
//
// Validation of ndt_cuda::NormalDistributionsTransformCUDA without a GPU.
//
// The test is linked against test/gpu_backend_emulated.cpp, which runs the exact per-point code of the CUDA kernels
// (src/ndt_cuda_kernels.cuh) on the CPU. koide3/ndt_omp is used as the reference implementation. It checks:
//   0. the voxel grid is identical to ndt_omp's,
//   1. score / gradient / hessian against ndt_omp at several poses, for every neighbor search method,
//   2. gradient and hessian against central finite differences on a synthetic grid,
//   3. end-to-end align() results against ndt_omp, including large map coordinates,
//   4. drop-in compatibility (pcl::Registration interface, pclomp_compat.hpp),
//   5. target handling (empty target, no kd-tree build on the map, not copyable).
//
// usage: ndt_cuda_test [data_dir]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pclomp/ndt_omp.h>

#include <ndt_cuda/angle_derivatives.hpp>
#include <ndt_cuda/gpu_backend.hpp>
#include <ndt_cuda/ndt_cuda.hpp>
#include <ndt_cuda/pclomp_compat.hpp>  // after <pclomp/ndt_omp.h>, as documented
#include <ndt_cuda/voxel_grid_covariance.hpp>

using PointT = pcl::PointXYZ;
using Cloud = pcl::PointCloud<PointT>;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat6 = Eigen::Matrix<double, 6, 6>;

static int g_failures = 0;

static void check(bool ok, const char* what, double value, double tol) {
  std::printf("  %-58s %12.3e  (tol %.0e)  %s\n", what, value, tol, ok ? "OK" : "FAIL");
  if (!ok) g_failures++;
}

static double relErr(const double* a, const double* b, int n) {
  double num = 0.0, den = 0.0;
  for (int i = 0; i < n; i++) {
    num = std::max(num, std::fabs(a[i] - b[i]));
    den = std::max(den, std::fabs(b[i]));
  }
  return den > 0 ? num / den : num;
}

static const char* methodName(int m) {
  static const char* names[] = {"KDTREE", "DIRECT26", "DIRECT7", "DIRECT1"};
  return names[m];
}

static void toRowMajor(const Eigen::Matrix4f& T, float out[16]) {
  for (int r = 0; r < 4; r++)
    for (int c = 0; c < 4; c++) out[r * 4 + c] = T(r, c);
}

static void gaussParams(double resolution, double outlier_ratio, double& d1, double& d2, double& d3) {
  const double c1 = 10.0 * (1 - outlier_ratio);
  const double c2 = outlier_ratio / std::pow(resolution, 3);
  d3 = -std::log(c2);
  d1 = -std::log(c1 + c2) - d3;
  d2 = -2 * std::log((-std::log(c1 * std::exp(-0.5) + c2) - d3) / d1);
}

/** \brief Exposes the internals of the OpenMP NDT. */
struct OmpProbe : public pclomp::NormalDistributionsTransform<PointT, PointT> {
  double eval(const Vec6& p, Vec6& g, Mat6& H) {
    Eigen::Matrix4f T;
    convertTransform(p, T);
    Cloud trans;
    pcl::transformPointCloud(*input_, trans, T);
    Vec6 pp = p;
    return computeDerivatives(g, H, trans, pp, true);
  }
};

/** \brief Evaluates the (emulated) CUDA backend. ndt_omp_sign reproduces ndt_omp's float-path sign of h_ang d1. */
static ndt_cuda::DerivativeResult evalBackend(ndt_cuda::GpuBackend& backend, const Vec6& p, int mode, float resolution, bool ndt_omp_sign) {
  ndt_cuda::DerivativeParams params;
  Eigen::Matrix4f T;
  pclomp::NormalDistributionsTransform<PointT, PointT>::convertTransform(p, T);
  toRowMajor(T, params.transform);
  ndt_cuda::computeAngleDerivatives(p, true, params.j_ang, params.h_ang);
  if (ndt_omp_sign) params.h_ang[6][2] = -params.h_ang[6][2];
  double d3;
  gaussParams(resolution, 0.55, params.gauss_d1, params.gauss_d2, d3);
  params.radius = resolution;
  params.search_method = mode;
  params.compute_hessian = true;
  return backend.computeDerivatives(params);
}

static std::vector<float> pack(const Cloud& cloud) {
  std::vector<float> v;
  v.reserve(cloud.size() * 4);
  for (const auto& pt : cloud) v.insert(v.end(), {pt.x, pt.y, pt.z, 0.0f});
  return v;
}

// ---------------------------------------------------------------------------------------------
// 0. Voxel grid identical to ndt_omp's
// ---------------------------------------------------------------------------------------------
static void testVoxelGrid(const Cloud::Ptr& target) {
  std::printf("\n[0] voxel grid vs pclomp::VoxelGridCovariance\n");
  for (float resolution : {0.5f, 1.0f, 2.0f}) {
    pclomp::VoxelGridCovariance<PointT> ref;
    ref.setLeafSize(resolution, resolution, resolution);
    ref.setInputCloud(target);
    ref.filter(false);

    ndt_cuda::VoxelGridCovariance<PointT> ours;
    ours.setLeafSize(resolution);
    ours.build(*target);

    const bool same_box = ref.getMinBoxCoordinates() == ours.getMinBoxCoordinates() && ref.getMaxBoxCoordinates() == ours.getMaxBoxCoordinates() &&
                          ref.getDivisionMultiplier() == ours.getDivisionMultiplier();
    std::size_t compared = 0, mismatches = 0;
    double max_mean = 0.0, max_icov = 0.0;
    const auto& ref_leaves = ref.getLeaves();
    for (const auto& [key, leaf] : ref_leaves) {
      const auto it = ours.getLeaves().find(static_cast<int>(key));
      if (it == ours.getLeaves().end() || it->second.nr_points != leaf.nr_points) {
        mismatches++;
        continue;
      }
      if (leaf.nr_points < ref.getMinPointPerVoxel()) continue;
      compared++;
      max_mean = std::max(max_mean, (it->second.mean - leaf.mean_).cwiseAbs().maxCoeff());
      max_icov = std::max(max_icov, (it->second.icov - leaf.icov_).cwiseAbs().maxCoeff());
    }
    mismatches += ours.getLeaves().size() != ref_leaves.size() ? 1 : 0;
    char label[128];
    std::snprintf(label, sizeof(label), "res %.1f: bounding box / leaf set / point counts mismatches", resolution);
    check(same_box && mismatches == 0, label, double(mismatches + (same_box ? 0 : 1)), 0.0);
    std::snprintf(label, sizeof(label), "res %.1f: max |mean diff| over %zu voxels", resolution, compared);
    check(max_mean == 0.0, label, max_mean, 0.0);
    std::snprintf(label, sizeof(label), "res %.1f: max |inverse covariance diff|", resolution);
    check(max_icov == 0.0, label, max_icov, 0.0);
  }
}

// ---------------------------------------------------------------------------------------------
// 1. Per-evaluation comparison with ndt_omp
// ---------------------------------------------------------------------------------------------
static void testDerivativesAgainstOmp(const Cloud::Ptr& target, const Cloud::Ptr& source) {
  std::printf("\n[1] score / gradient / hessian vs pclomp::NormalDistributionsTransform\n");
  const float resolution = 1.0f;

  OmpProbe omp;
  omp.setResolution(resolution);
  omp.setInputTarget(target);
  omp.setInputSource(source);

  ndt_cuda::VoxelGridCovariance<PointT> cells;
  cells.setLeafSize(resolution);
  cells.build(*target);
  ndt_cuda::VoxelGridHost grid;
  cells.exportTo(grid);

  ndt_cuda::GpuBackend backend;
  backend.setTarget(grid);
  const std::vector<float> packed = pack(*source);
  backend.setSource(packed.data(), source->size());
  std::printf("  %zu voxels, %zu source points\n", backend.numTargetVoxels(), backend.numSourcePoints());

  std::vector<Vec6> poses(3);
  poses[0] << 0, 0, 0, 0, 0, 0;
  poses[1] << 0.3, -0.2, 0.05, 0.02, -0.03, 0.05;
  poses[2] << -0.5, 0.4, -0.1, -0.05, 0.04, -0.12;

  for (int mode = 0; mode < 4; mode++) {
    omp.setNeighborhoodSearchMethod(static_cast<pclomp::NeighborSearchMethod>(mode));
    double worst_s = 0, worst_g = 0, worst_h = 0, worst_h_fixed = 0;
    for (const Vec6& p : poses) {
      Vec6 g;
      Mat6 H;
      const double s = omp.eval(p, g, H);
      const auto r = evalBackend(backend, p, mode, resolution, true);
      Mat6 Hr = Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(r.hessian);
      worst_s = std::max(worst_s, std::fabs(r.score - s) / std::fabs(s));
      worst_g = std::max(worst_g, relErr(r.gradient, g.data(), 6));
      worst_h = std::max(worst_h, relErr(Hr.data(), H.data(), 36));

      // With the corrected sign, only H(4,4) may differ from ndt_omp.
      const auto rf = evalBackend(backend, p, mode, resolution, false);
      Mat6 Hf = Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(rf.hessian);
      Hf(4, 4) = H(4, 4);
      worst_h_fixed = std::max(worst_h_fixed, relErr(Hf.data(), H.data(), 36));
    }
    char label[128];
    std::snprintf(label, sizeof(label), "%s score rel. error", methodName(mode));
    check(worst_s < 1e-4, label, worst_s, 1e-4);
    std::snprintf(label, sizeof(label), "%s gradient rel. error", methodName(mode));
    check(worst_g < 1e-4, label, worst_g, 1e-4);
    std::snprintf(label, sizeof(label), "%s hessian rel. error (ndt_omp sign)", methodName(mode));
    check(worst_h < 1e-4, label, worst_h, 1e-4);
    std::snprintf(label, sizeof(label), "%s hessian rel. error excl. H(4,4) (fixed sign)", methodName(mode));
    check(worst_h_fixed < 1e-4, label, worst_h_fixed, 1e-4);
  }

  // calculateScore()
  for (int mode = 0; mode < 4; mode++) {
    omp.setNeighborhoodSearchMethod(static_cast<pclomp::NeighborSearchMethod>(mode));
    Eigen::Matrix4f T;
    pclomp::NormalDistributionsTransform<PointT, PointT>::convertTransform(poses[1], T);
    Cloud trans;
    pcl::transformPointCloud(*source, trans, T);
    const double s_omp = omp.calculateScore(trans);
    const std::vector<float> pt = pack(trans);
    double d1, d2, d3;
    gaussParams(resolution, 0.55, d1, d2, d3);
    const double s_gpu = backend.calculateScore(pt.data(), trans.size(), d1, d2, d3, resolution, mode);
    char label[128];
    std::snprintf(label, sizeof(label), "%s calculateScore rel. error", methodName(mode));
    const double err = std::fabs(s_gpu - s_omp) / std::fabs(s_omp);
    check(err < 1e-4, label, err, 1e-4);
  }
}

// ---------------------------------------------------------------------------------------------
// 2. Finite differences on a synthetic grid
// ---------------------------------------------------------------------------------------------
static void testFiniteDifferences() {
  std::printf("\n[2] gradient / hessian vs central finite differences (synthetic grid)\n");
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> u(-1.0, 1.0);

  ndt_cuda::VoxelGridHost grid;
  const int lo[3] = {-3, -3, -1}, hi[3] = {3, 3, 1};
  for (int d = 0; d < 3; d++) {
    grid.leaf_size[d] = 1.0f;
    grid.min_b[d] = lo[d];
    grid.max_b[d] = hi[d];
  }
  grid.divb_mul[0] = 1;
  grid.divb_mul[1] = hi[0] - lo[0] + 1;
  grid.divb_mul[2] = grid.divb_mul[1] * (hi[1] - lo[1] + 1);

  std::vector<Eigen::Vector3d> world;
  for (int z = lo[2]; z <= hi[2]; z++) {
    for (int y = lo[1]; y <= hi[1]; y++) {
      for (int x = lo[0]; x <= hi[0]; x++) {
        const Eigen::Vector3d center(x + 0.5, y + 0.5, z + 0.5);
        const Eigen::Vector3d mean = center + 0.1 * Eigen::Vector3d(u(rng), u(rng), u(rng));
        const Eigen::Matrix3d R = Eigen::AngleAxisd(u(rng) * M_PI, Eigen::Vector3d(u(rng), u(rng), u(rng)).normalized()).toRotationMatrix();
        const Eigen::Vector3d s(0.11 + 0.09 * u(rng), 0.11 + 0.09 * u(rng), 0.03 + 0.01 * u(rng));
        const Eigen::Matrix3d icov = (R * s.asDiagonal() * R.transpose()).inverse();
        grid.keys.push_back((x - lo[0]) * grid.divb_mul[0] + (y - lo[1]) * grid.divb_mul[1] + (z - lo[2]) * grid.divb_mul[2]);
        grid.means.insert(grid.means.end(), {mean(0), mean(1), mean(2)});
        grid.icovs.insert(grid.icovs.end(), {icov(0, 0), icov(0, 1), icov(0, 2), icov(1, 1), icov(1, 2), icov(2, 2)});
        grid.flags.push_back(ndt_cuda::VoxelGridHost::kFlagDirect | ndt_cuda::VoxelGridHost::kFlagCentroid);
        for (int i = 0; i < 30; i++) world.push_back(center + 0.3 * Eigen::Vector3d(u(rng), u(rng), u(rng)));
      }
    }
  }

  Vec6 p0;
  p0 << 0.3, -0.2, 0.1, 0.15, -0.25, 0.4;
  Eigen::Matrix4f T0;
  pclomp::NormalDistributionsTransform<PointT, PointT>::convertTransform(p0, T0);
  const Eigen::Matrix4d T0inv = T0.cast<double>().inverse();
  std::vector<float> src;
  for (const auto& y : world) {
    const Eigen::Vector3d x = (T0inv * y.homogeneous()).head<3>();
    src.insert(src.end(), {float(x(0)), float(x(1)), float(x(2)), 0.0f});
  }

  ndt_cuda::GpuBackend backend;
  backend.setTarget(grid);
  backend.setSource(src.data(), world.size());

  const double h = 1e-3;
  for (int mode : {pclomp::DIRECT1, pclomp::DIRECT7, pclomp::DIRECT26}) {
    const auto r0 = evalBackend(backend, p0, mode, 1.0f, false);
    const auto r0_omp_sign = evalBackend(backend, p0, mode, 1.0f, true);
    Mat6 H = Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(r0.hessian);
    Mat6 H_omp_sign = Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(r0_omp_sign.hessian);
    Mat6 H_fd;
    Vec6 g_fd;
    for (int k = 0; k < 6; k++) {
      Vec6 pp = p0, pm = p0;
      pp(k) += h;
      pm(k) -= h;
      const auto rp = evalBackend(backend, pp, mode, 1.0f, false);
      const auto rm = evalBackend(backend, pm, mode, 1.0f, false);
      g_fd(k) = (rp.score - rm.score) / (2 * h);
      for (int i = 0; i < 6; i++) H_fd(i, k) = (rp.gradient[i] - rm.gradient[i]) / (2 * h);
    }
    char label[128];
    const double eg = relErr(r0.gradient, g_fd.data(), 6);
    std::snprintf(label, sizeof(label), "%s gradient vs FD", methodName(mode));
    check(eg < 1e-2, label, eg, 1e-2);
    const double eh = relErr(H.data(), H_fd.data(), 36);
    std::snprintf(label, sizeof(label), "%s hessian vs FD", methodName(mode));
    check(eh < 1e-2, label, eh, 1e-2);
    const double e44 = std::fabs(H(4, 4) - H_fd(4, 4)) / std::fabs(H_fd(4, 4));
    const double e44_omp = std::fabs(H_omp_sign(4, 4) - H_fd(4, 4)) / std::fabs(H_fd(4, 4));
    std::snprintf(label, sizeof(label), "%s H(4,4) vs FD, -sin(pitch) (used here)", methodName(mode));
    check(e44 < 1e-2, label, e44, 1e-2);
    std::printf("  %-58s %12.3e  (ndt_omp float path, info only)\n", (std::string(methodName(mode)) + " H(4,4) vs FD, +sin(pitch)").c_str(), e44_omp);
  }
}

// ---------------------------------------------------------------------------------------------
// 3. End-to-end registration
// ---------------------------------------------------------------------------------------------
static void transformDiff(const Eigen::Matrix4d& a, const Eigen::Matrix4d& b, double& dt, double& dr_deg) {
  const Eigen::Matrix4d d = a.inverse() * b;
  dt = d.block<3, 1>(0, 3).norm();
  const double c = std::max(-1.0, std::min(1.0, (d.block<3, 3>(0, 0).trace() - 1.0) / 2.0));
  dr_deg = std::acos(c) * 180.0 / M_PI;
}

static void testAlignment(const Cloud::Ptr& target, const Cloud::Ptr& source) {
  std::printf("\n[3] align() vs pclomp::NormalDistributionsTransform\n");

  Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();
  guess.block<3, 3>(0, 0) = Eigen::AngleAxisf(0.05f, Eigen::Vector3f::UnitZ()).toRotationMatrix();
  guess.block<3, 1>(0, 3) = Eigen::Vector3f(0.4f, -0.3f, 0.05f);

  for (int with_guess = 0; with_guess < 2; with_guess++) {
    for (int mode = 0; mode < 4; mode++) {
      pclomp::NormalDistributionsTransform<PointT, PointT> omp;
      pclomp::NormalDistributionsTransformCUDA<PointT, PointT> gpu;  // alias from <ndt_cuda/pclomp_compat.hpp>
      for (auto* reg : std::vector<pcl::Registration<PointT, PointT>*>{&omp, &gpu}) {
        reg->setInputTarget(target);
        reg->setInputSource(source);
      }
      omp.setResolution(1.0);
      gpu.setResolution(1.0);
      omp.setNeighborhoodSearchMethod(static_cast<pclomp::NeighborSearchMethod>(mode));
      gpu.setNeighborhoodSearchMethod(static_cast<pclomp::NeighborSearchMethod>(mode));  // ndt_omp enum accepted

      Cloud a_omp, a_gpu;
      if (with_guess) {
        omp.align(a_omp, guess);
        gpu.align(a_gpu, guess);
      } else {
        omp.align(a_omp);
        gpu.align(a_gpu);
      }

      double dt, dr;
      transformDiff(omp.getFinalTransformation().cast<double>(), gpu.getFinalTransformation().cast<double>(), dt, dr);
      // The aligned output cloud must be the source transformed by the final transformation.
      Cloud expected;
      pcl::transformPointCloud(*source, expected, gpu.getFinalTransformation());
      double max_out = 0.0;
      for (std::size_t i = 0; i < expected.size(); i++) {
        max_out = std::max(max_out, double((expected[i].getVector3fMap() - a_gpu[i].getVector3fMap()).norm()));
      }

      std::printf("  %-8s %-10s iters omp/gpu %2d/%2d  trans_prob %.5f/%.5f  dt %.2e m  dr %.2e deg\n", methodName(mode), with_guess ? "guess" : "identity", omp.getFinalNumIteration(),
                  gpu.getFinalNumIteration(), omp.getTransformationProbability(), gpu.getTransformationProbability(), dt, dr);
      char label[128];
      std::snprintf(label, sizeof(label), "%s %s transform translation diff [m]", methodName(mode), with_guess ? "guess" : "identity");
      check(dt < 1e-2, label, dt, 1e-2);
      std::snprintf(label, sizeof(label), "%s %s transform rotation diff [deg]", methodName(mode), with_guess ? "guess" : "identity");
      check(dr < 5e-2, label, dr, 5e-2);
      std::snprintf(label, sizeof(label), "%s %s aligned cloud consistency [m]", methodName(mode), with_guess ? "guess" : "identity");
      check(max_out < 1e-5, label, max_out, 1e-5);
    }
  }

  // Large map coordinates, as in localization: the map is far from the origin, the scan stays in
  // the sensor frame and the initial guess carries the large translation.
  std::printf("\n[3b] large map coordinates (target shifted by (40000, -25000, 50) m, guess = shift)\n");
  const Eigen::Vector3d offset(40000.0, -25000.0, 50.0);
  Cloud::Ptr target_s(new Cloud(*target));
  for (auto& pt : *target_s) pt.getVector3fMap() = (pt.getVector3fMap().cast<double>() + offset).cast<float>();
  Eigen::Matrix4d S = Eigen::Matrix4d::Identity();
  S.block<3, 1>(0, 3) = offset;

  for (int mode : {pclomp::DIRECT7, pclomp::KDTREE}) {
    ndt_cuda::NormalDistributionsTransformCUDA<PointT, PointT> gpu_ref, gpu_s;
    pclomp::NormalDistributionsTransform<PointT, PointT> omp_s;
    for (auto* reg : std::vector<pcl::Registration<PointT, PointT>*>{&gpu_ref, &gpu_s, &omp_s}) {
      reg->setInputSource(source);
    }
    gpu_ref.setResolution(1.0);
    gpu_s.setResolution(1.0);
    omp_s.setResolution(1.0);
    gpu_ref.setNeighborhoodSearchMethod(static_cast<ndt_cuda::NeighborSearchMethod>(mode));
    gpu_s.setNeighborhoodSearchMethod(static_cast<ndt_cuda::NeighborSearchMethod>(mode));
    omp_s.setNeighborhoodSearchMethod(static_cast<pclomp::NeighborSearchMethod>(mode));
    gpu_ref.setInputTarget(target);
    gpu_s.setInputTarget(target_s);
    omp_s.setInputTarget(target_s);

    Cloud out;
    gpu_ref.align(out);
    gpu_s.align(out, S.cast<float>());
    omp_s.align(out, S.cast<float>());

    const Eigen::Matrix4d expected = S * gpu_ref.getFinalTransformation().cast<double>();
    double dt, dr, dt_o, dr_o;
    transformDiff(expected, gpu_s.getFinalTransformation().cast<double>(), dt, dr);
    transformDiff(expected, omp_s.getFinalTransformation().cast<double>(), dt_o, dr_o);
    char label[128];
    std::snprintf(label, sizeof(label), "%s cuda: shifted vs unshifted, translation [m]", methodName(mode));
    check(dt < 1e-2, label, dt, 1e-2);
    std::snprintf(label, sizeof(label), "%s cuda: shifted vs unshifted, rotation [deg]", methodName(mode));
    check(dr < 5e-2, label, dr, 5e-2);
    std::printf("  %-58s %12.3e  (info; rotation %.2e deg, iters omp %d / cuda %d)\n", (std::string(methodName(mode)) + " omp : shifted vs unshifted, translation [m]").c_str(), dt_o, dr_o,
                omp_s.getFinalNumIteration(), gpu_s.getFinalNumIteration());
  }
}

// ---------------------------------------------------------------------------------------------
// 4. Drop-in compatibility
// ---------------------------------------------------------------------------------------------
int compatStandaloneDirect7();  // compat_standalone.cpp: <ndt_cuda/pclomp_compat.hpp> without ndt_omp

static void testCompat(const Cloud::Ptr& target, const Cloud::Ptr& source) {
  std::printf("\n[4] drop-in compatibility\n");
  // Used through the pcl::Registration interface, exactly like pclomp::NormalDistributionsTransform.
  pcl::Registration<PointT, PointT>::Ptr reg(new pclomp::NormalDistributionsTransformCUDA<PointT, PointT>());
  reg->setInputTarget(target);
  reg->setInputSource(source);
  Cloud out;
  reg->align(out);
  check(reg->hasConverged(), "align() through pcl::Registration::Ptr converged", reg->hasConverged() ? 0.0 : 1.0, 0.0);
  check(compatStandaloneDirect7() == static_cast<int>(ndt_cuda::DIRECT7), "pclomp::DIRECT7 without ndt_omp (compat header)", 0.0, 0.0);
}

// ---------------------------------------------------------------------------------------------
// 5. Target handling
// ---------------------------------------------------------------------------------------------
static void testTargetHandling(const Cloud::Ptr& target, const Cloud::Ptr& source) {
  std::printf("\n[5] target handling\n");
  using NDT = ndt_cuda::NormalDistributionsTransformCUDA<PointT, PointT>;
  static_assert(!std::is_copy_constructible_v<NDT> && !std::is_copy_assignable_v<NDT>, "an instance owns its GPU state");

  // pcl::Registration rejects an empty cloud; this must not crash and align() must report the failure.
  NDT empty;
  empty.setInputTarget(Cloud::Ptr(new Cloud));
  empty.setInputSource(source);
  Cloud out;
  empty.align(out);
  check(!empty.hasConverged(), "empty target as first target: align() does not converge", empty.hasConverged() ? 1.0 : 0.0, 0.0);

  // initCompute() builds its kd-tree only while this flag is set.
  struct Probe : NDT {
    bool kdtreePending() const { return this->target_cloud_updated_; }
  };
  Probe probe;
  probe.setInputTarget(target);
  check(!probe.kdtreePending(), "no kd-tree pending after setInputTarget()", probe.kdtreePending() ? 1.0 : 0.0, 0.0);
  probe.setInputSource(source);
  probe.align(out);
  check(probe.hasConverged(), "align() without the kd-tree converged", probe.hasConverged() ? 0.0 : 1.0, 0.0);
}

int main(int argc, char** argv) {
  const std::string data_dir = argc > 1 ? argv[1] : NDT_CUDA_DATA_DIR;

  Cloud::Ptr target(new Cloud), source(new Cloud);
  if (pcl::io::loadPCDFile(data_dir + "/251370668.pcd", *target) || pcl::io::loadPCDFile(data_dir + "/251371071.pcd", *source)) {
    std::fprintf(stderr, "failed to load the test clouds from %s\n", data_dir.c_str());
    return 1;
  }
  pcl::VoxelGrid<PointT> voxelgrid;
  voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);
  Cloud::Ptr tmp(new Cloud);
  voxelgrid.setInputCloud(target);
  voxelgrid.filter(*tmp);
  *target = *tmp;
  voxelgrid.setInputCloud(source);
  voxelgrid.filter(*tmp);
  *source = *tmp;

  testVoxelGrid(target);
  testDerivativesAgainstOmp(target, source);
  testFiniteDifferences();
  testAlignment(target, source);
  testCompat(target, source);
  testTargetHandling(target, source);

  std::printf("\n%s (%d failure%s)\n", g_failures ? "FAILED" : "PASSED", g_failures, g_failures == 1 ? "" : "s");
  return g_failures ? 1 : 0;
}
