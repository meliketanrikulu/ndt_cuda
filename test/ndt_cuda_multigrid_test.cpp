// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Melike
//
// Validation of ndt_cuda::MultiGridNormalDistributionsTransformCUDA and of the tier4/ndt_omp additions of
// ndt_cuda::NormalDistributionsTransformCUDA, with tier4/ndt_omp as the reference. With Autoware's default parameters, it checks:
//   1. align() with the map as one tile,
//   2. a map split into tiles whose borders lie on voxel borders,
//   3. a map split into tiles that share voxels,
//   4. removing tiles and replacing a tile,
//   5. copies, as Autoware's map update makes them,
//   6. updating a copy on another thread while the original aligns,
//   7. regularization,
//   8. the single grid class against tier4's pclomp::NormalDistributionsTransform.
//
// Linked with test/gpu_backend_emulated.cpp, it needs no GPU; linked with the library, it runs the real CUDA kernels.
//
// usage: ndt_cuda_multigrid_test [data_dir]      exit code 77 (reported as skipped by ctest) if no GPU is available

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <multigrid_pclomp/multigrid_ndt_omp.h>
#include <pclomp/ndt_omp.h>

#include <ndt_cuda/pclomp_compat.hpp>  // after the tier4/ndt_omp headers, as documented

using PointT = pcl::PointXYZ;
using Cloud = pcl::PointCloud<PointT>;
using Reference = pclomp::MultiGridNormalDistributionsTransform<PointT, PointT>;
using Ours = pclomp::MultiGridNormalDistributionsTransformCUDA<PointT, PointT>;
using ReferenceSingle = pclomp::NormalDistributionsTransform<PointT, PointT>;
using OursSingle = pclomp::NormalDistributionsTransformCUDA<PointT, PointT>;
using Tiles = std::map<std::string, Cloud::Ptr>;

static int g_failures = 0;

static void check(bool ok, const char* what, double value, double tol) {
  std::printf("  %-66s %12.3e  (tol %.0e)  %s\n", what, value, tol, ok ? "OK" : "FAIL");
  if (!ok) g_failures++;
}

static void checkLabel(bool ok, const char* name, const char* what, double value, double tol) {
  char label[128];
  std::snprintf(label, sizeof(label), "%s: %s", name, what);
  check(ok, label, value, tol);
}

static void transformDiff(const Eigen::Matrix4f& a, const Eigen::Matrix4f& b, double& dt, double& dr_deg) {
  const Eigen::Matrix4d d = a.cast<double>().inverse() * b.cast<double>();
  dt = d.block<3, 1>(0, 3).norm();
  const double c = std::max(-1.0, std::min(1.0, (d.block<3, 3>(0, 0).trace() - 1.0) / 2.0));
  dr_deg = std::acos(c) * 180.0 / M_PI;
}

static double relative(double a, double b) {
  return std::fabs(a - b) / std::max(std::fabs(b), 1e-12);
}

/** \brief Autoware's default parameters (autoware_ndt_scan_matcher/config/ndt_scan_matcher.param.yaml). */
template <typename Params>
static Params autowareParams() {
  Params params;
  params.trans_epsilon = 0.01;
  params.step_size = 0.1;
  params.resolution = 2.0f;
  params.max_iterations = 30;
  params.num_threads = 4;
  params.regularization_scale_factor = 0.01f;
  return params;
}

static Eigen::Matrix4f perturbedGuess() {
  Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();
  guess.block<3, 3>(0, 0) = Eigen::AngleAxisf(0.03f, Eigen::Vector3f::UnitZ()).toRotationMatrix();
  guess.block<3, 1>(0, 3) = Eigen::Vector3f(0.5f, -0.4f, 0.05f);
  return guess;
}

static Tiles splitIntoTiles(const Cloud& map, float tile_size) {
  Tiles tiles;
  for (const auto& pt : map) {
    const std::string id = std::to_string(static_cast<int>(std::floor(pt.x / tile_size))) + "_" + std::to_string(static_cast<int>(std::floor(pt.y / tile_size)));
    Cloud::Ptr& tile = tiles[id];
    if (!tile) tile.reset(new Cloud);
    tile->push_back(pt);
  }
  return tiles;
}

/** \brief Aligns both and compares everything Autoware's ndt_scan_matcher reads from the result. */
template <typename RefT, typename OursT>
static void compare(const char* name, RefT& reference, OursT& ours, const Cloud::Ptr& source, const Eigen::Matrix4f& guess) {
  reference.setInputSource(source);
  ours.setInputSource(source);
  Cloud out_ref, out_ours;
  reference.align(out_ref, guess);
  ours.align(out_ours, guess);
  const auto r = reference.getResult();
  const auto o = ours.getResult();

  double dt, dr;
  transformDiff(r.pose, o.pose, dt, dr);
  std::printf("  %s: iterations %d/%d  TP %.5f/%.5f  NVTL %.5f/%.5f\n", name, r.iteration_num, o.iteration_num, r.transform_probability, o.transform_probability, r.nearest_voxel_transformation_likelihood, o.nearest_voxel_transformation_likelihood);
  checkLabel(dt < 2e-3, name, "translation diff [m]", dt, 2e-3);
  checkLabel(dr < 5e-2, name, "rotation diff [deg]", dr, 5e-2);
  checkLabel(r.iteration_num == o.iteration_num, name, "iteration difference", std::abs(r.iteration_num - o.iteration_num), 0.0);
  checkLabel(ours.hasConverged() == reference.hasConverged(), name, "same convergence flag", ours.hasConverged() == reference.hasConverged() ? 0.0 : 1.0, 0.0);

  const double e_tp = relative(o.transform_probability, r.transform_probability);
  checkLabel(e_tp < 1e-3, name, "transform probability rel. error", e_tp, 1e-3);
  const double e_nvtl = relative(o.nearest_voxel_transformation_likelihood, r.nearest_voxel_transformation_likelihood);
  checkLabel(e_nvtl < 1e-3, name, "NVTL rel. error", e_nvtl, 1e-3);

  checkLabel(r.transformation_array.size() == o.transformation_array.size(), name, "pose array size difference", double(r.transformation_array.size()) - double(o.transformation_array.size()), 0.0);
  // tier4's single grid class leaves its score arrays empty.
  if constexpr (std::is_same_v<RefT, Reference>) {
    const bool same_sizes = r.transform_probability_array.size() == o.transform_probability_array.size() &&
                            r.nearest_voxel_transformation_likelihood_array.size() == o.nearest_voxel_transformation_likelihood_array.size();
    checkLabel(same_sizes, name, "score arrays have the same sizes", same_sizes ? 0.0 : 1.0, 0.0);
    if (same_sizes) {
      double worst_nvtl = 0.0, worst_tp = 0.0;
      for (std::size_t i = 0; i < r.transform_probability_array.size(); i++) {
        worst_tp = std::max(worst_tp, relative(o.transform_probability_array[i], r.transform_probability_array[i]));
        worst_nvtl = std::max(worst_nvtl, relative(o.nearest_voxel_transformation_likelihood_array[i], r.nearest_voxel_transformation_likelihood_array[i]));
      }
      checkLabel(worst_tp < 2e-3, name, "worst transform probability in the arrays", worst_tp, 2e-3);
      checkLabel(worst_nvtl < 2e-3, name, "worst NVTL in the arrays", worst_nvtl, 2e-3);
    }
  }

  // ndt_cuda fixes the sign of a term of H(4, 4), see the README.
  Eigen::Matrix<double, 6, 6> hr = r.hessian, ho = o.hessian;
  hr(4, 4) = ho(4, 4) = 0.0;
  const double e_h = (hr - ho).cwiseAbs().maxCoeff() / hr.cwiseAbs().maxCoeff();
  checkLabel(e_h < 2e-2, name, "hessian rel. error, without H(4,4)", e_h, 2e-2);

  // The scores Autoware computes for the no ground points and for debugging.
  Cloud moved;
  pcl::transformPointCloud(*source, moved, r.pose);
  const double e_calc_tp = relative(ours.calculateTransformationProbability(moved), reference.calculateTransformationProbability(moved));
  checkLabel(e_calc_tp < 1e-3, name, "calculateTransformationProbability() rel. error", e_calc_tp, 1e-3);
  const double e_calc_nvtl = relative(ours.calculateNearestVoxelTransformationLikelihood(moved), reference.calculateNearestVoxelTransformationLikelihood(moved));
  checkLabel(e_calc_nvtl < 1e-3, name, "calculateNearestVoxelTransformationLikelihood() rel. error", e_calc_nvtl, 1e-3);

  // tier4's single grid class does not have this function.
  if constexpr (std::is_same_v<RefT, Reference>) {
    const auto points_ref = reference.calculateNearestVoxelScoreEachPoint(moved);
    const auto points_ours = ours.calculateNearestVoxelScoreEachPoint(moved);
    checkLabel(points_ref.size() == points_ours.size(), name, "scored points difference", double(points_ref.size()) - double(points_ours.size()), 0.0);
    // tier4/ndt_omp returns the points in an order that depends on its threads.
    std::vector<float> scores_ref, scores_ours;
    for (const auto& pt : points_ref) scores_ref.push_back(pt.intensity);
    for (const auto& pt : points_ours) scores_ours.push_back(pt.intensity);
    std::sort(scores_ref.begin(), scores_ref.end());
    std::sort(scores_ours.begin(), scores_ours.end());
    double worst = 0.0;
    for (std::size_t i = 0; i < std::min(scores_ref.size(), scores_ours.size()); i++) {
      worst = std::max(worst, double(std::fabs(scores_ref[i] - scores_ours[i])));
    }
    checkLabel(worst < 1e-3, name, "largest difference of the sorted point scores", worst, 1e-3);
  }
}

static void testMultiGrid(const char* title, const char* name, const Tiles& tiles, const Cloud::Ptr& source) {
  std::printf("\n%s (%zu tiles)\n", title, tiles.size());
  Reference reference;
  Ours ours;
  reference.setParams(autowareParams<pclomp::NdtParams>());
  // tier4's own NdtParams type is in use here, so the fields go over one by one.
  const auto p = reference.getParams();
  ndt_cuda::NdtParams params;
  params.trans_epsilon = p.trans_epsilon;
  params.step_size = p.step_size;
  params.resolution = p.resolution;
  params.max_iterations = p.max_iterations;
  params.num_threads = p.num_threads;
  params.regularization_scale_factor = p.regularization_scale_factor;
  ours.setParams(params);
  for (const auto& [id, tile] : tiles) {
    reference.addTarget(tile, id);
    ours.addTarget(tile, id);
  }
  reference.createVoxelKdtree();
  ours.createVoxelKdtree();

  const auto ids_ref = reference.getCurrentMapIDs();
  const auto ids_ours = ours.getCurrentMapIDs();
  checkLabel(std::is_permutation(ids_ref.begin(), ids_ref.end(), ids_ours.begin(), ids_ours.end()), name, "same map ids", 0.0, 0.0);
  const std::size_t voxels_ref = reference.getVoxelPCD().size();
  const std::size_t voxels_ours = ours.getVoxelPCD().size();
  std::printf("  %s: %zu voxels\n", name, voxels_ref);
  checkLabel(voxels_ref == voxels_ours, name, "voxel count difference", double(voxels_ref) - double(voxels_ours), 0.0);

  char label[128];
  std::snprintf(label, sizeof(label), "%s, identity guess", name);
  compare(label, reference, ours, source, Eigen::Matrix4f::Identity());
  std::snprintf(label, sizeof(label), "%s, perturbed guess", name);
  compare(label, reference, ours, source, perturbedGuess());
}

static ndt_cuda::NdtParams oursAutowareParams() {
  return autowareParams<ndt_cuda::NdtParams>();
}

static void testTileChanges(const Tiles& tiles, const Cloud::Ptr& source) {
  std::printf("\n[4] removing and replacing tiles\n");
  Reference reference;
  Ours ours;
  reference.setParams(autowareParams<pclomp::NdtParams>());
  ours.setParams(oursAutowareParams());
  for (const auto& [id, tile] : tiles) {
    reference.addTarget(tile, id);
    ours.addTarget(tile, id);
  }
  reference.createVoxelKdtree();
  ours.createVoxelKdtree();

  // Removes every third tile and replaces one with an empty part of the map, as the vehicle moving would.
  int i = 0;
  for (const auto& [id, tile] : tiles) {
    (void)tile;
    if (i++ % 3 == 0) {
      reference.removeTarget(id);
      ours.removeTarget(id);
    }
  }
  const auto replaced = std::next(tiles.begin());
  Cloud::Ptr half(new Cloud);
  for (std::size_t k = 0; k < replaced->second->size(); k += 2) half->push_back((*replaced->second)[k]);
  reference.addTarget(half, replaced->first);
  ours.addTarget(half, replaced->first);
  reference.createVoxelKdtree();
  ours.createVoxelKdtree();

  checkLabel(reference.getVoxelPCD().size() == ours.getVoxelPCD().size(), "changed tiles", "voxel count difference", double(reference.getVoxelPCD().size()) - double(ours.getVoxelPCD().size()), 0.0);
  compare("changed tiles", reference, ours, source, Eigen::Matrix4f::Identity());

  // Tiles of different resolutions cannot share the GPU grid.
  Ours mixed;
  mixed.setResolution(2.0f);
  mixed.addTarget(half, "a");
  mixed.setResolution(1.0f);
  mixed.addTarget(half, "b");
  bool threw = false;
  try {
    mixed.createVoxelKdtree();
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  check(threw, "tiles of different resolutions are rejected", threw ? 0.0 : 1.0, 0.0);
}

static void testCopies(const Tiles& tiles, const Cloud::Ptr& source) {
  std::printf("\n[5] copies, as in Autoware's map update\n");
  Ours original;
  original.setParams(oursAutowareParams());
  for (const auto& [id, tile] : tiles) original.addTarget(tile, id);
  original.createVoxelKdtree();
  original.setInputSource(source);
  Cloud out;
  original.align(out);
  const Eigen::Matrix4f expected = original.getFinalTransformation();

  Ours::Ptr secondary(new Ours);
  *secondary = original;
  secondary->setInputSource(source);
  secondary->align(out);
  const float diff = (expected - secondary->getFinalTransformation()).cwiseAbs().maxCoeff();
  check(diff == 0.0f, "assigned copy aligns like the original", diff, 0.0);

  // Changing the copy must leave the original alone.
  secondary->removeTarget(tiles.begin()->first);
  secondary->createVoxelKdtree();
  check(secondary->getVoxelPCD().size() < original.getVoxelPCD().size(), "copy lost the voxels of a removed tile", double(secondary->getVoxelPCD().size()), double(original.getVoxelPCD().size()));
  original.align(out);
  const float unchanged = (expected - original.getFinalTransformation()).cwiseAbs().maxCoeff();
  check(unchanged == 0.0f, "original is unchanged by the copy", unchanged, 0.0);

  Ours constructed(original);
  constructed.setInputSource(source);
  constructed.align(out);
  const float diff_constructed = (expected - constructed.getFinalTransformation()).cwiseAbs().maxCoeff();
  check(diff_constructed == 0.0f, "constructed copy aligns like the original", diff_constructed, 0.0);
}

static void testBackgroundUpdate(const Tiles& tiles, const Cloud::Ptr& source) {
  std::printf("\n[6] updating a copy on another thread while the original aligns\n");
  Ours original;
  original.setParams(oursAutowareParams());
  for (const auto& [id, tile] : tiles) original.addTarget(tile, id);
  original.createVoxelKdtree();
  original.setInputSource(source);
  Cloud out;
  original.align(out);
  const Eigen::Matrix4f expected = original.getFinalTransformation();

  Ours secondary(original);
  std::thread update([&] {
    for (int i = 0; i < 20; i++) {
      secondary.removeTarget(tiles.begin()->first);
      secondary.createVoxelKdtree();
      secondary.addTarget(tiles.begin()->second, tiles.begin()->first);
      secondary.createVoxelKdtree();
    }
  });
  float worst = 0.0f;
  for (int i = 0; i < 20; i++) {
    original.align(out);
    worst = std::max(worst, (expected - original.getFinalTransformation()).cwiseAbs().maxCoeff());
  }
  update.join();
  check(worst == 0.0f, "aligning is not disturbed", worst, 0.0);

  secondary.setInputSource(source);
  secondary.align(out);
  const float diff = (expected - secondary.getFinalTransformation()).cwiseAbs().maxCoeff();
  check(diff == 0.0f, "the updated copy aligns like the original", diff, 0.0);
}

static void testRegularization(const Cloud::Ptr& target, const Cloud::Ptr& source) {
  std::printf("\n[7] regularization\n");
  Reference reference;
  Ours ours;
  reference.setParams(autowareParams<pclomp::NdtParams>());
  ours.setParams(oursAutowareParams());
  reference.setInputTarget(target);
  ours.setInputTarget(target);
  Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
  pose(0, 3) = 0.5f;
  reference.setRegularizationPose(pose);
  ours.setRegularizationPose(pose);
  compare("regularization", reference, ours, source, Eigen::Matrix4f::Identity());

  ours.unsetRegularizationPose();
  reference.unsetRegularizationPose();
  compare("after unset", reference, ours, source, Eigen::Matrix4f::Identity());
}

static void testSingleGrid(const Cloud::Ptr& target, const Cloud::Ptr& source) {
  std::printf("\n[8] single grid class vs tier4's pclomp::NormalDistributionsTransform\n");
  for (const int mode : {pclomp::KDTREE, pclomp::DIRECT7}) {
    ReferenceSingle reference;
    OursSingle ours;
    pclomp::NdtParams params = reference.getParams();
    params.resolution = 2.0;
    params.trans_epsilon = 0.01;
    params.search_method = static_cast<pclomp::NeighborSearchMethod>(mode);
    params.use_line_search = false;
    reference.setParams(params);
    ndt_cuda::NdtParams ours_params = oursAutowareParams();
    ours_params.max_iterations = params.max_iterations;
    ours_params.search_method = static_cast<ndt_cuda::NeighborSearchMethod>(mode);
    ours.setParams(ours_params);
    reference.setInputTarget(target);
    ours.setInputTarget(target);
    compare(mode == pclomp::KDTREE ? "single KDTREE" : "single DIRECT7", reference, ours, source, perturbedGuess());
  }
}

int main(int argc, char** argv) {
  const std::string data_dir = argc > 1 ? argv[1] : NDT_CUDA_DATA_DIR;
  if (ndt_cuda::GpuBackend::deviceCount() <= 0) {
    std::printf("no CUDA device found, skipping\n");
    return 77;
  }
  std::printf("GPU: %s\n", ndt_cuda::GpuBackend::deviceName(0).c_str());

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

  Tiles one;
  one["default"] = target;
  testMultiGrid("[1] the map as one tile", "one tile", one, source);
  const Tiles aligned_tiles = splitIntoTiles(*target, 20.0f);
  testMultiGrid("[2] tile borders on voxel borders", "20 m tiles", aligned_tiles, source);
  const Tiles shared_cells = splitIntoTiles(*target, 7.0f);
  testMultiGrid("[3] tiles that share voxels", "7 m tiles", shared_cells, source);
  testTileChanges(aligned_tiles, source);
  testCopies(aligned_tiles, source);
  testBackgroundUpdate(aligned_tiles, source);
  testRegularization(target, source);
  testSingleGrid(target, source);

  std::printf("\n%s (%d failure%s)\n", g_failures ? "FAILED" : "PASSED", g_failures, g_failures == 1 ? "" : "s");
  return g_failures ? 1 : 0;
}
