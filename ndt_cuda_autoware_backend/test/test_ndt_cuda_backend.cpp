// Copyright 2026 Autoware Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The backend "ndt_cuda" must work as the interface requires, and its results must agree with the
// backend "ndt_omp" of autoware_ndt_scan_matcher. The tests need a CUDA device. Without one, they are
// skipped.

#include <Eigen/Geometry>

#include <autoware/ndt_scan_matcher_common/ndt_backend.hpp>
#include <pluginlib/class_loader.hpp>

#include <gtest/gtest.h>
#include <pcl/common/transforms.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace autoware::ndt_scan_matcher
{
namespace
{

NdtParams make_params()
{
  NdtParams params{};
  params.trans_epsilon = 0.01;
  params.step_size = 0.1;
  params.resolution = 2.0F;
  params.max_iterations = 30;
  params.num_threads = 1;
  params.regularization_scale_factor = 0.01F;
  return params;
}

// Three walls that meet in a corner, 20 m long, with a small deterministic roughness.
pcl::PointCloud<pcl::PointXYZ>::Ptr make_map()
{
  auto map = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  const float step = 0.2F;
  const int count = 100;
  for (int i = 0; i < count; ++i) {
    for (int j = 0; j < count; ++j) {
      const float a = step * static_cast<float>(i);
      const float b = step * static_cast<float>(j);
      const float rough = 0.05F * std::sin(1.7F * a) * std::cos(2.3F * b);
      map->emplace_back(a, b, rough);
      map->emplace_back(a, rough, b);
      map->emplace_back(rough, a, b);
    }
  }
  return map;
}

// Every 10th point of the map, as the scan.
pcl::PointCloud<pcl::PointXYZ>::Ptr make_scan()
{
  const auto map = make_map();
  auto scan = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  for (std::size_t i = 0; i < map->size(); i += 10) {
    scan->push_back(map->points[i]);
  }
  return scan;
}

Eigen::Matrix4f make_guess()
{
  Eigen::Affine3f guess = Eigen::Affine3f::Identity();
  guess.translate(Eigen::Vector3f(0.3F, -0.2F, 0.1F));
  guess.rotate(Eigen::AngleAxisf(0.02F, Eigen::Vector3f::UnitZ()));
  return guess.matrix();
}

// The node never unloads a backend library, so the loader of the tests lives until the process
// ends.
pluginlib::ClassLoader<NdtBackend> & loader()
{
  static auto * const instance = new pluginlib::ClassLoader<NdtBackend>(
    "autoware_ndt_scan_matcher_common", "autoware::ndt_scan_matcher::NdtBackend");
  return *instance;
}

// A backend from pluginlib, initialized as the node does it. Null if the computer has no GPU.
NdtBackend::SharedPtr make_backend(const std::string & name, const NdtParams & params)
{
  NdtBackend::SharedPtr backend = loader().createSharedInstance(name);
  try {
    backend->initialize(params);
  } catch (const std::runtime_error & error) {
    std::cerr << "initialize() of '" << name << "' failed: " << error.what() << std::endl;
    return nullptr;
  }
  return backend;
}

NdtBackend::SharedPtr make_backend_with_map(const std::string & name)
{
  NdtBackend::SharedPtr backend = make_backend(name, make_params());
  if (backend) {
    backend->add_target(make_map(), "a");
    backend->commit_target();
  }
  return backend;
}

#define SKIP_WITHOUT_GPU(backend) \
  if (!(backend)) GTEST_SKIP() << "No CUDA device."

}  // namespace

TEST(NdtCudaBackend, PluginlibLoadsTheBackend)
{
  EXPECT_TRUE(loader().isClassAvailable("ndt_cuda"));
  const NdtBackend::SharedPtr backend = make_backend("ndt_cuda", make_params());
  SKIP_WITHOUT_GPU(backend);
  EXPECT_EQ(backend->name(), "ndt_cuda");
  EXPECT_FALSE(backend->has_target());
  EXPECT_TRUE(backend->get_current_map_ids().empty());
}

TEST(NdtCudaBackend, AlignAgreesWithTheOmpBackend)
{
  const NdtBackend::SharedPtr cuda = make_backend_with_map("ndt_cuda");
  SKIP_WITHOUT_GPU(cuda);
  const NdtBackend::SharedPtr omp = make_backend_with_map("ndt_omp");
  ASSERT_NE(omp, nullptr);

  for (const bool regularization : {false, true}) {
    SCOPED_TRACE(regularization ? "with regularization" : "without regularization");
    if (regularization) {
      cuda->set_regularization_pose(Eigen::Matrix4f::Identity());
      omp->set_regularization_pose(Eigen::Matrix4f::Identity());
    }
    const NdtResult result = cuda->align(make_guess(), make_scan());
    const NdtResult expected = omp->align(make_guess(), make_scan());

    EXPECT_LT((result.pose - expected.pose).cwiseAbs().maxCoeff(), 1e-3F);
    EXPECT_EQ(result.iteration_num, expected.iteration_num);
    EXPECT_NEAR(result.transform_probability, expected.transform_probability, 1e-3);
    EXPECT_NEAR(
      result.nearest_voxel_transformation_likelihood,
      expected.nearest_voxel_transformation_likelihood, 1e-3);
    EXPECT_LT(
      (result.hessian - expected.hessian).norm(), 1e-2 * expected.hessian.norm());

    // What the node needs from the arrays (see NdtResult).
    const auto size = static_cast<std::size_t>(result.iteration_num) + 1;
    ASSERT_EQ(result.transformation_array.size(), size);
    EXPECT_EQ(result.transformation_array.front(), make_guess());
    EXPECT_EQ(result.transformation_array.back(), result.pose);
    ASSERT_EQ(result.transform_probability_array.size(), size);
    ASSERT_EQ(result.nearest_voxel_transformation_likelihood_array.size(), size);
    EXPECT_EQ(result.transform_probability_array.back(), result.transform_probability);
    EXPECT_EQ(
      result.nearest_voxel_transformation_likelihood_array.back(),
      result.nearest_voxel_transformation_likelihood);
    EXPECT_NEAR(
      result.transform_probability_array.front(), expected.transform_probability_array.front(),
      1e-3);
    EXPECT_NEAR(
      result.nearest_voxel_transformation_likelihood_array.front(),
      expected.nearest_voxel_transformation_likelihood_array.front(), 1e-3);
  }
}

TEST(NdtCudaBackend, ScoreFunctionsAgreeWithTheOmpBackend)
{
  const NdtBackend::SharedPtr cuda = make_backend_with_map("ndt_cuda");
  SKIP_WITHOUT_GPU(cuda);
  const NdtBackend::SharedPtr omp = make_backend_with_map("ndt_omp");
  ASSERT_NE(omp, nullptr);

  // Some points of this cloud have no voxel near them.
  pcl::PointCloud<pcl::PointXYZ> cloud_in_map;
  Eigen::Affine3f shift = Eigen::Affine3f::Identity();
  shift.translate(Eigen::Vector3f(1.5F, 0.4F, -0.3F));
  pcl::transformPointCloud(*make_scan(), cloud_in_map, shift.matrix());

  EXPECT_NEAR(
    cuda->calculate_transformation_probability(cloud_in_map),
    omp->calculate_transformation_probability(cloud_in_map), 1e-3);
  EXPECT_NEAR(
    cuda->calculate_nearest_voxel_transformation_likelihood(cloud_in_map),
    omp->calculate_nearest_voxel_transformation_likelihood(cloud_in_map), 1e-3);

  const auto scores = cuda->calculate_nearest_voxel_score_each_point(cloud_in_map);
  const auto expected_scores = omp->calculate_nearest_voxel_score_each_point(cloud_in_map);
  EXPECT_LT(scores.size(), cloud_in_map.size());
  ASSERT_EQ(scores.size(), expected_scores.size());
  double sum = 0.0;
  double expected_sum = 0.0;
  for (std::size_t i = 0; i < scores.size(); ++i) {
    sum += scores[i].intensity;
    expected_sum += expected_scores[i].intensity;
  }
  EXPECT_NEAR(sum, expected_sum, 1e-3 * expected_sum);
}

TEST(NdtCudaBackend, CloneIsIndependent)
{
  const NdtBackend::SharedPtr backend = make_backend_with_map("ndt_cuda");
  SKIP_WITHOUT_GPU(backend);
  backend->set_regularization_pose(Eigen::Matrix4f::Identity());
  const NdtResult expected = backend->align(make_guess(), make_scan());

  const NdtBackend::SharedPtr clone = backend->clone();
  EXPECT_EQ(clone->name(), "ndt_cuda");
  EXPECT_EQ(clone->get_current_map_ids(), std::vector<std::string>{"a"});
  // The same map and the same regularization pose give the same result.
  const NdtResult result = clone->align(make_guess(), make_scan());
  EXPECT_EQ(result.pose, expected.pose);
  EXPECT_EQ(result.transform_probability, expected.transform_probability);

  // A change of the clone does not change the original, and the reverse.
  clone->remove_target("a");
  clone->commit_target();
  clone->unset_regularization_pose();
  EXPECT_FALSE(clone->has_target());
  EXPECT_TRUE(backend->has_target());
  EXPECT_EQ(backend->align(make_guess(), make_scan()).pose, expected.pose);
}

TEST(NdtCudaBackend, CloneOutlivesTheOriginal)
{
  NdtBackend::SharedPtr backend = make_backend_with_map("ndt_cuda");
  SKIP_WITHOUT_GPU(backend);
  const NdtResult expected = backend->align(make_guess(), make_scan());
  const NdtBackend::SharedPtr clone = backend->clone();
  backend.reset();
  EXPECT_EQ(clone->align(make_guess(), make_scan()).pose, expected.pose);
}

TEST(NdtCudaBackend, CloneEmptyHasTheParametersAndNoMap)
{
  NdtParams params = make_params();
  params.max_iterations = 17;
  const NdtBackend::SharedPtr backend = make_backend("ndt_cuda", params);
  SKIP_WITHOUT_GPU(backend);
  backend->add_target(make_map(), "a");
  backend->commit_target();
  backend->set_regularization_pose(Eigen::Matrix4f::Identity());

  const NdtBackend::SharedPtr empty = backend->clone_empty();
  EXPECT_FALSE(empty->has_target());
  EXPECT_EQ(empty->name(), "ndt_cuda");
  const NdtParams empty_params = empty->get_params();
  EXPECT_EQ(empty_params.trans_epsilon, params.trans_epsilon);
  EXPECT_EQ(empty_params.step_size, params.step_size);
  EXPECT_EQ(empty_params.resolution, params.resolution);
  EXPECT_EQ(empty_params.max_iterations, 17);
  EXPECT_EQ(empty_params.num_threads, params.num_threads);
  EXPECT_EQ(empty_params.regularization_scale_factor, params.regularization_scale_factor);

  // The map update module of the node fills the empty instance and then uses it.
  empty->add_target(make_map(), "b");
  empty->commit_target();
  EXPECT_EQ(empty->get_current_map_ids(), std::vector<std::string>{"b"});
  const NdtBackend::SharedPtr reference = make_backend("ndt_cuda", params);
  reference->add_target(make_map(), "a");
  reference->commit_target();
  // The empty instance has no regularization pose, as the reference.
  EXPECT_EQ(
    empty->align(make_guess(), make_scan()).pose,
    reference->align(make_guess(), make_scan()).pose);
}

// The node replaces a part of the map while the active instance aligns in another thread.
TEST(NdtCudaBackend, MapUpdateOfAnInstanceInAnotherThread)
{
  const NdtBackend::SharedPtr active = make_backend_with_map("ndt_cuda");
  SKIP_WITHOUT_GPU(active);
  const NdtResult expected = active->align(make_guess(), make_scan());

  NdtBackend::SharedPtr updated;
  std::thread map_update([&]() {
    const NdtBackend::SharedPtr building = active->clone_empty();
    building->add_target(make_map(), "b");
    building->commit_target();
    updated = building->clone();
  });
  for (int i = 0; i < 20; ++i) {
    EXPECT_EQ(active->align(make_guess(), make_scan()).pose, expected.pose);
  }
  map_update.join();
  EXPECT_EQ(updated->align(make_guess(), make_scan()).pose, expected.pose);
}

}  // namespace autoware::ndt_scan_matcher
