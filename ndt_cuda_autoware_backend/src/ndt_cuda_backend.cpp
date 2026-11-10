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

// The NDT backend "ndt_cuda": ndt_cuda behind the NdtBackend interface of
// autoware_ndt_scan_matcher_common.

#include <autoware/ndt_scan_matcher_common/ndt_backend.hpp>
#include <ndt_cuda/multigrid_ndt_cuda.hpp>
#include <pluginlib/class_list_macros.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ndt_cuda_autoware_backend
{

using autoware::ndt_scan_matcher::NdtBackend;
using autoware::ndt_scan_matcher::NdtParams;
using autoware::ndt_scan_matcher::NdtResult;

namespace
{

ndt_cuda::NdtParams to_cuda_params(const NdtParams & params)
{
  ndt_cuda::NdtParams cuda_params{};
  cuda_params.trans_epsilon = params.trans_epsilon;
  cuda_params.step_size = params.step_size;
  cuda_params.resolution = params.resolution;
  cuda_params.max_iterations = params.max_iterations;
  // The search and the step of ndt_omp in Autoware: all voxels whose mean is within the
  // resolution of a point, and no line search.
  cuda_params.search_method = ndt_cuda::KDTREE;
  cuda_params.use_line_search = false;
  // ndt_cuda keeps the value, but it has no effect on the GPU.
  cuda_params.num_threads = params.num_threads;
  cuda_params.regularization_scale_factor = params.regularization_scale_factor;
  return cuda_params;
}

NdtResult from_cuda_result(ndt_cuda::NdtResult && cuda_result)
{
  NdtResult result;
  result.pose = cuda_result.pose;
  result.transform_probability = cuda_result.transform_probability;
  result.nearest_voxel_transformation_likelihood =
    cuda_result.nearest_voxel_transformation_likelihood;
  result.iteration_num = cuda_result.iteration_num;
  result.transformation_array = std::move(cuda_result.transformation_array);
  result.hessian = cuda_result.hessian;
  result.transform_probability_array = std::move(cuda_result.transform_probability_array);
  result.nearest_voxel_transformation_likelihood_array =
    std::move(cuda_result.nearest_voxel_transformation_likelihood_array);
  return result;
}

}  // namespace

class NdtCudaBackend : public NdtBackend
{
  using Ndt = ndt_cuda::MultiGridNormalDistributionsTransformCUDA<PointSource, PointTarget>;

public:
  // pluginlib needs a default constructor that does not use the GPU. initialize() gets the GPU.
  NdtCudaBackend() = default;

  // The copy of ndt_cuda shares the map that is already on the GPU, and it gets its own CUDA
  // stream and work buffers.
  NdtCudaBackend(const NdtCudaBackend & other)
  : NdtBackend(other),
    params_(other.params_),
    ndt_(other.ndt_ ? std::make_unique<Ndt>(*other.ndt_) : nullptr)
  {
  }

  void initialize(const NdtParams & params) override
  {
    params_ = params;
    // Throws std::runtime_error if the computer has no CUDA device.
    ndt_ = std::make_unique<Ndt>();
    // The voxel covariance of the ndt_omp copy in Autoware.
    ndt_->setUseAutowareCovariance(true);
    ndt_->setParams(to_cuda_params(params));
  }

  std::string name() const override { return "ndt_cuda"; }

  NdtParams get_params() const override { return params_; }

  // Builds the voxels of the cloud on the CPU.
  void add_target(
    const pcl::PointCloud<PointTarget>::ConstPtr & cloud, const std::string & map_id) override
  {
    ndt_->addTarget(cloud, map_id);
  }

  void remove_target(const std::string & map_id) override { ndt_->removeTarget(map_id); }

  // Merges the voxels of all clouds and copies them to the GPU.
  void commit_target() override { ndt_->createVoxelKdtree(); }

  std::vector<std::string> get_current_map_ids() const override
  {
    return ndt_->getCurrentMapIDs();
  }

  bool has_target() const override { return !ndt_->getCurrentMapIDs().empty(); }

  SharedPtr clone() const override { return std::make_shared<NdtCudaBackend>(*this); }

  SharedPtr clone_empty() const override
  {
    auto empty = std::make_shared<NdtCudaBackend>();
    empty->params_ = params_;
    empty->ndt_ = std::make_unique<Ndt>(ndt_->getDevice());
    empty->ndt_->setUseAutowareCovariance(true);
    empty->ndt_->setParams(to_cuda_params(params_));
    return empty;
  }

  void set_regularization_pose(const Eigen::Matrix4f & pose) override
  {
    ndt_->setRegularizationPose(pose);
  }

  void unset_regularization_pose() override { ndt_->unsetRegularizationPose(); }

  NdtResult align(
    const Eigen::Matrix4f & guess, const pcl::PointCloud<PointSource>::ConstPtr & source) override
  {
    // pcl::Registration needs a cloud for the aligned source. The node does not use it.
    pcl::PointCloud<PointSource> aligned_source;
    ndt_->setInputSource(source);
    ndt_->align(aligned_source, guess);
    return from_cuda_result(ndt_->getResult());
  }

  double calculate_transformation_probability(
    const pcl::PointCloud<PointSource> & cloud_in_map) const override
  {
    return ndt_->calculateTransformationProbability(cloud_in_map);
  }

  double calculate_nearest_voxel_transformation_likelihood(
    const pcl::PointCloud<PointSource> & cloud_in_map) const override
  {
    return ndt_->calculateNearestVoxelTransformationLikelihood(cloud_in_map);
  }

  pcl::PointCloud<pcl::PointXYZI> calculate_nearest_voxel_score_each_point(
    const pcl::PointCloud<PointSource> & cloud_in_map) const override
  {
    return ndt_->calculateNearestVoxelScoreEachPoint(cloud_in_map);
  }

private:
  NdtParams params_{};
  // Null until initialize(). The node calls no other function before initialize().
  std::unique_ptr<Ndt> ndt_;
};

}  // namespace ndt_cuda_autoware_backend

PLUGINLIB_EXPORT_CLASS(
  ndt_cuda_autoware_backend::NdtCudaBackend, autoware::ndt_scan_matcher::NdtBackend)
