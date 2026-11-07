# ndt_cuda

CUDA accelerated **Normal Distributions Transform (NDT)** point cloud registration for C++17, CUDA 12 and ROS 2.

`ndt_cuda::NormalDistributionsTransformCUDA` is a GPU port of the NDT of the [Point Cloud Library](https://pointclouds.org) and of its OpenMP variant [koide3/ndt_omp](https://github.com/koide3/ndt_omp). It keeps ndt_omp's interface, parameters, neighbor search methods and optimizer, so existing ndt_omp code can switch to the GPU with a one-line change.

> **Status (v0.1.0):** validated against ndt_omp on real hardware (NVIDIA GeForce RTX 3050 Ti Laptop GPU, see [Validation](#validation) and [Benchmark](#benchmark)) and, in CI, by a test that runs the exact kernel arithmetic on the CPU. Other GPUs and Jetson Orin are compiled for but not yet tested; benchmark results for them are welcome.

## Features

- **Same API as `pclomp::NormalDistributionsTransform`** — derives from `pcl::Registration`, same setters/getters and defaults, accepts `pclomp::NeighborSearchMethod`. A compatibility header provides `pclomp::NormalDistributionsTransformCUDA`.
- **Same results** — the voxel grid is bit-identical to ndt_omp's; score, gradient and hessian match to ~1e-6 relative error; `align()` results match to the micrometer on the sample data.
- **All four neighbor search methods on the GPU** — `KDTREE` (exact radius search), `DIRECT26`, `DIRECT7`, `DIRECT1`.
- **Deterministic** — double precision reduction with a fixed reduction tree: repeated runs on the same GPU give identical results.
- **Modern packaging** — C++17, CUDA 11.4–12.x, default architectures Turing → Hopper incl. Jetson Orin; `find_package(ndt_cuda)` + `ndt_cuda::ndt_cuda` in plain CMake and ROS 2 (Humble, Jazzy); the CUDA runtime is linked statically (only the NVIDIA driver is needed at runtime).
- **GPU-less validation** — the test suite runs without a GPU or CUDA toolkit, so it also runs in CI.

## Requirements

| | Version |
|---|---|
| CMake | ≥ 3.18 |
| C++ compiler | C++17 (GCC ≥ 9, Clang ≥ 10) |
| CUDA toolkit | ≥ 11.4 (12.x recommended; ≥ 11.8 for Ada / Hopper native code) |
| PCL | ≥ 1.11 (Ubuntu 22.04: 1.12, Ubuntu 24.04: 1.14) |
| Eigen | 3.3+ |
| ROS 2 (optional) | Humble, Jazzy (any ament_cmake distro should work) |

## Build

### ROS 2

```bash
cd ~/ros2_ws/src
git clone https://github.com/meliketanrikulu/ndt_cuda.git
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -y   # PCL, Eigen (install the CUDA toolkit separately)
colcon build --packages-select ndt_cuda --cmake-args -DCMAKE_BUILD_TYPE=Release
```

In your package:

```xml
<depend>ndt_cuda</depend>
```
```cmake
find_package(ndt_cuda REQUIRED)
target_link_libraries(your_node ndt_cuda::ndt_cuda)
```

### Plain CMake

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
sudo cmake --install build        # optional
```

```cmake
find_package(ndt_cuda REQUIRED)
target_link_libraries(your_target ndt_cuda::ndt_cuda)
```

### CMake options

| Option | Default | |
|---|---|---|
| `CMAKE_CUDA_ARCHITECTURES` | `75;86;87;89;90` (`75;86;87` with CUDA < 11.8) | Target GPUs. Each entry also embeds PTX, so newer GPUs JIT compile. Use `native` (CMake ≥ 3.24) to build only for the installed GPU. |
| `NDT_CUDA_BUILD_EXAMPLES` | `ON` | `ndt_cuda_align` example |
| `NDT_CUDA_BUILD_TESTS` | `OFF` | Validation test (downloads koide3/ndt_omp as reference) |
| `NDT_CUDA_BUILD_GPU_TESTS` | `OFF` | Test of the real CUDA kernels, skipped when no GPU is available (downloads koide3/ndt_omp as reference) |
| `NDT_CUDA_BUILD_BENCHMARK` | `OFF` | ndt_omp vs ndt_cuda benchmark (downloads koide3/ndt_omp) |
| `NDT_OMP_SOURCE_DIR` | — | Use a local ndt_omp checkout instead of downloading it |
| `NDT_CUDA_ENABLE_CUDA` | `ON` | With `ON`, configuring fails if no CUDA compiler is found. `OFF` builds only the GPU-less tests (no CUDA toolkit needed) |

## Usage

```cpp
#include <ndt_cuda/ndt_cuda.hpp>

ndt_cuda::NormalDistributionsTransformCUDA<pcl::PointXYZ, pcl::PointXYZ> ndt;  // optional: CUDA device id
ndt.setResolution(1.0);
ndt.setNeighborhoodSearchMethod(ndt_cuda::DIRECT7);
ndt.setInputTarget(map);          // builds the voxel grid and uploads it to the GPU
ndt.setInputSource(scan);

pcl::PointCloud<pcl::PointXYZ> aligned;
ndt.align(aligned, initial_guess);
Eigen::Matrix4f pose = ndt.getFinalTransformation();
```

Instantiated for `pcl::PointXYZ`, `pcl::PointXYZI` and `pcl::PointXYZRGB`. For other point types, include `<ndt_cuda/impl/ndt_cuda.hpp>` and `<ndt_cuda/impl/voxel_grid_covariance.hpp>` in one translation unit and instantiate the templates there.

### Migrating from ndt_omp

Either replace the type:

```cpp
// pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr ndt(new pclomp::NormalDistributionsTransform<PointT, PointT>());
ndt_cuda::NormalDistributionsTransformCUDA<PointT, PointT>::Ptr ndt(new ndt_cuda::NormalDistributionsTransformCUDA<PointT, PointT>());
ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);   // ndt_omp's enum is accepted
```

or use the compatibility header, which provides `pclomp::NormalDistributionsTransformCUDA` (and, when ndt_omp is not used, `pclomp::NeighborSearchMethod` / `pclomp::DIRECT7` / ...):

```cpp
#include <pclomp/ndt_omp.h>             // only if you still use ndt_omp; must come first
#include <ndt_cuda/pclomp_compat.hpp>

pclomp::NormalDistributionsTransformCUDA<PointT, PointT>::Ptr ndt(new pclomp::NormalDistributionsTransformCUDA<PointT, PointT>());
```

Code that holds the registration as `pcl::Registration<PointT, PointT>::Ptr` (e.g. hdl_localization-style code) only needs to construct the new type.

### Neighbor search methods

| Method | Voxels used per point | Notes |
|---|---|---|
| `KDTREE` | all voxels whose mean is within `resolution` | Same neighbor set as ndt_omp's FLANN radius search, evaluated exactly on the GPU |
| `DIRECT26` | the 26 surrounding voxels (not the containing one) | As in ndt_omp |
| `DIRECT7` | containing voxel + 6 face neighbors | **Recommended** (default) |
| `DIRECT1` | containing voxel only | Fastest, least robust |

## How it works

| Step | Where |
|---|---|
| Voxel grid (mean / inverse covariance per voxel), when the target or resolution changes | CPU (same arithmetic as ndt_omp), then uploaded once as an open-addressing hash table |
| Every score / gradient / hessian evaluation: transform all source points, look up neighbor voxels, accumulate Eq. 6.9–6.13 [Magnusson 2009] | **GPU** — one thread per point, per-block warp-shuffle reduction, fixed-order final reduction in double precision |
| 6×6 Newton solve and More-Thuente line search | CPU (unchanged from PCL / ndt_omp) |

The per-point math lives in [`src/ndt_cuda_kernels.cuh`](src/ndt_cuda_kernels.cuh) as `__host__ __device__` functions, shared by the kernels and by the CPU emulation used in the tests.

## Differences from ndt_omp

- **Hessian sign fix.** ndt_omp's float code path uses `+sin(pitch)` in row `d1` of the angular hessian (`h_ang.row(6)`), while its double path (`h_ang_d1_`) and the derivation give `-sin(pitch)`. ndt_cuda uses `-sin(pitch)`; the test verifies this against finite differences. Only the (pitch, pitch) hessian entry is affected — the optimum is the same, individual Newton steps can differ slightly.
- `setNumThreads()` exists for compatibility and has no effect.
- Optional kd-tree on the map. The alignment does not use it, but `pcl::Registration` builds it in the first `align()` after a target change, because `getFitnessScore()` needs it. This is on by default, as in ndt_omp. If you never call `getFitnessScore()`, `setBuildTargetKdTree(false)` makes that first `align()` much faster on large maps (see the [benchmark report](docs/benchmark.md#map-updates)); while it is off, `getFitnessScore()` must not be used.
- `setResolution()` after `setInputTarget()` rebuilds the voxel grid right away. ndt_omp rebuilds it there only if a source cloud is already set, and otherwise keeps the old resolution.
- Non-finite target points are always skipped, even if the cloud is wrongly marked `is_dense`.
- `align()` reports no convergence if no voxel is available on the GPU: the map has no usable voxel, or an earlier `setInputTarget()` threw (for example, out of GPU memory). Set the target again to recover.
- Not copyable, because an instance owns its GPU state. Use a separate instance or a pointer.
- With several GPUs, an instance on a non-current device restores the thread's current device after each call, which can also create a CUDA context there. Use `CUDA_VISIBLE_DEVICES` to keep a process off the other GPUs completely. Only single-GPU operation has been tested.
- Faithful to ndt_omp (kept on purpose so that both give the same results): the covariance accumulator starts from the identity, which adds a small `I·(n−1)/n²` regularization; `DIRECT26` excludes the voxel that contains the point.

## Validation

`test/ndt_cuda_test.cpp` checks the GPU implementation **without a GPU**: it links `NormalDistributionsTransformCUDA` against a CPU emulation of the backend ([`test/gpu_backend_emulated.cpp`](test/gpu_backend_emulated.cpp)) that runs the exact kernel code and reproduces the GPU data layout. koide3/ndt_omp is downloaded at a pinned commit and used as reference.

| Check | Result on the sample data |
|---|---|
| Voxel grid vs ndt_omp (resolution 0.5 / 1 / 2 m) | identical (max difference 0) |
| Score / gradient / hessian vs ndt_omp, 4 search methods, 3 poses | ≤ 1e-5 relative error |
| Gradient / hessian vs central finite differences | ≤ 3e-4 relative error |
| `align()` vs ndt_omp, 4 search methods, identity and perturbed initial guess | ≤ 3e-6 m, ≤ 0.01° |
| Map shifted by 47 km, scan in sensor frame | same result as unshifted (mm level, as ndt_omp) |
| `pcl::Registration::Ptr` usage, `pclomp_compat.hpp` with and without ndt_omp | compiles and converges |
| `getFitnessScore()` vs ndt_omp, also through `pcl::Registration*` | ≤ 1e-6 relative error |

```bash
cmake -B build -DNDT_CUDA_BUILD_TESTS=ON && cmake --build build -j && (cd build && ctest --output-on-failure)
# ROS 2:
colcon build --packages-select ndt_cuda --cmake-args -DNDT_CUDA_BUILD_TESTS=ON && colcon test --packages-select ndt_cuda
```

`test/ndt_cuda_gpu_test.cpp` runs the real CUDA kernels and reductions, which the test above and CI do not execute: `align()` against ndt_omp for all search methods (also with 300000 source points) and repeated runs being bit-identical. Run it on a machine with a GPU:

```bash
cmake -B build -DNDT_CUDA_BUILD_GPU_TESTS=ON && cmake --build build -j && (cd build && ctest --output-on-failure -R gpu)
```

## Benchmark

```bash
cmake -B build -DNDT_CUDA_BUILD_BENCHMARK=ON && cmake --build build -j
./build/ndt_cuda_benchmark                              # data/*.pcd, all search methods
./build/ndt_cuda_benchmark target.pcd source.pcd 2.0    # your own clouds and resolution
```

It runs ndt_omp (all CPU threads) and ndt_cuda on the same clouds and prints the time per alignment, iterations, transformation probability and the difference between the two results, and exits with a non-zero code if the results disagree.

Average time per `align()` on the sample data (about 16000 source points, resolution 1 m), NVIDIA GeForce RTX 3050 Ti Laptop GPU against ndt_omp with 16 threads on an Intel Core i7-11800H:

| Method | ndt_omp | ndt_cuda | Speedup |
|---|---|---|---|
| `KDTREE` | 20.08 ms | 0.84 ms | ×24 |
| `DIRECT26` | 60.23 ms | 0.61 ms | ×98 |
| `DIRECT7` | 14.54 ms | 0.68 ms | ×21 |
| `DIRECT1` | 6.86 ms | 0.63 ms | ×11 |

These numbers come from one machine and one pair of small scans; run the benchmark on your own hardware and data. See the [benchmark report](docs/benchmark.md) for repeated measurements, large clouds, map updates and limitations.

## Roadmap

- Benchmark results on other desktop GPUs and Jetson Orin
- Voxel grid construction on the GPU (faster map updates)
- Dynamic map loading (add / remove voxels incrementally)
- ROS 2 component node (map + scan → pose)

## Acknowledgements and license

ndt_cuda is released under the [BSD 3-Clause License](LICENSE). It is a derivative work of:

- **PCL** `NormalDistributionsTransform` and `VoxelGridCovariance` — BSD 3-Clause, © Willow Garage, Inc. and Open Perception, Inc.; NDT implementation by Brian Okorn, math refactored by Todor Stoyanov.
- **[koide3/ndt_omp](https://github.com/koide3/ndt_omp)** — BSD 2-Clause, © 2019 Kenji Koide. The API, the neighbor search methods, the float derivative formulation and the sample data come from ndt_omp.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the full notices and the list of derived files. Algorithm references:

- M. Magnusson, *The Three-Dimensional Normal-Distributions Transform*, PhD thesis, Örebro University, 2009.
- J. J. Moré and D. J. Thuente, *Line Search Algorithms with Guaranteed Sufficient Decrease*, ACM TOMS, 1994.
