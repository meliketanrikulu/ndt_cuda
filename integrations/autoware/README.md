# Using ndt_cuda in Autoware

`apply.sh` makes Autoware's `autoware_ndt_scan_matcher` use ndt_cuda on the GPU. The node, its topics, parameters and launch files stay the same. It was tested with Autoware 1.5.0; see the [Autoware benchmark](../../docs/autoware_benchmark.md).

```bash
cd ~/autoware/src
git clone https://github.com/meliketanrikulu/ndt_cuda.git
bash ndt_cuda/integrations/autoware/apply.sh core/autoware_core/localization/autoware_ndt_scan_matcher
cd ~/autoware
colcon build --packages-up-to autoware_ndt_scan_matcher --cmake-args -DCMAKE_BUILD_TYPE=Release
```

The CUDA toolkit is needed to build ndt_cuda; the Autoware CUDA images have it under `/usr/local/cuda`, which is not always on `PATH`.

## What it changes

| File | Change |
| --- | --- |
| `include/autoware/ndt_scan_matcher/ndt_omp/multigrid_ndt_omp.h` | Replaced: `pclomp::MultiGridNormalDistributionsTransform` becomes ndt_cuda's `MultiGridNormalDistributionsTransformCUDA`, with Autoware's covariance formula |
| `include/autoware/ndt_scan_matcher/ndt_omp/ndt_struct.hpp` | Replaced: `pclomp::NdtParams`, `NdtResult` and the search methods become ndt_cuda's |
| `src/ndt_omp/multigrid_*`, `multi_voxel_grid_covariance_omp*` | Removed |
| `CMakeLists.txt`, `package.xml` | Depend on ndt_cuda |
| `src/ndt_scan_matcher_core.cpp` | Two calls get the `pclomp::` prefix, which argument dependent lookup no longer supplies |

To go back to the CPU, restore the package from git (`git checkout -- .` in `autoware_core`).

## Differences from Autoware's NDT

- `ndt.num_threads` has no effect on the alignment, which runs on the GPU.
- Map tiles are merged and uploaded to the GPU again at each map update. Autoware loads tiles in a background thread, so this does not delay the alignment.
- The results are not bit-identical: the GPU computes in single precision, and ndt_cuda corrects the sign of one hessian term. See the README of ndt_cuda.
- The node needs an NVIDIA GPU; without one it stops at startup.
