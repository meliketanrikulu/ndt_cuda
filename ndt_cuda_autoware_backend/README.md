# ndt_cuda_autoware_backend

## Purpose

This package is the NDT backend `ndt_cuda` for [autoware_ndt_scan_matcher](https://github.com/autowarefoundation/autoware_core/tree/main/localization/autoware_ndt_scan_matcher). It runs the NDT scan matching on an NVIDIA GPU with [ndt_cuda](../README.md), the library of this repository. The node loads the backend with pluginlib, so no file of the node changes.

The backend is a wrapper of `ndt_cuda::MultiGridNormalDistributionsTransformCUDA` behind the interface `autoware::ndt_scan_matcher::NdtBackend` of `autoware_ndt_scan_matcher_common`.

This package is a prototype. It needs the NDT backend interface of Autoware Core, which is not merged yet.

## Requirements

- An NVIDIA GPU and the CUDA toolkit, version 11.4 or later.
- Autoware Core with the package `autoware_ndt_scan_matcher_common`.

## Build

Put this repository in the `src` directory of a workspace that has Autoware Core, or that has Autoware Core as an underlay. Then build the two packages of this repository:

```bash
colcon build --packages-up-to ndt_cuda_autoware_backend --cmake-args -DCMAKE_BUILD_TYPE=Release
```

Do not add compiler options that change the alignment of Eigen (for example `-mavx`) to only this package. The node then refuses to load the backend.

## Use

Set the parameter `ndt.backend` of `autoware_ndt_scan_matcher` to `ndt_cuda`:

```yaml
ndt:
  backend: "ndt_cuda"
```

At startup, the node writes `NDT backend: ndt_cuda` to its log. If the computer has no CUDA device, the node stops at startup with an error message.

## Behavior

The backend uses the neighbor search and the step of ndt_omp in Autoware: all voxels whose mean is within the resolution of a point, and no line search. The scores follow the definitions of ndt_omp, but the GPU computes with other rounding. So the results are close to the results of the backend `omp`, and they are not bit-identical.

The parameter `ndt.num_threads` has no effect on the GPU.

The map work follows the rules of the interface:

- `add_target()` builds the voxels of a map cloud on the CPU.
- `commit_target()` merges the voxels and copies them to the GPU.
- `clone()` shares the map on the GPU with the original, so the copy is fast and `align()` does no map work.

## Tests

`test_ndt_cuda_backend` loads the backends `ndt_cuda` and `omp` with pluginlib. It compares their alignment results and scores, and it tests the copies, the empty copy and a map update in a second thread. Without a CUDA device, the tests are skipped.

## License

Apache License 2.0, as the interface of Autoware Core. The library ndt_cuda has its own licenses (see the [README](../README.md) of the repository).
