# ndt_cuda benchmark report

Measured on 2026-11-07 at commit `763df1a`.

On the sample data, ndt_cuda runs 9–93 times faster than ndt_omp and reaches the same result. These results apply to one laptop GPU and a small dataset.

- **Speed:** with the recommended `DIRECT7` method, one `align()` takes 0.64 ms instead of 12.04 ms (19 times faster).
- **Scaling:** the gap grows with the number of points. On a synthetic cloud of 5 million points, the speedup is 90–190 times.
- **Agreement with ndt_omp:** the largest differences are 23 µm in translation and 0.017° in rotation.
- **Map updates:** on a 3 million point map, the first `align()` takes 656 ms by default, against 712 ms for ndt_omp. It takes 0.75 ms if the kd-tree that only `getFitnessScore()` needs is turned off.

## Test setup

The reference is koide3/ndt_omp at commit `5495fd9`, run with 16 threads.

| Component | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GB, driver 595.99.02 |
| CPU | Intel Core i7-11800H, 8 cores, 16 threads |
| Software | Ubuntu 24.04, g++ 13.3.0, CUDA 12.8, PCL 1.14.0 |
| Build | CMake Release for both; `-march=native` not used |
| NDT settings | Resolution 1 m, other parameters at their defaults |

Times are medians for a single `align()` call, excluding warm-up calls. `setInputTarget()` is reported separately under [Map updates](#map-updates).

## Data

The only real data is the pair of LiDAR scans from the ndt_omp repository, also included in this repository. Their SHA-256 hashes match upstream. The sensor and recording location are unknown, and the data has no ground truth.

| File | Raw points | After 0.1 m downsampling | Used as |
| --- | --- | --- | --- |
| `data/251370668.pcd` | 69088 | 15772 | Target (map) |
| `data/251371071.pcd` | 69792 | 15950 | Source (scan) |

The larger clouds are synthetic. They were made by repeating the downsampled points with Gaussian noise (5 mm standard deviation) added to each axis:

- **Large source clouds (100 thousand to 5 million points):** made from the source scan and aligned to the 15772 point target.
- **Large maps (1 and 3 million points):** made from the target scan.

These clouds add points but no new scene: the 3 million point map still has only 1166 voxels.

## Results on the sample data

GPU time is about 0.6–0.8 ms for all four methods, while ndt_omp's time depends strongly on the method. The largest speedup is therefore on `DIRECT26`, the slowest method in ndt_omp.

| Method | ndt_omp median (ms) | ndt_omp p10–p90 (ms) | ndt_cuda median (ms) | ndt_cuda p10–p90 (ms) | Speedup | Iterations |
| --- | --- | --- | --- | --- | --- | --- |
| `DIRECT7` | 12.04 | 11.86–12.25 | 0.64 | 0.64–0.65 | 18.7x | 5 |
| `KDTREE` | 18.29 | 18.16–20.18 | 0.79 | 0.78–0.80 | 23.1x | 5 |
| `DIRECT26` | 56.45 | 56.11–60.55 | 0.61 | 0.60–0.62 | 92.5x | 2 |
| `DIRECT1` | 6.06 | 5.99–6.15 | 0.64 | 0.63–0.64 | 9.5x | 5 |

Each row comes from 101 measurements.

Using 16 threads is the fairest baseline for ndt_omp here. With `DIRECT7`, 1, 4, 8 and 16 threads gave 65.5, 25.8, 17.3 and 12.2 ms.

## Scaling with large clouds

At 5 million points, ndt_cuda takes 53–102 ms and ndt_omp takes 4.7–19.4 s. The times below are medians in milliseconds, based on 5–31 repetitions per measurement.

| Source points | `DIRECT7` ndt_omp | `DIRECT7` ndt_cuda | `KDTREE` ndt_omp | `KDTREE` ndt_cuda | `DIRECT26` ndt_omp | `DIRECT26` ndt_cuda | `DIRECT1` ndt_omp | `DIRECT1` ndt_cuda |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100000 | 81.2 | 2.90 | 124.7 | 3.50 | 573.8 | 5.06 | 42.1 | 2.84 |
| 300000 | 366.1 | 7.13 | 547.1 | 8.57 | 1103.6 | 7.68 | 258.8 | 7.57 |
| 1000000 | 1276.1 | 16.90 | 1848.6 | 22.08 | 6275.1 | 27.61 | 903.1 | 14.65 |
| 5000000 | 6659.6 | 68.27 | 9257.4 | 100.61 | 19404.2 | 102.30 | 4745.0 | 52.63 |

Speedup at 5 million points:

| Method | Speedup |
| --- | --- |
| `DIRECT7` | 97.5x |
| `KDTREE` | 92.0x |
| `DIRECT26` | 189.7x |
| `DIRECT1` | 90.2x |

When the point count grows 50 times, from 100 thousand to 5 million, the `DIRECT7` time grows 82 times in ndt_omp and 24 times in ndt_cuda.

## Agreement and determinism

ndt_cuda converges to the same result as ndt_omp in every trial, with the same number of iterations. This compares the two implementations with each other, not with a ground truth pose.

| Method | Identity guess: translation difference (µm) | Identity guess: rotation difference (°) | Perturbed guess: translation difference (µm) | Perturbed guess: rotation difference (°) | Differing results in 200 repeats |
| --- | --- | --- | --- | --- | --- |
| `DIRECT7` | 1.6 | 0.00000 | 1.3 | 0.00266 | 0 |
| `KDTREE` | 1.0 | 0.00068 | 2.7 | 0.00000 | 0 |
| `DIRECT26` | 0.7 | 0.00000 | 23.3 | 0.00733 | 0 |
| `DIRECT1` | 1.8 | 0.00130 | 1.8 | 0.00978 | 0 |

The perturbed guess is offset by about 0.95 m and 2.9°. On the synthetic clouds, up to 5 million points, the largest differences were 4.6 µm and 0.017°.

All 200 repeats on the same GPU are bit-identical to the first result. This was not measured across different GPU models.

`getFitnessScore()` also agrees after the perturbed guess:

| Method | ndt_omp | ndt_cuda |
| --- | --- | --- |
| `DIRECT7` | 0.213726 | 0.213726 |
| `KDTREE` | 0.220164 | 0.220164 |
| `DIRECT26` | 0.377661 | 0.377651 |
| `DIRECT1` | 0.254760 | 0.254758 |

## Map updates

By default, the first `align()` after a map change costs about as much in ndt_cuda as in ndt_omp, because both build the kd-tree that `getFitnessScore()` needs. Code that never calls `getFitnessScore()` can skip it with `setBuildTargetKdTree(false)`, which removes nearly all of that cost. Later calls are fast either way.

| Map points | Voxels | Implementation | `setInputTarget()` (ms) | First `align()` (ms) | Second `align()` (ms) |
| --- | --- | --- | --- | --- | --- |
| 1000000 | 1156 | ndt_cuda, default | 15.0 | 168.56 | 0.89 |
| 1000000 | 1156 | ndt_cuda, kd-tree off | 15.6 | 1.04 | 0.86 |
| 1000000 | 1156 | ndt_omp | 30.7 | 228.08 | 52.94 |
| 3000000 | 1166 | ndt_cuda, default | 45.2 | 655.76 | 0.58 |
| 3000000 | 1166 | ndt_cuda, kd-tree off | 46.0 | 0.75 | 0.57 |
| 3000000 | 1166 | ndt_omp | 91.5 | 712.01 | 36.10 |

Each value is a single measurement, so read these as orders of magnitude. The source cloud has 15950 points and the method is `DIRECT7`.

## Limitations

These results come from one machine, so the speedups should not be generalized.

- **Hardware:** only the RTX 3050 Ti was tested. There are no measurements for desktop GPUs or Jetson Orin.
- **Data:** the real data is two small scans. The larger clouds and maps are synthetic, and no real driving log was tested. Maps with many more voxels may behave differently.
- **Compiler flags:** `-march=native` was not used. It could speed up ndt_omp and so lower the speedup.
- **Not measured:** GPU memory use, power draw, end-to-end latency inside ROS, and thermal behavior over long runs.

## Reproducing

`ndt_cuda_benchmark` compares both implementations on the sample data or on your own clouds:

```bash
cmake -B build -DNDT_CUDA_BUILD_BENCHMARK=ON && cmake --build build -j
./build/ndt_cuda_benchmark                              # data/*.pcd, all search methods
./build/ndt_cuda_benchmark target.pcd source.pcd 2.0    # your own clouds and resolution
```

`ndt_cuda_benchmark_report` runs the measurements of this report: repeated timings, synthetic clouds, thread scaling, determinism, the perturbed guess and map changes. It uses all CPU threads, and the full run took about 30 minutes on this machine. A smaller limit on the source points shortens it:

```bash
./build/ndt_cuda_benchmark_report                       # full run, up to 5 million source points
./build/ndt_cuda_benchmark_report data 300000           # shorter run, up to 300000 source points
```
