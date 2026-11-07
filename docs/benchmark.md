# ndt_cuda benchmark report

Measured on 2026-11-07 at commit `2661076`.

On the sample data, ndt_cuda runs 10–97 times faster than ndt_omp and reaches the same result. These results apply to one laptop GPU and a small dataset.

- **Speed:** with the recommended `DIRECT7` method, one `align()` takes 0.64 ms instead of 12.62 ms (20 times faster).
- **Scaling:** the gap grows with the number of points. On a synthetic cloud of 5 million points, the speedup is 89–201 times.
- **Agreement with ndt_omp:** the largest differences are 23 µm in translation and 0.017° in rotation.
- **Map updates:** on a 3 million point map, the first `align()` takes 1.3 ms, against 714 ms for ndt_omp.

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
| `DIRECT7` | 12.62 | 12.25–14.77 | 0.64 | 0.64–0.66 | 19.8x | 5 |
| `KDTREE` | 19.25 | 18.44–23.85 | 0.79 | 0.79–0.80 | 24.3x | 5 |
| `DIRECT26` | 58.11 | 57.18–74.66 | 0.60 | 0.59–0.61 | 96.5x | 2 |
| `DIRECT1` | 6.21 | 6.08–8.49 | 0.63 | 0.62–0.64 | 9.9x | 5 |

Each row comes from 101 measurements.

Using 16 threads is the fairest baseline for ndt_omp here. With `DIRECT7`, 1, 4, 8 and 16 threads gave 66.0, 24.1, 17.7 and 12.3 ms.

## Scaling with large clouds

At 5 million points, ndt_cuda takes 53–102 ms and ndt_omp takes 4.7–18.5 s. The times below are medians in milliseconds, based on 5–31 repetitions per measurement.

| Source points | `DIRECT7` ndt_omp | `DIRECT7` ndt_cuda | `KDTREE` ndt_omp | `KDTREE` ndt_cuda | `DIRECT26` ndt_omp | `DIRECT26` ndt_cuda | `DIRECT1` ndt_omp | `DIRECT1` ndt_cuda |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100000 | 109.8 | 3.05 | 139.0 | 3.51 | 632.0 | 5.09 | 58.1 | 2.89 |
| 300000 | 380.5 | 6.72 | 639.1 | 9.56 | 1168.2 | 7.56 | 272.6 | 7.82 |
| 1000000 | 1320.0 | 16.49 | 2113.8 | 25.03 | 6278.8 | 27.78 | 921.8 | 14.77 |
| 5000000 | 6781.9 | 69.36 | 9467.8 | 102.09 | 18514.6 | 91.97 | 4725.3 | 53.25 |

Speedup at 5 million points:

| Method | Speedup |
| --- | --- |
| `DIRECT7` | 97.8x |
| `KDTREE` | 92.7x |
| `DIRECT26` | 201.3x |
| `DIRECT1` | 88.7x |

When the point count grows 50 times, from 100 thousand to 5 million, the `DIRECT7` time grows 62 times in ndt_omp and 23 times in ndt_cuda. The 5 million point `DIRECT1` measurement comes from a separate run; the others come from one run.

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

## Map updates

The first `align()` after a map change is nearly as fast as later calls in ndt_cuda. Before the kd-tree fix, this call took 678 ms on the 3 million point map.

| Map points | Voxels | Implementation | `setInputTarget()` (ms) | First `align()` (ms) | Second `align()` (ms) |
| --- | --- | --- | --- | --- | --- |
| 1000000 | 1156 | ndt_cuda | 14.7 | 1.58 | 0.87 |
| 1000000 | 1156 | ndt_omp | 30.5 | 244.1 | 54.8 |
| 3000000 | 1166 | ndt_cuda | 42.0 | 1.27 | 0.59 |
| 3000000 | 1166 | ndt_omp | 91.1 | 714.2 | 36.3 |

Each value is a single measurement, so read these as orders of magnitude. The source cloud has 15950 points and the method is `DIRECT7`.

## Limitations

These results come from one machine, so the speedups should not be generalized.

- **Hardware:** only the RTX 3050 Ti was tested. There are no measurements for desktop GPUs or Jetson Orin.
- **Data:** the real data is two small scans. The larger clouds and maps are synthetic, and no real driving log was tested. Maps with many more voxels may behave differently.
- **Compiler flags:** `-march=native` was not used. It could speed up ndt_omp and so lower the speedup.
- **Not measured:** GPU memory use, power draw, end-to-end latency inside ROS, and thermal behavior over long runs.
- **README numbers:** the table in the README comes from an earlier measurement with 10 repetitions (for example, 21x for `DIRECT7`). The medians here use more repetitions, so the values differ slightly.

## Reproducing

`ndt_cuda_benchmark` compares both implementations on the sample data or on your own clouds:

```bash
cmake -B build -DNDT_CUDA_BUILD_BENCHMARK=ON && cmake --build build -j
./build/ndt_cuda_benchmark                              # data/*.pcd, all search methods
./build/ndt_cuda_benchmark target.pcd source.pcd 2.0    # your own clouds and resolution
```

`ndt_cuda_benchmark_report` runs the measurements of this report: repeated timings, synthetic clouds, thread scaling, determinism, the perturbed guess and map changes. It uses all CPU threads, and the full run takes about 10 minutes. A smaller limit on the source points shortens it:

```bash
./build/ndt_cuda_benchmark_report                       # full run, up to 5 million source points
./build/ndt_cuda_benchmark_report data 300000           # shorter run, up to 300000 source points
```

The numbers above were measured with an earlier version of this program, which used the same steps and repetition counts. The 5 million point `DIRECT1` measurement was taken in a separate run.
