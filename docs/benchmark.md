# ndt_cuda benchmark report

Measured on 2026-11-07 at commit `763df1a`.

On the sample data, ndt_cuda runs 9–93 times faster than ndt_omp and reaches the same result. These results apply to one laptop GPU and a small dataset.

- **Speed:** with the recommended `DIRECT7` method, one `align()` takes 0.64 ms instead of 12.04 ms (19 times faster).
- **Scaling:** the gap grows with the number of points. On a synthetic cloud of 5 million points, the speedup is 90–190 times.
- **Agreement with ndt_omp:** the largest differences are 23 µm in translation and 0.017° in rotation.
- **GPU resources:** about 94 MiB of video memory and about 1% GPU utilization at 10 Hz on the sample data. Larger maps and scans need more, as shown in [GPU memory and utilization](#gpu-memory-and-utilization).
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

## GPU memory and utilization

These values describe the library, because this repository has no ROS node. They come from `nvidia-smi` on the same GPU, with no other compute program running. Only 15 MiB of video memory was in use beforehand, and the driver reserved another 326 MiB of the card's 4096 MiB.

### Video memory

ndt_cuda uses about 94 MiB of video memory on the sample data. Most of that is a fixed cost, and the rest grows with the map and the scan.

| Map voxels | Map points | Scan points | Process: context only (MiB) | Process: with map (MiB) | Process: with map and scan (MiB) | Whole GPU: increase (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| 599 (sample data) | 15772 | 15950 | 82 | 84 | 86 | 94 |
| 100000 | 800000 | 100000 | 82 | 90 | 94 | 102 |
| 1000000 | 8000000 | 100000 | 82 | 144 | 148 | 156 |
| 5000000 | 40000000 | 100000 | 82 | 440 | 444 | 452 |
| 1000000 | 8000000 | 1000000 | 82 | 144 | 164 | 172 |
| 1000000 | 8000000 | 5000000 | 82 | 144 | 226 | 234 |

For planning, the use on this GPU can be estimated from three parts:

- **Fixed cost:** about 90 MiB, made up of the 82 MiB CUDA context and the 8 MiB more that the whole GPU shows. It could differ on other GPUs and drivers.
- **Map:** 48 bytes per voxel, plus 8 bytes per hash table slot. The slot count is the smallest power of two that is at least twice the voxel count, with a minimum of 1024. This gives 61.8 MiB for 1 million voxels and 356.9 MiB for 5 million, against 62 and 358 MiB measured.
- **Scan:** 16 bytes per point. Measured use is 2.5–6 MiB higher than this, because of scratch buffers and allocation granularity.

The map size depends on voxels, not points: more points in the same voxels do not use more video memory. The kd-tree from [Map updates](#map-updates) is built on the CPU and uses no video memory.

### Utilization

At a typical LiDAR rate, ndt_cuda keeps the GPU about 1% busy on the sample data and about 16% busy with a million voxel map and 100000 point scans.

| Map voxels | Scan points | Rate | `align()` (ms) | Time inside `align()` (%) | GPU utilization (%) | Peak GPU utilization (%) | Memory utilization (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 599 | 15950 | idle | — | 0.0 | 0.0 | 0 | 0.0 |
| 599 | 15950 | 10 Hz | 0.92 | 0.9 | 1.0 | 1 | 0.0 |
| 599 | 15950 | 20 Hz | 0.94 | 1.9 | 1.0 | 1 | 0.0 |
| 599 | 15950 | continuous (1881 per second) | 0.53 | 100.0 | 75.6 | 83 | 0.0 |
| 599 | 100000 | 10 Hz | 4.50 | 4.5 | 2.9 | 3 | 0.0 |
| 599 | 100000 | 20 Hz | 4.48 | 9.0 | 5.9 | 6 | 0.0 |
| 599 | 100000 | continuous (365 per second) | 2.74 | 100.0 | 78.9 | 81 | 0.0 |
| 1000000 | 100000 | 10 Hz | 18.00 | 18.0 | 15.7 | 16 | 7.9 |
| 1000000 | 100000 | 20 Hz | 17.84 | 35.7 | 32.4 | 33 | 16.7 |
| 1000000 | 100000 | continuous (59 per second) | 16.99 | 100.0 | 96.0 | 97 | 48.5 |

Each row is a 30 second run, sampled about every 275 ms (109–113 samples). "Memory utilization" is the share of time the GPU was reading or writing its memory, not how much memory is in use.

- **Low values:** `nvidia-smi` reports whole percentages averaged over a window, so readings near 1% are approximate.
- **Larger maps:** with the same 100000 point scans, one `align()` takes about 6 times longer on the million voxel map when run continuously (2.74 against 16.99 ms).
- **Slower calls at low rates:** one `align()` takes longer at 10 or 20 Hz than in a continuous loop (0.92 against 0.53 ms). The GPU is probably leaving its idle state between calls, although this was not measured.
- **Synthetic data:** the large maps fill a cube of voxels with 8 random points each, and the scans are random map points. Real maps have a different structure, so the larger rows are estimates.

## Limitations

These results come from one machine, so the speedups should not be generalized.

- **Hardware:** only the RTX 3050 Ti was tested. There are no measurements for desktop GPUs or Jetson Orin.
- **Data:** the real data is two small scans. The larger clouds and maps are synthetic, and no real driving log was tested.
- **Compiler flags:** `-march=native` was not used. It could speed up ndt_omp and so lower the speedup.
- **Not measured:** host memory use, power draw, end-to-end latency inside ROS, several instances in one process, and thermal behavior over long runs.

## Reproducing

The GPU memory and utilization section was measured with the same library code as the rest of this report, using `ndt_cuda_benchmark_gpu_usage`.

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

`ndt_cuda_benchmark_gpu_usage` measures video memory and GPU utilization with `nvidia-smi`. The full run took about 6 minutes, and it should be the only program using the GPU:

```bash
./build/ndt_cuda_benchmark_gpu_usage
```
