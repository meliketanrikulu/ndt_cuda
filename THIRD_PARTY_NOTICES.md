# Third-party notices

ndt_cuda is a GPU port of the Normal Distributions Transform (NDT) registration of the
Point Cloud Library and of its OpenMP variant koide3/ndt_omp. The following files contain
code derived from these projects; each of them keeps the original copyright notice in its header.

| File | Derived from |
|---|---|
| `include/ndt_cuda/ndt_cuda.hpp`, `include/ndt_cuda/impl/ndt_cuda.hpp` | PCL `pcl/registration/ndt.h` (optimizer, More-Thuente line search) via ndt_omp `ndt_omp.h` / `ndt_omp_impl.hpp` |
| `include/ndt_cuda/voxel_grid_covariance.hpp`, `include/ndt_cuda/impl/voxel_grid_covariance.hpp` | PCL `pcl/filters/voxel_grid_covariance.h` (voxel statistics) via ndt_omp `voxel_grid_covariance_omp*.h` |
| `include/ndt_cuda/angle_derivatives.hpp`, `src/ndt_cuda_kernels.cuh` | PCL / ndt_omp derivative computation (`computeAngleDerivatives`, `computePointDerivatives`, `updateDerivatives`) |
| `include/ndt_cuda/multigrid_ndt_cuda.hpp`, `include/ndt_cuda/impl/multigrid_ndt_cuda.hpp`, `include/ndt_cuda/ndt_struct.hpp`, and the NVTL, `getResult()` and regularization parts of `include/ndt_cuda/ndt_cuda.hpp`, `include/ndt_cuda/impl/ndt_cuda.hpp` and `src/ndt_cuda_kernels.cuh` | tier4/ndt_omp `multigrid_ndt_omp*.h`, `ndt_struct.hpp` (a fork of koide3/ndt_omp under the same BSD 2-Clause license) |
| `data/251370668.pcd`, `data/251371071.pcd` | Sample scans shipped with ndt_omp |

The algorithm itself is described in:

- M. Magnusson, *The Three-Dimensional Normal-Distributions Transform — an Efficient Representation for Registration, Surface Analysis, and Loop Detection*, PhD thesis, Örebro University, 2009.
- J. J. Moré and D. J. Thuente, *Line Search Algorithms with Guaranteed Sufficient Decrease*, ACM Transactions on Mathematical Software, 1994.

The test and benchmark programs download koide3/ndt_omp and tier4/ndt_omp at build time to use them as reference
implementations; they are not part of the library.

---

## Point Cloud Library (PCL) — BSD 3-Clause License

```
Software License Agreement (BSD License)

Point Cloud Library (PCL) - www.pointclouds.org
Copyright (c) 2010-2012, Willow Garage, Inc.
Copyright (c) 2012-, Open Perception, Inc.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.
 * Neither the name of the copyright holder(s) nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

The NDT implementation in PCL was written by Brian Okorn (Space and Naval Warfare Systems Center Pacific), with math refactored by Todor Stoyanov.

## koide3/ndt_omp — BSD 2-Clause License

https://github.com/koide3/ndt_omp

```
BSD 2-Clause License

Copyright (c) 2019, k.koide
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
