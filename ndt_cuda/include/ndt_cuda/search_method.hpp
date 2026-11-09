// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Melike

#pragma once

namespace ndt_cuda {

/** \brief Neighbor voxel search methods. The names and values are the same as pclomp::NeighborSearchMethod. */
enum NeighborSearchMethod {
  /** \brief All voxels whose mean lies within `resolution` of the point (same result as ndt_omp's kd-tree radius search). */
  KDTREE = 0,
  /** \brief The 26 voxels around the voxel containing the point (not the voxel itself), as in ndt_omp. */
  DIRECT26 = 1,
  /** \brief The voxel containing the point and its 6 face neighbors. Recommended. */
  DIRECT7 = 2,
  /** \brief Only the voxel containing the point. Fastest, least robust. */
  DIRECT1 = 3,
};

}  // namespace ndt_cuda
