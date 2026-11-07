#!/usr/bin/env bash
# Makes Autoware's autoware_ndt_scan_matcher package use ndt_cuda instead of its copy of tier4/ndt_omp.
# usage: apply.sh <path to autoware_core/localization/autoware_ndt_scan_matcher>
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE="${1:?usage: apply.sh <path to autoware_ndt_scan_matcher>}"
OMP_INCLUDE="$PACKAGE/include/autoware/ndt_scan_matcher/ndt_omp"

if [[ ! -f "$PACKAGE/package.xml" || ! -f "$OMP_INCLUDE/multigrid_ndt_omp.h" ]]; then
  echo "$PACKAGE is not an autoware_ndt_scan_matcher package" >&2
  exit 1
fi
if grep -q ndt_cuda "$PACKAGE/package.xml"; then
  echo "already applied to $PACKAGE"
  exit 0
fi

cp "$HERE"/include/autoware/ndt_scan_matcher/ndt_omp/* "$OMP_INCLUDE/"
rm -f "$OMP_INCLUDE/multi_voxel_grid_covariance_omp.h"
rm -f "$PACKAGE"/src/ndt_omp/multi_voxel_grid_covariance_omp* "$PACKAGE"/src/ndt_omp/multigrid_ndt_omp*

sed -i '/src\/ndt_omp\/multi_voxel_grid_covariance_omp.cpp/d; /src\/ndt_omp\/multigrid_ndt_omp.cpp/d' "$PACKAGE/CMakeLists.txt"
sed -i 's/^target_link_libraries(multigrid_ndt_omp \${PCL_LIBRARIES})$/find_package(ndt_cuda REQUIRED)\ntarget_link_libraries(multigrid_ndt_omp ${PCL_LIBRARIES} ndt_cuda::ndt_cuda)/' "$PACKAGE/CMakeLists.txt"
# These calls relied on argument dependent lookup in pclomp::, but the types are now in ndt_cuda::.
sed -i 's/\bestimate_xy_covariance_by_multi_ndt/pclomp::&/' "$PACKAGE/src/ndt_scan_matcher_core.cpp"
sed -i 's/^\(\s*\)<depend>nav_msgs<\/depend>/\1<depend>nav_msgs<\/depend>\n\1<depend>ndt_cuda<\/depend>/' "$PACKAGE/package.xml"

grep -q 'ndt_cuda::ndt_cuda' "$PACKAGE/CMakeLists.txt" || { echo "CMakeLists.txt was not patched: this Autoware version needs a manual change" >&2; exit 1; }
grep -q '<depend>ndt_cuda</depend>' "$PACKAGE/package.xml" || { echo "package.xml was not patched: this Autoware version needs a manual change" >&2; exit 1; }
echo "$PACKAGE now uses ndt_cuda"
