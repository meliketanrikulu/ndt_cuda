cmake_minimum_required(VERSION 3.18)
project(ndt_cuda VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(NDT_CUDA_ENABLE_CUDA "Build the CUDA library (OFF: only the GPU-less tests can be built)" ON)
option(NDT_CUDA_BUILD_EXAMPLES "Build the example program" ON)
option(NDT_CUDA_BUILD_TESTS "Build the validation tests (no GPU needed; download koide3/ndt_omp and tier4/ndt_omp as references)" OFF)
option(NDT_CUDA_BUILD_GPU_TESTS "Build the tests of the real CUDA kernels (need a GPU to run; download koide3/ndt_omp and tier4/ndt_omp as references)" OFF)
option(NDT_CUDA_BUILD_BENCHMARK "Build the ndt_omp vs ndt_cuda benchmark (downloads koide3/ndt_omp)" OFF)
set(NDT_OMP_SOURCE_DIR "" CACHE PATH "Local checkout of koide3/ndt_omp used as reference by the tests / benchmark (downloaded if empty)")
set(NDT_OMP_TIER4_SOURCE_DIR "" CACHE PATH "Local checkout of tier4/ndt_omp used as reference by the tests (downloaded if empty)")

# ---------------------------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------------------------
# Workaround for Ubuntu 24.04's VTK package (see the file); also installed for consumers.
include(cmake/ndt_cuda-extras.cmake)

find_package(PCL 1.11 REQUIRED COMPONENTS common io filters registration)
find_package(Eigen3 REQUIRED NO_MODULE)

# PCL_DEFINITIONS can mix -D definitions and compiler flags; only the definitions are needed (and safe for nvcc).
set(NDT_CUDA_PCL_DEFINITIONS "")
foreach(def ${PCL_DEFINITIONS})
  if(def MATCHES "^-D")
    list(APPEND NDT_CUDA_PCL_DEFINITIONS ${def})
  endif()
endforeach()

set(NDT_CUDA_HAVE_CUDA OFF)
if(NDT_CUDA_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
endif()
if(NDT_CUDA_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(NDT_CUDA_DEFAULT_ARCHITECTURES ON)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(CUDAToolkit_VERSION VERSION_LESS 11.4)
    message(FATAL_ERROR "ndt_cuda requires CUDA >= 11.4 (found ${CUDAToolkit_VERSION})")
  endif()
  if(NDT_CUDA_DEFAULT_ARCHITECTURES)
    # Turing, Ampere, Jetson Orin, (Ada, Hopper with CUDA >= 11.8). Every entry also embeds PTX, so newer GPUs JIT compile.
    set(CMAKE_CUDA_ARCHITECTURES 75 86 87)
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 11.8)
      list(APPEND CMAKE_CUDA_ARCHITECTURES 89 90)
    endif()
  endif()
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(NDT_CUDA_HAVE_CUDA ON)
  message(STATUS "ndt_cuda: CUDA ${CUDAToolkit_VERSION}, architectures: ${CMAKE_CUDA_ARCHITECTURES}")
elseif(NDT_CUDA_ENABLE_CUDA)
  message(FATAL_ERROR "ndt_cuda: no CUDA compiler found. Install the CUDA toolkit, or configure with -DNDT_CUDA_ENABLE_CUDA=OFF to build only the GPU-less tests.")
endif()

# ---------------------------------------------------------------------------------------------
# Host part (voxel grid, optimizer). Shared by the library and the GPU-less tests.
# ---------------------------------------------------------------------------------------------
add_library(ndt_cuda_core OBJECT
  src/voxel_grid_covariance.cpp
  src/ndt_cuda.cpp
)
set_target_properties(ndt_cuda_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(ndt_cuda_core PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_definitions(ndt_cuda_core PUBLIC ${NDT_CUDA_PCL_DEFINITIONS})
target_link_libraries(ndt_cuda_core PUBLIC pcl_common pcl_registration Eigen3::Eigen)

# ---------------------------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------------------------
if(NDT_CUDA_HAVE_CUDA)
  add_library(ndt_cuda SHARED
    src/gpu_backend.cu
    $<TARGET_OBJECTS:ndt_cuda_core>
  )
  add_library(ndt_cuda::ndt_cuda ALIAS ndt_cuda)
  set_target_properties(ndt_cuda PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(ndt_cuda PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
  )
  target_compile_definitions(ndt_cuda PUBLIC ${NDT_CUDA_PCL_DEFINITIONS})
  target_compile_features(ndt_cuda PUBLIC cxx_std_17)
  target_compile_options(ndt_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  # The CUDA runtime is linked statically: at runtime only the NVIDIA driver is needed.
  target_link_libraries(ndt_cuda
    PUBLIC pcl_common pcl_registration Eigen3::Eigen
    PRIVATE CUDA::cudart_static
  )
endif()

# ---------------------------------------------------------------------------------------------
# Reference implementation (koide3/ndt_omp) for the tests and the benchmark
# ---------------------------------------------------------------------------------------------
if(NDT_CUDA_BUILD_TESTS OR ((NDT_CUDA_BUILD_BENCHMARK OR NDT_CUDA_BUILD_GPU_TESTS) AND NDT_CUDA_HAVE_CUDA))
  find_package(OpenMP REQUIRED)
  if(NDT_OMP_SOURCE_DIR)
    set(NDT_OMP_REFERENCE_DIR ${NDT_OMP_SOURCE_DIR})
  else()
    include(FetchContent)
    FetchContent_Declare(ndt_omp_reference
      GIT_REPOSITORY https://github.com/koide3/ndt_omp.git
      GIT_TAG 5495fd9214945afcb4b35d5a1da385e405c52bf9
      SOURCE_SUBDIR do-not-add-subdirectory  # sources only, ndt_omp's own CMakeLists.txt is not used
    )
    FetchContent_MakeAvailable(ndt_omp_reference)
    set(NDT_OMP_REFERENCE_DIR ${ndt_omp_reference_SOURCE_DIR})
  endif()

  add_library(ndt_omp_reference STATIC
    ${NDT_OMP_REFERENCE_DIR}/src/pclomp/ndt_omp.cpp
    ${NDT_OMP_REFERENCE_DIR}/src/pclomp/voxel_grid_covariance_omp.cpp
  )
  target_include_directories(ndt_omp_reference SYSTEM PUBLIC ${NDT_OMP_REFERENCE_DIR}/include)
  target_compile_definitions(ndt_omp_reference PUBLIC ${NDT_CUDA_PCL_DEFINITIONS})
  target_compile_options(ndt_omp_reference PRIVATE -w)
  target_link_libraries(ndt_omp_reference PUBLIC pcl_common pcl_filters pcl_registration OpenMP::OpenMP_CXX)
endif()

# ---------------------------------------------------------------------------------------------
# Reference implementation (tier4/ndt_omp) for the tests of the multi-tile NDT. In a separate
# target because it defines the same pclomp classes as koide3/ndt_omp.
# ---------------------------------------------------------------------------------------------
if(NDT_CUDA_BUILD_TESTS OR (NDT_CUDA_BUILD_GPU_TESTS AND NDT_CUDA_HAVE_CUDA))
  if(NDT_OMP_TIER4_SOURCE_DIR)
    set(NDT_OMP_TIER4_REFERENCE_DIR ${NDT_OMP_TIER4_SOURCE_DIR})
  else()
    include(FetchContent)
    FetchContent_Declare(ndt_omp_tier4_reference
      GIT_REPOSITORY https://github.com/tier4/ndt_omp.git
      GIT_TAG 025f263f41b7cd2d00a547ca787f2d6e8e8fa79b
      SOURCE_SUBDIR do-not-add-subdirectory  # sources only, its own CMakeLists.txt is not used
    )
    FetchContent_MakeAvailable(ndt_omp_tier4_reference)
    set(NDT_OMP_TIER4_REFERENCE_DIR ${ndt_omp_tier4_reference_SOURCE_DIR})
  endif()

  add_library(ndt_omp_tier4_reference STATIC
    ${NDT_OMP_TIER4_REFERENCE_DIR}/src/multigrid_pclomp/multigrid_ndt_omp.cpp
    ${NDT_OMP_TIER4_REFERENCE_DIR}/src/multigrid_pclomp/multi_voxel_grid_covariance_omp.cpp
    ${NDT_OMP_TIER4_REFERENCE_DIR}/src/pclomp/ndt_omp.cpp
    ${NDT_OMP_TIER4_REFERENCE_DIR}/src/pclomp/voxel_grid_covariance_omp.cpp
  )
  target_include_directories(ndt_omp_tier4_reference SYSTEM PUBLIC ${NDT_OMP_TIER4_REFERENCE_DIR}/include)
  target_compile_definitions(ndt_omp_tier4_reference PUBLIC ${NDT_CUDA_PCL_DEFINITIONS})
  target_compile_options(ndt_omp_tier4_reference PRIVATE -w)
  target_link_libraries(ndt_omp_tier4_reference PUBLIC pcl_common pcl_filters pcl_registration OpenMP::OpenMP_CXX)
endif()

# ---------------------------------------------------------------------------------------------
# Tests (CPU emulation of the GPU backend: no GPU and no CUDA toolkit required)
# ---------------------------------------------------------------------------------------------
if(NDT_CUDA_BUILD_TESTS)
  enable_testing()
  add_executable(ndt_cuda_test
    test/ndt_cuda_test.cpp
    test/gpu_backend_emulated.cpp
    test/compat_standalone.cpp
    $<TARGET_OBJECTS:ndt_cuda_core>
  )
  target_include_directories(ndt_cuda_test PRIVATE include src)
  target_compile_definitions(ndt_cuda_test PRIVATE NDT_CUDA_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data" ${NDT_CUDA_PCL_DEFINITIONS})
  target_link_libraries(ndt_cuda_test PRIVATE ndt_omp_reference pcl_common pcl_io pcl_filters pcl_registration Eigen3::Eigen)
  add_test(NAME ndt_cuda_test COMMAND ndt_cuda_test)

  add_executable(ndt_cuda_multigrid_test
    test/ndt_cuda_multigrid_test.cpp
    test/gpu_backend_emulated.cpp
    $<TARGET_OBJECTS:ndt_cuda_core>
  )
  target_include_directories(ndt_cuda_multigrid_test PRIVATE include src)
  target_compile_definitions(ndt_cuda_multigrid_test PRIVATE NDT_CUDA_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data" ${NDT_CUDA_PCL_DEFINITIONS})
  target_link_libraries(ndt_cuda_multigrid_test PRIVATE ndt_omp_tier4_reference pcl_common pcl_io pcl_filters pcl_registration Eigen3::Eigen)
  add_test(NAME ndt_cuda_multigrid_test COMMAND ndt_cuda_multigrid_test)
endif()

# ---------------------------------------------------------------------------------------------
# Test of the real CUDA kernels (reports "skipped" when no GPU is available)
# ---------------------------------------------------------------------------------------------
if(NDT_CUDA_HAVE_CUDA AND NDT_CUDA_BUILD_GPU_TESTS)
  enable_testing()
  add_executable(ndt_cuda_gpu_test test/ndt_cuda_gpu_test.cpp)
  target_compile_definitions(ndt_cuda_gpu_test PRIVATE NDT_CUDA_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
  target_link_libraries(ndt_cuda_gpu_test PRIVATE ndt_cuda ndt_omp_reference pcl_io pcl_filters)
  add_test(NAME ndt_cuda_gpu_test COMMAND ndt_cuda_gpu_test)
  set_tests_properties(ndt_cuda_gpu_test PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(ndt_cuda_multigrid_gpu_test test/ndt_cuda_multigrid_test.cpp)
  target_compile_definitions(ndt_cuda_multigrid_gpu_test PRIVATE NDT_CUDA_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
  target_link_libraries(ndt_cuda_multigrid_gpu_test PRIVATE ndt_cuda ndt_omp_tier4_reference pcl_io pcl_filters)
  add_test(NAME ndt_cuda_multigrid_gpu_test COMMAND ndt_cuda_multigrid_gpu_test)
  set_tests_properties(ndt_cuda_multigrid_gpu_test PROPERTIES SKIP_RETURN_CODE 77)
endif()

# ---------------------------------------------------------------------------------------------
# Example and benchmark
# ---------------------------------------------------------------------------------------------
if(NDT_CUDA_HAVE_CUDA AND NDT_CUDA_BUILD_EXAMPLES)
  add_executable(ndt_cuda_align examples/align.cpp)
  target_link_libraries(ndt_cuda_align PRIVATE ndt_cuda pcl_io pcl_filters)
endif()

if(NDT_CUDA_HAVE_CUDA AND NDT_CUDA_BUILD_BENCHMARK)
  add_executable(ndt_cuda_benchmark benchmark/benchmark.cpp)
  target_link_libraries(ndt_cuda_benchmark PRIVATE ndt_cuda ndt_omp_reference pcl_io pcl_filters)
  target_compile_definitions(ndt_cuda_benchmark PRIVATE NDT_CUDA_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

  add_executable(ndt_cuda_benchmark_report benchmark/benchmark_report.cpp)
  target_link_libraries(ndt_cuda_benchmark_report PRIVATE ndt_cuda ndt_omp_reference pcl_io pcl_filters)
  target_compile_definitions(ndt_cuda_benchmark_report PRIVATE NDT_CUDA_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

  add_executable(ndt_cuda_benchmark_gpu_usage benchmark/benchmark_gpu_usage.cpp)
  target_link_libraries(ndt_cuda_benchmark_gpu_usage PRIVATE ndt_cuda pcl_io pcl_filters)
  target_compile_definitions(ndt_cuda_benchmark_gpu_usage PRIVATE NDT_CUDA_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
endif()

# ---------------------------------------------------------------------------------------------
# Install / export
# ---------------------------------------------------------------------------------------------
include(GNUInstallDirs)

find_package(ament_cmake QUIET)

if(NDT_CUDA_HAVE_CUDA)
  # Headers go to include/ndt_cuda/ndt_cuda/... (the layout recommended since ROS 2 Humble, works everywhere).
  install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})
  install(TARGETS ndt_cuda
    EXPORT ndt_cudaTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION include/${PROJECT_NAME}
  )
  if(TARGET ndt_cuda_align)
    if(ament_cmake_FOUND)
      install(TARGETS ndt_cuda_align DESTINATION lib/${PROJECT_NAME})
    else()
      install(TARGETS ndt_cuda_align DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
  endif()
endif()

if(ament_cmake_FOUND)
  # ROS 2 (colcon): consumers use find_package(ndt_cuda) + target_link_libraries(... ndt_cuda::ndt_cuda)
  if(NDT_CUDA_HAVE_CUDA)
    ament_export_targets(ndt_cudaTargets HAS_LIBRARY_TARGET)
    ament_export_dependencies(PCL Eigen3)
  endif()
  ament_package(CONFIG_EXTRAS cmake/ndt_cuda-extras.cmake)  # included before the exported dependencies are found
elseif(NDT_CUDA_HAVE_CUDA)
  # Plain CMake: find_package(ndt_cuda) + target_link_libraries(... ndt_cuda::ndt_cuda)
  include(CMakePackageConfigHelpers)
  set(NDT_CUDA_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
  install(EXPORT ndt_cudaTargets NAMESPACE ndt_cuda:: DESTINATION ${NDT_CUDA_CMAKE_DIR})
  configure_package_config_file(cmake/ndt_cudaConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/ndt_cudaConfig.cmake INSTALL_DESTINATION ${NDT_CUDA_CMAKE_DIR})
  write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/ndt_cudaConfigVersion.cmake COMPATIBILITY SameMajorVersion)
  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ndt_cudaConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ndt_cudaConfigVersion.cmake
    cmake/ndt_cuda-extras.cmake
    DESTINATION ${NDT_CUDA_CMAKE_DIR})
endif()
