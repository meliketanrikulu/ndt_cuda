cmake_minimum_required(VERSION 3.14)
project(ndt_cuda_autoware_backend)

find_package(autoware_cmake REQUIRED)
autoware_package()

# The NDT backend "ndt_cuda". autoware_ndt_scan_matcher loads it with pluginlib, so no other library
# may link it. It is a plain add_library(), because ament_auto exports its libraries.
add_library(${PROJECT_NAME} SHARED
  src/ndt_cuda_backend.cpp
)
target_link_libraries(${PROJECT_NAME} ndt_cuda::ndt_cuda)
ament_target_dependencies(${PROJECT_NAME} autoware_ndt_scan_matcher_common pluginlib)
install(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION lib
)
pluginlib_export_plugin_description_file(autoware_ndt_scan_matcher_common plugins.xml)

if(BUILD_TESTING)
  # pluginlib loads the backends "ndt_cuda" and "omp" here. The test does not link them.
  ament_auto_add_gtest(test_ndt_cuda_backend
    test/test_ndt_cuda_backend.cpp
  )
endif()

ament_auto_package()
