cmake_minimum_required(VERSION 3.16)
project(posegrasp_cpp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Werror=return-type)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)

add_library(backproject STATIC src/backproject.cpp)
target_include_directories(backproject PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
set_target_properties(backproject PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(segment_cloud SHARED src/segment_cloud.cpp)
target_link_libraries(segment_cloud backproject)
ament_target_dependencies(segment_cloud rclcpp rclcpp_components sensor_msgs message_filters)
rclcpp_components_register_node(segment_cloud
  PLUGIN "posegrasp_cpp::SegmentCloud"
  EXECUTABLE segment_cloud_node)

install(TARGETS backproject segment_cloud
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_backproject test/test_backproject.cpp)
  target_link_libraries(test_backproject backproject)
endif()

ament_package()
