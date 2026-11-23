cmake_minimum_required(VERSION 3.8)
project(ros_onnx_segmentation)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(MSVC)
  add_compile_options(/wd4251 /wd4275)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(onnxruntime_vendor REQUIRED)
find_package(OpenCV REQUIRED)
find_package(vision_msgs REQUIRED)

add_library(${PROJECT_NAME}_component SHARED
  src/onnx_processor.cpp
  src/yolo_processor.cpp)

target_include_directories(${PROJECT_NAME}_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${OpenCV_INCLUDE_DIRS})

ament_target_dependencies(${PROJECT_NAME}_component
  rclcpp
  rclcpp_components
  std_msgs
  sensor_msgs
  cv_bridge
  onnxruntime_vendor
  vision_msgs)

target_link_libraries(${PROJECT_NAME}_component ${OpenCV_LIBS})

rclcpp_components_register_nodes(${PROJECT_NAME}_component "ros_onnx_segmentation::YoloProcessor")

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_component)

install(TARGETS ${PROJECT_NAME}_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

install(TARGETS ${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.onnx")

ament_package()
