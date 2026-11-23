# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.18)
project(onnxruntime_vendor)

find_package(ament_cmake REQUIRED)

set(ORT_VERSION 1.26.0)
set(ORT_CPU_SHA512 cc4f5cf194d2f74aa711daf749737ec74db01fa1aa0229648876d2818ee21033e293743527c70560413bb3b8f386ea3049af2b1d9fe04b66e480978f539dcced)
set(ORT_GPU_SHA512 f90ace6518b4f72a80b61d6ab659cb75d56ca5f594d2e88031f9539ca9e1cbb2b36336f27a6b5d66dbb3dfa5e5fafbaad7b6cc82f4b872fe348d2608f771e38b)

option(ONNXRUNTIME_GPU "Use the CUDA 12 / TensorRT 10 build (Linux x64 only)" OFF)
set(ONNXRUNTIME_ROOT "" CACHE PATH "Use an existing ONNX Runtime install instead of downloading one")

if(ONNXRUNTIME_ROOT)
  # e.g. a source build on Jetson with CUDA/TensorRT enabled
  find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS
      ${ONNXRUNTIME_ROOT}/include
      ${ONNXRUNTIME_ROOT}/include/onnxruntime
      ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session
    NO_DEFAULT_PATH REQUIRED)
  set(ORT_LIB_DIR ${ONNXRUNTIME_ROOT}/lib)
  set(ORT_LIB_NAME libonnxruntime.so)
  set(ORT_LICENSE_DIR ${ONNXRUNTIME_ROOT})
  message(STATUS "Using ONNX Runtime from ${ONNXRUNTIME_ROOT}")
else()
  if(WIN32)
    set(ORT_RID win-x64)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ORT_RID linux-arm64)
  else()
    set(ORT_RID linux-x64)
  endif()

  if(ONNXRUNTIME_GPU AND NOT ORT_RID STREQUAL "linux-x64")
    message(FATAL_ERROR "ONNXRUNTIME_GPU is only available for linux-x64. Use ONNXRUNTIME_ROOT on other platforms.")
  endif()

  set(ORT_DIR ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime-${ORT_VERSION})

  # Headers and license files come from the CPU package even for GPU builds.
  file(DOWNLOAD
    https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/${ORT_VERSION}
    ${ORT_DIR}/cpu.nupkg
    EXPECTED_HASH SHA512=${ORT_CPU_SHA512}
    INACTIVITY_TIMEOUT 60
    SHOW_PROGRESS)
  file(ARCHIVE_EXTRACT INPUT ${ORT_DIR}/cpu.nupkg DESTINATION ${ORT_DIR}/cpu
    PATTERNS build/native/include runtimes/${ORT_RID} LICENSE ThirdPartyNotices.txt)

  set(ORT_INCLUDE_DIR ${ORT_DIR}/cpu/build/native/include)
  set(ORT_LIB_DIR ${ORT_DIR}/cpu/runtimes/${ORT_RID}/native)
  set(ORT_LICENSE_DIR ${ORT_DIR}/cpu)

  if(ONNXRUNTIME_GPU)
    file(DOWNLOAD
      https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.Gpu.Linux/${ORT_VERSION}
      ${ORT_DIR}/gpu.nupkg
      EXPECTED_HASH SHA512=${ORT_GPU_SHA512}
      INACTIVITY_TIMEOUT 60
      SHOW_PROGRESS)
    file(ARCHIVE_EXTRACT INPUT ${ORT_DIR}/gpu.nupkg DESTINATION ${ORT_DIR}/gpu
      PATTERNS runtimes/linux-x64)
    set(ORT_LIB_DIR ${ORT_DIR}/gpu/runtimes/linux-x64/native)
  endif()

  if(NOT WIN32)
    # Install under the SONAME so the runtime loader finds it.
    file(RENAME ${ORT_LIB_DIR}/libonnxruntime.so ${ORT_LIB_DIR}/libonnxruntime.so.1)
    set(ORT_LIB_NAME libonnxruntime.so.1)
  endif()
endif()

install(DIRECTORY ${ORT_INCLUDE_DIR}/
  DESTINATION include
  FILES_MATCHING PATTERN "*.h")

if(WIN32)
  install(FILES ${ORT_LIB_DIR}/onnxruntime.dll ${ORT_LIB_DIR}/onnxruntime_providers_shared.dll DESTINATION bin)
  install(FILES ${ORT_LIB_DIR}/onnxruntime.lib DESTINATION lib)
else()
  file(GLOB ORT_LIBS ${ORT_LIB_DIR}/libonnxruntime*.so*)
  install(FILES ${ORT_LIBS} DESTINATION lib)
  ament_environment_hooks("${ament_cmake_package_templates_ENVIRONMENT_HOOK_LIBRARY_PATH}")
endif()

foreach(f LICENSE ThirdPartyNotices.txt)
  if(EXISTS ${ORT_LICENSE_DIR}/${f})
    install(FILES ${ORT_LICENSE_DIR}/${f} DESTINATION share/${PROJECT_NAME})
  endif()
endforeach()

ament_package(
  CONFIG_EXTRAS "src/onnxruntime_vendor-extras.cmake.in"
)
