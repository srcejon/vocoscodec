cmake_minimum_required(VERSION 3.20)

project(vocoscodec VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

option(VOCOSCODEC_BUILD_RUNTIME
  "Build the dynamically loadable C API runtime (vocoscodec shared library)" ON)
option(VOCOSCODEC_BUILD_TESTS "Build vocoscodec tests" OFF)
option(VOCOSCODEC_BUILD_TOOLS "Build vocoscodec WAV tools and benchmark" OFF)
option(VOCOSCODEC_HALF_WEIGHTS
  "Hold the per-frame layers' weights in half precision (half the memory traffic)" ON)
set(VOCOSCODEC_RUNTIME_MODEL_DIR "${PROJECT_SOURCE_DIR}/model"
  CACHE PATH "Model directory reported by the vocoscodec runtime")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(VOCOSCODEC_BUILD_TESTS)
  enable_testing()
endif()
add_subdirectory(vocoscodec)

install(FILES README.md
  DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/vocoscodec")
