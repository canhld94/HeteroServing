# Copyright (C) 2020 canhld@kaist.ac.kr
# SPDX-License-Identifier: Apache-2.0
#
cmake_minimum_required (VERSION 3.13)

project(something)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if (CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()

if (NOT(BIN_FOLDER))
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7l")
        set (ARCH armv7l)
    elseif("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
        set (ARCH intel64)
    else()
        set (ARCH ia32)
    endif()

    set (BIN_FOLDER ${ARCH})
endif()

if (NOT(PROJECT_DIR))
    # in case if samples are built out of IE repo
    set (PROJECT_DIR ${CMAKE_CURRENT_BINARY_DIR})
endif()

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
set (LIBRARY_OUTPUT_DIRECTORY ${PROJECT_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
set (LIBRARY_OUTPUT_PATH ${LIBRARY_OUTPUT_DIRECTORY}/lib)


set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-error=deprecated-declarations -Wno-error=array-bounds") #treating warnings as errors

###################################### to use C++11
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
if (CMAKE_CXX_COMPILER_ID STREQUAL GNU)
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8)
        message(FATAL_ERROR "GCC version must be at least 4.8!")
    endif()
    set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
endif()
####################################

if (CMAKE_CXX_COMPILER_ID STREQUAL GNU)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

# Package manager
# OpenCV and Inference Engine come with openvino
if (INTEL_OPENVINO_DIR STREQUAL "")
    set(INTEL_OPENVINO_DIR $ENV{INTEL_OPENVINO_DIR})
endif()
if (NOT INTEL_OPENVINO_DIR STREQUAL "")
    message(STATUS "Looking for inference engine and opencv in ${INTEL_OPENVINO_DIR}")
    list(APPEND CMAKE_PREFIX_PATH ${INTEL_OPENVINO_DIR})
endif()
find_package(OpenCV REQUIRED)
find_package(InferenceEngine REQUIRED)
# CUDA
if (CUDA_INSTALL_DIR STREQUAL "")
    set(CUDA_INSTALL_DIR $ENV{CUDA_INSTALL_DIR})
endif()
if (NOT CUDA_INSTALL_DIR STREQUAL "")
    message(STATUS "Looking for CUDA in ${CUDA_INSTALL_DIR}")
    list(APPEND CMAKE_PREFIX_PATH ${CUDA_INSTALL_DIR})
endif()
find_package(CUDA REQUIRED)
# TensorRT
if (TensorRT_ROOT STREQUAL "")
    set(TensorRT_ROOT $ENV{TensorRT_ROOT})
endif()
if (NOT TensorRT_ROOT STREQUAL "")
    message(STATUS "Looking for TensorRT in ${TensorRT_ROOT}")
    list(APPEND CMAKE_PREFIX_PATH ${TensorRT_ROOT})
endif()
find_package(TensorRT REQUIRED)

# clone all submodules

# GRPC & protobuf
set(Protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")
set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)

find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")
set(_REFLECTION gRPC::grpc++_reflection)
set(_GRPC_GRPCPP gRPC::grpc++)

# Other libraries: boost, gflags, gtest, spdlog is managed by conan
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

# Project components
add_subdirectory(server)
add_subdirectory(client)

# Install to root
set (CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS ie_cpu_extension
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION bin
        ARCHIVE DESTINATION bin
)

# Testing
# add_subdirectory(test)
# enable_testing()

if (UNIX)
    set (LIB_DL dl)
endif()