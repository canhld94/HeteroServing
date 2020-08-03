# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2020 canhld@kaist.ac.kr
# SPDX-License-Identifier: Apache-2.0
#
cmake_minimum_required (VERSION 3.13)

project(mec_fpga_demo)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)

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
set (CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${PROJECT_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
set (CMAKE_PDB_OUTPUT_DIRECTORY ${PROJECT_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
set (LIBRARY_OUTPUT_DIRECTORY ${PROJECT_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
set (LIBRARY_OUTPUT_PATH ${LIBRARY_OUTPUT_DIRECTORY}/lib)


set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror") #treating warnings as errors
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized -Winit-self")

####################################
## to use C++11
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

include_directories (
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${CMAKE_CURRENT_SOURCE_DIR}/ncl
)

# Install to root
set (CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})
# Project components
add_subdirectory(format_reader) # helper library @Intel
# add_subdirectory(ssd_standalone) # SSD demonstration standalone
# add_subdirectory(ssd_inference_server) # SSD inference server (release)
add_subdirectory(client_server_beast) # development http server folder

# Testing
# add_subdirectory(test)
# enable_testing()

if (UNIX)
    set (LIB_DL dl)
endif()