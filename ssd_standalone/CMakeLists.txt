# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2020 canhld@kaist.ac.kr
# SPDX-License-Identifier: Apache-2.0
#
set(TARGET_NAME "ssd_standalone")

# Find OpenCV components if exist
find_package(InferenceEngine 1.6 REQUIRED)
find_package(OpenCV COMPONENTS highgui videoio imgproc REQUIRED)
find_package(GFLAGS REQUIRED)

file (GLOB MAIN_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        )

file (GLOB MAIN_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        )

source_group("src" FILES ${MAIN_SRC})
source_group("include" FILES ${MAIN_HEADERS})

link_directories(${LIB_FOLDER})

add_executable(${TARGET_NAME} ${MAIN_SRC} ${MAIN_HEADERS})

add_dependencies(${TARGET_NAME} format_reader ie_cpu_extension)

set_target_properties(${TARGET_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE"

COMPILE_PDB_NAME ${TARGET_NAME})

target_include_directories(${TARGET_NAME} PRIVATE ${InferenceEngine_INCLUDE_DIRS})

target_link_libraries(${TARGET_NAME} IE::ie_cpu_extension ${InferenceEngine_LIBRARIES} ${GFLAGS_LIBRARIES} ${OpenCV_LIBRARIES})

if(UNIX)
    target_link_libraries(${TARGET_NAME} ${LIB_DL} pthread)
endif()