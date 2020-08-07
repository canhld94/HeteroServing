# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "ssd_inference_server")



file (GLOB MAIN_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        )

file (GLOB MAIN_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        )

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${MAIN_SRC})
source_group("include" FILES ${MAIN_HEADERS})

link_directories(${LIB_FOLDER})

# Create library file from sources.
add_executable(${TARGET_NAME} ${MAIN_SRC} ${MAIN_HEADERS})

add_dependencies(${TARGET_NAME} format_reader ie_cpu_extension)

set_target_properties(${TARGET_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE"
COMPILE_PDB_NAME ${TARGET_NAME})

target_include_directories(${TARGET_NAME} PRIVATE 
                            ${InferenceEngine_INCLUDE_DIRS} 
                            ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${TARGET_NAME} 
                    IE::ie_cpu_extension
                    ${OpenCV_LIBRARIES}
                    ${InferenceEngine_LIBRARIES}
                    ${CONAN_LIBS})

install(TARGETS ${TARGET_NAME} ie_cpu_extension
                RUNTIME DESTINATION bin
                LIBRARY DESTINATION bin/lib
                ARCHIVE DESTINATION bin/lib
        )
if(UNIX)
    target_link_libraries( ${TARGET_NAME} ${LIB_DL} pthread)
endif()
