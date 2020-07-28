# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2020 canhld@kaist.ac.kr
# SPDX-License-Identifier: Apache-2.0
#

link_directories(${LIB_FOLDER})

add_executable(http_server http_server.cpp)

add_executable(http_client http_client.cpp)

set_target_properties(http_server PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE")
set_target_properties(http_client PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
add_dependencies(http_server format_reader ie_cpu_extension)

target_include_directories(http_server PRIVATE 
                            ${InferenceEngine_INCLUDE_DIRS} 
                            ${Boost_INCLUDE_DIRS}
                            ${OpenCV_INCLUDE_DIRS}
                            ${GFLAGS_INCLUDE_DIRS})
target_include_directories(http_client PRIVATE ${Boost_INCLUDE_DIRS})

target_link_libraries(http_server 
                        IE::ie_cpu_extension
                        ${Boost_LIBRARIES} 
                        ${OpenCV_LIBRARIES}
                        ${InferenceEngine_LIBRARIES}
                        ${GFLAGS_LIBRARIES})
target_link_libraries(http_client 
                        ${Boost_LIBRARIES})

install(TARGETS http_client http_server
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION bin/lib
        ARCHIVE DESTINATION bin/lib
)

if(UNIX)
    target_link_libraries(http_client ${LIB_DL} pthread)
    target_link_libraries(http_server ${LIB_DL} pthread)
endif()