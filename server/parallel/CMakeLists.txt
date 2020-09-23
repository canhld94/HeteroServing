# Copyright (C) 2020 canhld@kaist.ac.kr
# SPDX-License-Identifier: Apache-2.0
#

add_executable(parallel_server st_server_parallel.cpp)

set_target_properties(parallel_server PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE")

add_dependencies(parallel_server ie_cpu_extension)

target_link_libraries(parallel_server 
                        IE::ie_cpu_extension
                        ${CONAN_LIBS}
                        ${OpenCV_LIBRARIES}
                        ${InferenceEngine_LIBRARIES})

install(TARGETS parallel_server
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION bin/lib
        ARCHIVE DESTINATION bin/lib
)

if(UNIX)
    target_link_libraries(parallel_server ${LIB_DL} pthread)
endif()