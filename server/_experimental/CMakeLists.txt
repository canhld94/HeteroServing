# Copyright (C) 2020 canhld@kaist.ac.kr
# SPDX-License-Identifier: Apache-2.0
#
add_executable(fcn16s fcn16s.cpp)
add_executable(reactor_server st_server_reactor.cpp)

set_target_properties(fcn16s PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE")
set_target_properties(reactor_server PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE")

add_dependencies(reactor_server ie_cpu_extension)
add_dependencies(fcn16s ie_cpu_extension)

target_link_libraries(fcn16s 
                        IE::ie_cpu_extension
                        ${CONAN_LIBS}
                        ${OpenCV_LIBRARIES}
                        ${InferenceEngine_LIBRARIES})

target_link_libraries(reactor_server 
                        IE::ie_cpu_extension
                        ${CONAN_LIBS}
                        ${OpenCV_LIBRARIES}
                        ${InferenceEngine_LIBRARIES}
                        ${CUDA_LIBRARIES}
                        ${TensorRT_LIBRARIES})

install(TARGETS fcn16s reactor_server
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION bin/lib
        ARCHIVE DESTINATION bin/lib
)

if(UNIX)
    target_link_libraries(fcn16s ${LIB_DL} pthread)
    target_link_libraries(reactor_server ${LIB_DL} pthread)
endif()