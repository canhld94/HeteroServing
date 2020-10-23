# Copyright (C) 2020 canhld@kaist.ac.kr
# SPDX-License-Identifier: Apache-2.0
#

add_executable(serving ${cpp_proto_srcs} ${cpp_grpc_srcs} st_serving_server.cpp)

set_target_properties(serving PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE")

add_dependencies(serving ie_cpu_extension)

target_link_libraries(serving 
                        IE::ie_cpu_extension
                        ${CONAN_LIBS}
                        ${OpenCV_LIBRARIES}
                        ${InferenceEngine_LIBRARIES}
                        ${CUDA_LIBRARIES}
                        ${TensorRT_LIBRARIES}
                        ${gRPC_LIBRARIES}
                        ${_REFLECTION}
                        ${_GRPC_GRPCPP}
                        ${_PROTOBUF_LIBPROTOBUF})

install(TARGETS serving
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION bin/lib
        ARCHIVE DESTINATION bin/lib
)

if(UNIX)
    target_link_libraries(serving ${LIB_DL} pthread)
endif()