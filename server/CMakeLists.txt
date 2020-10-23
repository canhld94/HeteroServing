# Include
include_directories(libs
                    ${CONAN_INCLUDE_DIRS}
                    ${CUDA_INCLUDE_DIRS}
                    ${TensorRT_INCLUDE_DIRS}
                    ${InferenceEngine_INCLUDE_DIRS} 
                    ${OpenCV_INCLUDE_DIRS})

# Generate grpc stubs for server
find_program(_PROTOBUF_PROTOC protoc)
find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
get_filename_component(proto "libs/stubs/inference_rpc.proto" ABSOLUTE)
get_filename_component(proto_path "${proto}" PATH)
set(cpp_proto_srcs "${proto_path}/inference_rpc.pb.cc")
set(cpp_proto_hdrs "${proto_path}/inference_rpc.pb.h")
set(cpp_grpc_srcs "${proto_path}/inference_rpc.grpc.pb.cc")
set(cpp_grpc_hdrs "${proto_path}/inference_rpc.grpc.pb.h")
execute_process(COMMAND ${_PROTOBUF_PROTOC}
                --grpc_out ${proto_path}
                --cpp_out ${proto_path}
                --plugin=protoc-gen-grpc=${_GRPC_CPP_PLUGIN_EXECUTABLE}
                -I ${proto_path}
                ${proto}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GRPC_RESULT)
add_subdirectory(apps)