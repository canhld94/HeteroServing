
# Generate grpc stubs for client
find_program(_PROTOBUF_PROTOC protoc)
find_program(_GRPC_PY_PLUGIN_EXECUTABLE grpc_python_plugin)
get_filename_component(proto "../server/libs/stubs/inference_rpc.proto" ABSOLUTE)
get_filename_component(proto_path ${proto} PATH)
set(stub_path "grpc/")
execute_process(COMMAND ${_PROTOBUF_PROTOC}
                --grpc_out ${stub_path}
                --python_out ${stub_path}
                --plugin=protoc-gen-grpc=${_GRPC_PY_PLUGIN_EXECUTABLE}
                -I ${proto_path}
                ${proto}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GRPC_RESULT)