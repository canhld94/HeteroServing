# Include
include_directories(libs
                    ${CONAN_INCLUDE_DIRS}
                    ${CUDA_INCLUDE_DIRS}
                    ${TensorRT_INCLUDE_DIRS}
                    ${InferenceEngine_INCLUDE_DIRS} 
                    ${OpenCV_INCLUDE_DIRS})
add_subdirectory(parallel)
# add_subdirectory(reactor)