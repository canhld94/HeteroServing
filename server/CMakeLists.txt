
# These examples require Boost Beast
find_package(OpenCV REQUIRED)
find_package(InferenceEngine 1.6 REQUIRED)

include_directories(libs)
add_subdirectory(parallel)