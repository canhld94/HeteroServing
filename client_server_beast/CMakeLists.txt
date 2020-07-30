
# These examples require Boost Beast
find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GFLAGS REQUIRED)
find_package(InferenceEngine 1.6 REQUIRED)
add_subdirectory(ws)
add_subdirectory(http)