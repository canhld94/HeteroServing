
find_package(GTest REQUIRED)

add_executable(concurrent_queue_test concurrent_queue_test.cpp)
target_include_directories(concurrent_queue_test PRIVATE
                            ${GTEST_INCLUDE_DIRS})
target_link_libraries(concurrent_queue_test PRIVATE
                        ${GTEST_LIBRARIES}
                        pthread)

add_test(NAME QueueTest COMMAND concurrent_queue_test)