add_executable(concurrent_queue_test concurrent_queue_test.cpp)

target_link_libraries(concurrent_queue_test PRIVATE
                      ${CONAN_LIBS}
                      pthread)

add_test(NAME QueueTest COMMAND concurrent_queue_test)