# Copyright (C) 2020 canhld@kaist.ac.kr
# SPDX-License-Identifier: Apache-2.0
#

# This testing server use a modified version of beast for better opitmization
link_directories(${LIB_FOLDER})

add_executable(fcn16s fcn16s.cpp)

set_target_properties(fcn16s PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE")
set_target_properties(fcn16s PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
add_dependencies(fcn16s ie_cpu_extension)

target_include_directories(fcn16s PRIVATE 
                            ${InferenceEngine_INCLUDE_DIRS} 
                            ${OpenCV_INCLUDE_DIRS})

target_link_libraries(fcn16s 
                        IE::ie_cpu_extension
                        ${CONAN_LIBS}
                        ${OpenCV_LIBRARIES}
                        ${InferenceEngine_LIBRARIES})

install(TARGETS fcn16s
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION bin/lib
        ARCHIVE DESTINATION bin/lib
)

if(UNIX)
    target_link_libraries(fcn16s ${LIB_DL} pthread)
endif()