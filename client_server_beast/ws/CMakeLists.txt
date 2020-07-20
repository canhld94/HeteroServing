# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2020 canhld@kaist.ac.kr
# SPDX-License-Identifier: Apache-2.0
#

link_directories(${LIB_FOLDER})

add_executable(ws_server ws_server.cpp)

add_executable(ws_client ws_client.cpp)

set_target_properties(ws_server PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE")
set_target_properties(ws_client PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE")

target_include_directories(ws_server PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(ws_client PRIVATE ${Boost_INCLUDE_DIRS})

target_link_libraries(ws_server ${Boost_LIBRARIES})
target_link_libraries(ws_client ${Boost_LIBRARIES})

install(TARGETS ws_client ws_server
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION bin/lib
        ARCHIVE DESTINATION bin/lib
)

if(UNIX)
    target_link_libraries(ws_client ${LIB_DL} pthread)
    target_link_libraries(ws_server ${LIB_DL} pthread)
endif()