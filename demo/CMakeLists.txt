add_executable(demo main.cpp ../include/io.h)
include_directories(${PROJECT_SOURCE_DIR}/include)
target_link_libraries(demo miniXGBoost)

#[[
add_custom_command(
       TARGET demo POST_BUILD
       COMMAND ${CMAKE_COMMAND} -E copy
       ${CMAKE_SOURCE_DIR}/demo/config
       ${CMAKE_CURRENT_BINARY_DIR}/config
       COMMAND ${CMAKE_COMMAND} -E copy
       ${CMAKE_SOURCE_DIR}/demo/train.txt
       ${CMAKE_CURRENT_BINARY_DIR}/train.txt
       COMMAND ${CMAKE_COMMAND} -E copy
       ${CMAKE_SOURCE_DIR}/demo/test_debug.txt
       ${CMAKE_CURRENT_BINARY_DIR}/test_debug.txt
)
]]