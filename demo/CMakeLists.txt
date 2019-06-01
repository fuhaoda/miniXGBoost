add_executable(demo EXCLUDE_FROM_ALL main.cpp)
include_directories(${PROJECT_SOURCE_DIR}/include)
target_link_libraries(demo minixgboost)

add_custom_command(
        TARGET demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/demo/config
        ${CMAKE_CURRENT_BINARY_DIR}/config
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/demo/train.txt
        ${CMAKE_CURRENT_BINARY_DIR}/train.txt
)
