cmake_minimum_required(VERSION 3.12)
project(miniXGBoost)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

add_subdirectory(src)
add_subdirectory(include)
add_subdirectory(demo)

