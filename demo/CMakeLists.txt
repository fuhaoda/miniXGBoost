add_executable(demo main.cpp)
include_directories(${PROJECT_SOURCE_DIR}/include)
target_link_libraries(demo xgboost)