set(sources base.cpp data.cpp loadData.cpp)
include_directories(${PROJECT_SOURCE_DIR}/include)
add_library(xgboost STATIC ${sources})