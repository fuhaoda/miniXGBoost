set(sources base.cpp data.cpp ../include/tree.h)
include_directories(${PROJECT_SOURCE_DIR}/include)
add_library(xgboost STATIC ${sources})