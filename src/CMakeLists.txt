set(sources base.cpp data.cpp tree.cpp ../include/lossFunction.h)
include_directories(${PROJECT_SOURCE_DIR}/include)
add_library(xgboost STATIC ${sources})