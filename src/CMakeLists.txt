set(sources "config.cpp" "gbm.cpp" "io.cpp" "matrix.cpp") # "tree.cpp")
include_directories(${PROJECT_SOURCE_DIR}/include)
add_library(minixgboost STATIC ${sources})
install(TARGETS minixgboost DESTINATION lib)


#set(sources base.cpp data.cpp tree.cpp ../include/lossFunction.h)
#include_directories(${PROJECT_SOURCE_DIR}/include)
#add_library(xgboost STATIC ${sources})
