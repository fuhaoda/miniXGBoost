FILE(GLOB sources *.cpp)
include_directories(${PROJECT_SOURCE_DIR}/include)
add_library(miniXGBoost STATIC ${sources} gbTreeEvaluator.cpp gbTreeEvaluator.h)
#install(TARGETS miniXGBoost DESTINATION lib)