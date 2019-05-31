FILE(GLOB sources *.cpp)
include_directories(${PROJECT_SOURCE_DIR}/include)
add_library(minixgboost STATIC ${sources})
install(TARGETS minixgboost DESTINATION lib)
