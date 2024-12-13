cmake_minimum_required(VERSION 3.15)
project(example LANGUAGES CXX)

project(ArbitraryPrecisionResidual)

# we use eigen for vectors:
add_subdirectory(external/eigen)
# we reuse the cmake modules from eigen to use MPFR
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/external/eigen/cmake)
# find the MPFR Library
find_package(MPFR REQUIRED)

# our library:
add_library(ArbitraryPrecisionResidual STATIC src/arbitrary_precision_residual.cpp)
target_include_directories(ArbitraryPrecisionResidual PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(ArbitraryPrecisionResidual PUBLIC external/eigen)
target_include_directories(ArbitraryPrecisionResidual PUBLIC external/mpreal)
target_link_libraries(ArbitraryPrecisionResidual PUBLIC ${MPFR_LIBRARIES})
set_target_properties(ArbitraryPrecisionResidual PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

# executable:
add_executable(ArbitraryPrecisionMain src/main.cpp)
target_link_libraries(ArbitraryPrecisionMain PUBLIC ArbitraryPrecisionResidual)

# python support
add_subdirectory(${CMAKE_SOURCE_DIR}/external/pybind11)
add_subdirectory(${CMAKE_SOURCE_DIR}/python)