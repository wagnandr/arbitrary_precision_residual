cmake_minimum_required(VERSION 3.10)

project(HighprecisionCSR)

# we use eigen for vectors:
add_subdirectory(external/eigen)
# we reuse the cmake modules from eigen to use MPFR
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/external/eigen/cmake)
# find the MPFR Library
find_package(MPFR REQUIRED)

# our library:
add_library(HighprecisionCSR SHARED src/highprecision_csr.cpp)
target_include_directories(HighprecisionCSR PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(HighprecisionCSR PUBLIC external/eigen)
target_include_directories(HighprecisionCSR PUBLIC external/mpreal)
target_link_libraries(HighprecisionCSR PUBLIC ${MPFR_LIBRARIES})

# executable:
add_executable(HighPrecisionCSRMain src/main.cpp)
target_link_libraries(HighPrecisionCSRMain PUBLIC HighprecisionCSR)

# python support
add_subdirectory(${CMAKE_SOURCE_DIR}/external/pybind11)
add_subdirectory(${CMAKE_SOURCE_DIR}/python)