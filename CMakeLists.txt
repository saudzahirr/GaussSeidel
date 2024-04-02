cmake_minimum_required(VERSION 3.22)
project(GaussSeidel)

find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)

add_executable(${PROJECT_NAME} gauss_seidel.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} fmt::fmt)
