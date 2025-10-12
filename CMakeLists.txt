cmake_minimum_required(VERSION 4.0.3)
project(matrix_multiply)
set(CMAKE_CXX_STANDARD 20)
add_executable(matrix_multiply
    src/main.cpp
    src/matrix.cpp)
target_include_directories(matrix_multiply PRIVATE src)
