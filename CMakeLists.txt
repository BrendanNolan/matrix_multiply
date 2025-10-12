cmake_minimum_required(VERSION 3.20)
project(lin_alg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional Conan toolchain
include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake OPTIONAL)

enable_testing()

add_subdirectory(lin_alg)
add_subdirectory(app)
add_subdirectory(tests)

