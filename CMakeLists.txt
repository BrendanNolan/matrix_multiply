cmake_minimum_required(VERSION 3.20)
add_compile_options(-Wall -Wextra -Werror)
project(linear_algebra LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional Conan toolchain
include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake OPTIONAL)

# Ensure CMake can find Conan-generated package configs
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}")
enable_testing()

add_subdirectory(lin_alg)
add_subdirectory(app)
add_subdirectory(tests)

