cmake_minimum_required(VERSION 4.0.3)
project(matrix_multiply)
set(CMAKE_CXX_STANDARD 20)

# Include Conan-generated toolchain
include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake)

# Source files
file(GLOB SOURCES "src/*.cpp")
add_executable(matrix_multiply ${SOURCES})
target_include_directories(matrix_multiply PRIVATE src)

# Enable testing
enable_testing()

# Add test executable
file(GLOB TEST_SOURCES "tests/*.cpp")
add_executable(matrix_multiply_tests ${TEST_SOURCES})
target_link_libraries(matrix_multiply_tests PRIVATE CONAN_PKG::gtest CONAN_PKG::gtest_main matrix_multiply)
target_include_directories(matrix_multiply_tests PRIVATE src)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(matrix_multiply_tests)

