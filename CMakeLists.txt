cmake_minimum_required(VERSION 3.20)
add_compile_options(-Wall -Wextra -Werror)
project(linear_algebra LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(lin_alg)
add_subdirectory(app)

