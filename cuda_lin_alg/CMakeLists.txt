find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

file(GLOB LIB_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")
file(GLOB LIB_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh")
add_compile_options(-Wall -Wextra)
add_library(cuda_lin_alg ${LIB_SOURCES} ${LIB_HEADERS})
set_target_properties(cuda_lin_alg PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(cuda_lin_alg
    PRIVATE lin_alg
    PRIVATE CUDA::cudart)

