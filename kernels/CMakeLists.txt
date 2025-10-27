find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

file(GLOB LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.{cu}")
file(GLOB LIB_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.{cuh}")
add_library(kernels ${LIB_SOURCES} ${LIB_HEADERS})
set_target_properties(kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(kernels PRIVATE ${CMAKE_CUDA_RUNTIME_INCLUDE_DIRECTORIES})
target_link_libraries(kernels PRIVATE CUDA::cudart)

