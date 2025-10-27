file(GLOB LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.{cpp,cc}")
file(GLOB LIB_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.{hpp,h}")
add_library(lin_alg ${LIB_SOURCES} ${LIB_HEADERS})
target_include_directories(lin_alg
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

