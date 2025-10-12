find_package(GTest CONFIG REQUIRED)

file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(lin_alg_tests ${TEST_SOURCES})

target_link_libraries(lin_alg_tests
    PRIVATE
        lin_alg
        GTest::GTest
        GTest::Main
)

target_include_directories(lin_alg_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/lin_alg/src
)

include(GoogleTest)
gtest_discover_tests(lin_alg_tests)

