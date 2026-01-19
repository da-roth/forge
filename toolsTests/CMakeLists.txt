cmake_minimum_required(VERSION 3.20)

# Collect test sources in this directory
file(GLOB TOOL_TEST_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
)

add_executable(forge_tools_tests
    ${TOOL_TEST_SOURCES}
)

target_link_libraries(forge_tools_tests PRIVATE
    forge
    gtest
    gtest_main
)

# Pass FORGE_BUNDLE_AVX2 to tools tests if AVX2 is bundled
if(FORGE_BUNDLE_AVX2)
    target_compile_definitions(forge_tools_tests PRIVATE FORGE_BUNDLE_AVX2)
endif()

# Link pthread only on non-Windows platforms
if(NOT WIN32)
    target_link_libraries(forge_tools_tests PRIVATE pthread)
endif()

target_include_directories(forge_tools_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../tools
)

target_compile_features(forge_tools_tests PRIVATE cxx_std_17)

add_test(NAME forge_tools_tests COMMAND forge_tools_tests)


