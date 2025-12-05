# Test executable
add_executable(forge_tests
    # Test source files
    test_derivatives_gtest.cpp
    test_derivatives_gtest_avx2.cpp
    # Test header files (for VS visibility)
    test_functions_1d.hpp
    test_functions_2d.hpp
    test_functions_multi_io.hpp
    test_graph_base.hpp
)

# Link with Google Test and our library
target_link_libraries(forge_tests
    PRIVATE
    forge
    gtest_main
    gtest
)

# Set properties for better debugging and source location
if(MSVC)
    set_target_properties(forge_tests PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        COMPILE_PDB_NAME "forge_tests"
        COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # Ensure test executable gets full debug info
    target_compile_options(forge_tests PRIVATE /Zi /FS /GL-)
    target_link_options(forge_tests PRIVATE /DEBUG:FULL /INCREMENTAL:NO)
endif()

# Register tests with CTest using better discovery
include(GoogleTest)

# Use PRE_TEST discovery mode for better source mapping
# This discovers tests at test time rather than build time
gtest_discover_tests(forge_tests
    DISCOVERY_MODE PRE_TEST
    PROPERTIES
        LABELS "Unit"
        TIMEOUT 30
    WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

# Add a custom target to run tests
add_custom_target(run_unit_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS forge_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running unit tests"
)