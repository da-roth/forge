cmake_minimum_required(VERSION 3.16)
project(forge VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(FORGE_BUILD_TESTS "Build Forge tests" OFF)
option(FORGE_BUILD_TOOLS "Build Forge tools" OFF)

# AsmJit dependency (required for instruction set interfaces)
include(FetchContent)
FetchContent_Declare(
    asmjit
    GIT_REPOSITORY https://github.com/asmjit/asmjit.git
    GIT_TAG a3199e8857792cd10b7589ff5d58343d2c9008ea  # Same version as TapePresso
)
set(ASMJIT_STATIC ON CACHE BOOL "Build AsmJit as static library" FORCE)
set(ASMJIT_BUILD_EXAMPLES OFF CACHE BOOL "Build AsmJit examples" FORCE)
set(ASMJIT_BUILD_TESTS OFF CACHE BOOL "Build AsmJit tests" FORCE)
FetchContent_MakeAvailable(asmjit)

# Create Forge library as STATIC library with source files
add_library(forge STATIC
    # Core components
    src/forge/core/computation_graph.cpp
    
    # X86 instruction sets
    src/forge/x86/x86_instruction_set_base.cpp
    src/forge/x86/runtime_trace.cpp
    src/forge/x86/avx2_instruction_set.cpp
    
    # Runtime components
    src/forge/runtime/node_buffer_factory.cpp
    
    # Compiler components
    src/forge/compiler/analysis/stability_cleaner.cpp
    src/forge/compiler/utils/compilation_timer.cpp
    src/forge/compiler/generators/constant_pool_manager.cpp
    src/forge/compiler/generators/register_utils.cpp
    src/forge/compiler/operations/arithmetic_operations.cpp
    src/forge/compiler/operations/math_functions.cpp
    src/forge/compiler/operations/comparison_control.cpp
    src/forge/compiler/operations/boolean_operations.cpp
    src/forge/compiler/operations/integer_operations.cpp
    src/forge/compiler/forward_compiler.cpp
    src/forge/compiler/reverse_gradient_compiler.cpp
    src/forge/compiler/forge_engine.cpp
    
    # Tools
    tools/graph_serialization_service.cpp
)

# Add include directories
target_include_directories(forge PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${asmjit_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

# Link AsmJit to forge
target_link_libraries(forge PUBLIC asmjit)

# Define export for installation
install(TARGETS forge
    EXPORT forgeTargets
    INCLUDES DESTINATION include
)

# Install headers
install(DIRECTORY src/forge
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Export targets
install(EXPORT forgeTargets
    FILE forgeConfig.cmake
    NAMESPACE forge::
    DESTINATION lib/cmake/forge
)

# Build examples if requested
option(FORGE_BUILD_EXAMPLES "Build Forge examples" ON)
if(FORGE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Create version file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/forgeConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/forgeConfigVersion.cmake"
    DESTINATION lib/cmake/forge
)

# Tests
if(FORGE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# Tools
if(FORGE_BUILD_TOOLS)
    add_subdirectory(tools)
endif()