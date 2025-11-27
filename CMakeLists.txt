# This file is part of Forge <https://github.com/da-roth/forge>
#
# See LICENSE.md for license and copyright information
# SPDX-License-Identifier: Zlib

# Main build configuration for Forge - a high-performance JIT compiler and
# automatic differentiation library for mathematical expressions.
#
# Dependencies (auto-fetched): AsmJit, nlohmann/json, SLEEF, GoogleTest

cmake_minimum_required(VERSION 3.20)
project(Forge VERSION 0.1.0 LANGUAGES CXX)

# Enable CTest for better test discovery
include(CTest)
enable_testing()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for agents/tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

# Runtime library option for MSVC
option(FORGE_USE_STATIC_RUNTIME "Use static runtime library (/MT) instead of dynamic (/MD)" ON)

if(MSVC AND FORGE_USE_STATIC_RUNTIME)
    # Set static runtime for all targets in this project
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# Compiler-specific settings
if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # MSVC-specific flags (not for clang-cl)
    add_compile_options(/W4 /permissive-)
    add_compile_options(/MP)
    add_compile_options(/Zi /FS)
    add_compile_options(/GL-)
    add_compile_options(/arch:AVX2)  # Enable AVX2 instructions and __AVX__ define for SLEEF
    add_link_options(/DEBUG:FULL /INCREMENTAL:NO)
    add_compile_options(/wd4996 /wd4100 /wd4189 /wd4456 /wd4702 /wd4244 /wd4267)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND MSVC)
    # clang-cl specific flags
    add_compile_options(/W4 /permissive-)
    add_compile_options(/Zi)
    add_compile_options(/arch:AVX2)  # Enable AVX2 instructions and __AVX__ define for SLEEF
    add_link_options(/DEBUG:FULL)
    add_compile_options(/wd4996 /wd4100 /wd4189 /wd4456 /wd4702 /wd4244 /wd4267)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    add_compile_options(-g)
    add_compile_options(-mavx2)  # Enable AVX2 instructions for SLEEF
    add_compile_options(-fno-lto)
    add_link_options(-fno-lto)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# PDB output directory for MSVC
if(MSVC)
    set(CMAKE_PDB_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

# Enable Position Independent Code for shared library linking
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# AsmJit dependency
include(FetchContent)

FetchContent_Declare(
    asmjit
    GIT_REPOSITORY https://github.com/asmjit/asmjit.git
    GIT_TAG a3199e8857792cd10b7589ff5d58343d2c9008ea  # Master pinned from Sep 5, 2025
)
set(ASMJIT_STATIC ON CACHE BOOL "Build AsmJit as static library" FORCE)
set(ASMJIT_BUILD_EXAMPLES OFF CACHE BOOL "Build AsmJit examples" FORCE)
set(ASMJIT_BUILD_TESTS OFF CACHE BOOL "Build AsmJit tests" FORCE)
FetchContent_MakeAvailable(asmjit)

# nlohmann/json for graph serialization
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(json)

# SLEEF for vectorized math functions
FetchContent_Declare(
    sleef
    GIT_REPOSITORY https://github.com/shibatch/sleef.git
    GIT_TAG 3.6.1
)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build SLEEF as static library" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "Build SLEEF tests" FORCE)
set(BUILD_DFT OFF CACHE BOOL "Build SLEEF DFT" FORCE)
set(BUILD_GNUABI_LIBS OFF CACHE BOOL "Build SLEEF GNUABI libs" FORCE)
set(BUILD_INLINE_HEADERS ON CACHE BOOL "Build SLEEF inline headers" FORCE)
set(SLEEF_BUILD_SCALAR_LIB OFF CACHE BOOL "Don't build scalar library" FORCE)
FetchContent_MakeAvailable(sleef)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/tools)

# Main Forge library
add_library(forge STATIC
    # Type system files
    tools/types/fdouble.cpp
    tools/types/fdouble.hpp
    tools/types/fbool.cpp
    tools/types/fbool.hpp
    tools/types/fint.cpp
    tools/types/fint.hpp

    # Graph/Recording system
    src/graph/graph.cpp
    src/graph/graph.hpp
    src/graph/graph_recorder.cpp
    src/graph/graph_recorder.hpp
    src/graph/handles.hpp

    # Graph serialization tools
    tools/graphSerialization/graph_serialization.cpp
    tools/graphSerialization/graph_serialization.hpp

    # Compiler/JIT implementation
    src/compiler/forge_engine.cpp
    src/compiler/forge_engine.hpp
    src/compiler/forward_stitcher.cpp
    src/compiler/forward_stitcher.hpp

    src/compiler/node_value_buffers/node_value_buffer_factory.cpp
    src/compiler/node_value_buffers/node_value_buffer.hpp
    src/compiler/node_value_buffers/scalar_node_value_buffer.hpp
    src/compiler/node_value_buffers/avx2_node_value_buffer.hpp
    src/compiler/register_allocator.cpp
    src/compiler/register_allocator.hpp
    src/compiler/gradient_stitcher.cpp
    src/compiler/gradient_stitcher.hpp
    src/compiler/compiler_config.hpp
    src/compiler/instruction_sets/x86_instruction_set_base.cpp
    src/compiler/instruction_sets/x86_instruction_set_base.hpp
    src/compiler/instruction_sets/avx2_instruction_set.cpp
    src/compiler/instruction_sets/avx2_instruction_set.hpp

    # Runtime tracing system
    src/compiler/runtime_trace.cpp
    src/compiler/runtime_trace.hpp
    src/compiler/instruction_tracer.hpp

    # Graph optimization
    src/graph/graph_optimizer.cpp
    src/graph/graph_optimizer.hpp
    src/graph/graph_debug.hpp

    # Graph optimization implementations
    src/graph/optimizations/inactive_folding.cpp
    src/graph/optimizations/inactive_folding.hpp
    src/graph/optimizations/common_subexpression_elimination.cpp
    src/graph/optimizations/common_subexpression_elimination.hpp
    src/graph/optimizations/algebraic_simplification.cpp
    src/graph/optimizations/algebraic_simplification.hpp
    src/graph/optimizations/stability_cleaning.cpp
    src/graph/optimizations/stability_cleaning.hpp
    src/graph/optimizations/placeholder_optimization.cpp
    src/graph/optimizations/constant_cleanup.cpp
    src/graph/optimizations/constant_cleanup.hpp
    src/graph/optimizations/placeholder_optimization.hpp
    src/graph/optimizations/optimizations.hpp
)

# Link AsmJit, nlohmann_json, and SLEEF to Forge library
# Note: We need both the AVX2-specific library and the main sleef library for helper functions
target_link_libraries(forge PUBLIC asmjit nlohmann_json::nlohmann_json sleefavx2 sleef)
target_include_directories(forge PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tools
    ${asmjit_SOURCE_DIR}/src
    ${sleef_BINARY_DIR}/include
)

# Define SLEEF_STATIC_LIBS for static linking on Windows
target_compile_definitions(forge PUBLIC SLEEF_STATIC_LIBS)

# Define FORGE_RELEASE_BUILD for Release builds to disable debug output
target_compile_definitions(forge PUBLIC $<$<CONFIG:Release>:FORGE_RELEASE_BUILD>)
target_compile_definitions(forge PUBLIC $<$<CONFIG:RelWithDebInfo>:FORGE_RELEASE_BUILD>)

# Enable Position Independent Code for shared library linking
set_target_properties(forge PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Create alias for external consumption
add_library(forge::forge ALIAS forge)

# Windows-specific settings for static libraries
if(WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)
endif()

# Add tests subdirectory if it exists (can be disabled by parent project)
if(NOT FORGE_SKIP_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    # Enable testing
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    # Match googletest runtime library to forge's runtime library setting
    if(FORGE_USE_STATIC_RUNTIME)
        set(gtest_force_shared_crt OFF CACHE BOOL "" FORCE)
    else()
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    endif()
    set(INSTALL_GTEST OFF CACHE BOOL "Install GoogleTest" FORCE)
    set(BUILD_GMOCK ON CACHE BOOL "Build GoogleMock" FORCE)
    set(gtest_build_tests OFF CACHE BOOL "Build GoogleTest tests" FORCE)
    set(gtest_build_samples OFF CACHE BOOL "Build GoogleTest samples" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_subdirectory(tests)

    # Add toolsTests subdirectory if it exists
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/toolsTests/CMakeLists.txt")
        add_subdirectory(toolsTests)
    endif()

endif()

# Add examples subdirectory if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt")
    add_subdirectory(examples)
endif()

# Print build configuration
message(STATUS "Forge build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
