# This file is part of Forge <https://github.com/da-roth/forge>
#
# See LICENSE.md for license and copyright information
# SPDX-License-Identifier: Zlib

# Main build configuration for Forge - a high-performance JIT compiler and
# automatic differentiation library for mathematical expressions.
#
# Dependencies (auto-fetched): AsmJit, nlohmann/json, SLEEF, GoogleTest
#
# Architecture:
# The library is split into OBJECT libraries to support per-target compiler flags:
# - forge_core: Graph, optimizer, compiler core (no SIMD flags)
# - forge_x86_common: x86 common code (no SIMD flags)
# - forge_x86_double_avx2: AVX2 instruction set implementation (-mavx2)

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

# Backend bundling options
option(FORGE_BUNDLE_AVX2 "Bundle AVX2 backend into the library (can still be loaded at runtime if OFF)" ON)

if(MSVC AND FORGE_USE_STATIC_RUNTIME)
    # Set static runtime for all targets in this project
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# =============================================================================
# Compiler Flags - Split by instruction set requirements
# =============================================================================

# Base compile options (no SIMD)
if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(FORGE_BASE_COMPILE_OPTIONS
        /W4 /permissive-
        /MP
        /Zi /FS
        /GL-
        /wd4996 /wd4100 /wd4189 /wd4456 /wd4702 /wd4244 /wd4267
    )
    set(FORGE_AVX2_COMPILE_OPTIONS
        ${FORGE_BASE_COMPILE_OPTIONS}
        /arch:AVX2
    )
    set(FORGE_LINK_OPTIONS /DEBUG:FULL /INCREMENTAL:NO)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND MSVC)
    set(FORGE_BASE_COMPILE_OPTIONS
        /W4 /permissive-
        /Zi
        /wd4996 /wd4100 /wd4189 /wd4456 /wd4702 /wd4244 /wd4267
    )
    set(FORGE_AVX2_COMPILE_OPTIONS
        ${FORGE_BASE_COMPILE_OPTIONS}
        /arch:AVX2
    )
    set(FORGE_LINK_OPTIONS /DEBUG:FULL)
else()
    set(FORGE_BASE_COMPILE_OPTIONS
        -Wall -Wextra -Wpedantic
        -g
        -fno-lto
    )
    set(FORGE_AVX2_COMPILE_OPTIONS
        ${FORGE_BASE_COMPILE_OPTIONS}
        -mavx2
    )
    set(FORGE_LINK_OPTIONS -fno-lto)
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

# =============================================================================
# Dependencies
# =============================================================================

include(FetchContent)

# AsmJit dependency
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

# Common include directories for all OBJECT libraries
set(FORGE_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/api
    ${CMAKE_CURRENT_SOURCE_DIR}/tools
    ${CMAKE_CURRENT_SOURCE_DIR}/backends
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${asmjit_SOURCE_DIR}/src
    ${sleef_BINARY_DIR}/include
)

# =============================================================================
# OBJECT Libraries - Each with appropriate compiler flags
# =============================================================================

# -----------------------------------------------------------------------------
# forge_core: Graph, optimizer, native API (no SIMD intrinsics)
# -----------------------------------------------------------------------------
add_library(forge_core OBJECT
    # Native API types (operator-overloading based graph recording)
    api/native/fdouble.cpp
    api/native/fbool.cpp
    api/native/fint.cpp

    # Graph/Recording system
    src/graph/graph.cpp
    src/graph/graph_recorder.cpp
    src/graph/graph_optimizer.cpp

    # Graph optimization implementations
    src/graph/optimizations/inactive_folding.cpp
    src/graph/optimizations/common_subexpression_elimination.cpp
    src/graph/optimizations/algebraic_simplification.cpp
    src/graph/optimizations/stability_cleaning.cpp
    src/graph/optimizations/constant_cleanup.cpp

    # Graph serialization tools
    tools/graphSerialization/graph_serialization.cpp

    # Compiler core (no SIMD intrinsics used)
    src/compiler/forge_engine.cpp
    src/compiler/forward_stitcher.cpp
    src/compiler/gradient_stitcher.cpp
    src/compiler/runtime_trace.cpp
)
target_include_directories(forge_core PUBLIC ${FORGE_INCLUDE_DIRS})
target_compile_options(forge_core PRIVATE ${FORGE_BASE_COMPILE_OPTIONS})
target_link_libraries(forge_core PUBLIC asmjit nlohmann_json::nlohmann_json)
if(FORGE_BUNDLE_AVX2)
    target_compile_definitions(forge_core PRIVATE FORGE_BUNDLE_AVX2)
endif()

# -----------------------------------------------------------------------------
# forge_x86_common: x86 common code, register allocator, buffer factory (no SIMD)
# Note: node_value_buffer_factory.cpp has #ifdef guards for AVX2
# -----------------------------------------------------------------------------
add_library(forge_x86_common OBJECT
    src/compiler/x86/common/x86_instruction_set_base.cpp
    src/compiler/x86/common/node_value_buffer_factory.cpp
)
target_include_directories(forge_x86_common PUBLIC ${FORGE_INCLUDE_DIRS})
target_compile_options(forge_x86_common PRIVATE ${FORGE_BASE_COMPILE_OPTIONS})
target_link_libraries(forge_x86_common PUBLIC asmjit)
if(FORGE_BUNDLE_AVX2)
    target_compile_definitions(forge_x86_common PRIVATE FORGE_BUNDLE_AVX2)
endif()

# -----------------------------------------------------------------------------
# forge_x86_double_avx2: AVX2 instruction set implementation (-mavx2)
# Only built when FORGE_BUNDLE_AVX2 is ON
# -----------------------------------------------------------------------------
if(FORGE_BUNDLE_AVX2)
    add_library(forge_x86_double_avx2 OBJECT
        backends/double/avx2/avx2_instruction_set.cpp
        backends/double/avx2/avx2_static_registration.cpp
    )
    target_include_directories(forge_x86_double_avx2 PUBLIC ${FORGE_INCLUDE_DIRS})
    target_compile_options(forge_x86_double_avx2 PRIVATE ${FORGE_AVX2_COMPILE_OPTIONS})
    target_compile_definitions(forge_x86_double_avx2 PUBLIC SLEEF_STATIC_LIBS FORGE_BUNDLE_AVX2)
    target_link_libraries(forge_x86_double_avx2 PUBLIC asmjit sleefavx2 sleef)
endif()

# -----------------------------------------------------------------------------
# forge_avx2 (optional): Loadable AVX2 backend shared library
# Build with: cmake -DFORGE_BUILD_AVX2_BACKEND=ON ...
# Use with: InstructionSetFactory::loadBackend("./libforge_avx2.so")
# -----------------------------------------------------------------------------
option(FORGE_BUILD_AVX2_BACKEND "Build AVX2 as a loadable shared library" OFF)

if(FORGE_BUILD_AVX2_BACKEND)
    add_library(forge_avx2 SHARED
        # AVX2-specific implementation
        backends/double/avx2/avx2_instruction_set.cpp
        backends/double/avx2/avx2_backend.cpp
        # Required dependencies (must be self-contained for dynamic loading)
        src/compiler/x86/common/x86_instruction_set_base.cpp
        src/compiler/runtime_trace.cpp
        # Minimal registration support for loadable backends
        backends/double/avx2/avx2_backend_registration.cpp
    )
    target_include_directories(forge_avx2 PRIVATE ${FORGE_INCLUDE_DIRS})
    target_compile_options(forge_avx2 PRIVATE ${FORGE_AVX2_COMPILE_OPTIONS})
    target_compile_definitions(forge_avx2 PRIVATE SLEEF_STATIC_LIBS FORGE_LOADABLE_BACKEND)
    target_link_libraries(forge_avx2 PRIVATE asmjit sleefavx2 sleef)
    set_target_properties(forge_avx2 PROPERTIES
        OUTPUT_NAME "forge_avx2"
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
endif()

# =============================================================================
# Main Forge Library - Combines all OBJECT libraries
# =============================================================================

if(FORGE_BUNDLE_AVX2)
    add_library(forge STATIC
        $<TARGET_OBJECTS:forge_core>
        $<TARGET_OBJECTS:forge_x86_common>
        $<TARGET_OBJECTS:forge_x86_double_avx2>
    )
    # Link dependencies (with SLEEF AVX2)
    target_link_libraries(forge PUBLIC asmjit nlohmann_json::nlohmann_json sleefavx2 sleef)
    target_compile_definitions(forge PUBLIC FORGE_BUNDLE_AVX2)
else()
    add_library(forge STATIC
        $<TARGET_OBJECTS:forge_core>
        $<TARGET_OBJECTS:forge_x86_common>
    )
    # Link dependencies (without SLEEF AVX2)
    target_link_libraries(forge PUBLIC asmjit nlohmann_json::nlohmann_json sleef)
endif()

# Include directories for consumers
target_include_directories(forge PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/api
    ${CMAKE_CURRENT_SOURCE_DIR}/tools
    ${asmjit_SOURCE_DIR}/src
    ${sleef_BINARY_DIR}/include
)

# Link options (applied to final library)
target_link_options(forge PRIVATE ${FORGE_LINK_OPTIONS})

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

# =============================================================================
# Tests and Examples
# =============================================================================

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
message(STATUS "Forge architecture: x86/double (scalar + AVX2)")
