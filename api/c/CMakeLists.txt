##############################################################################
#
# Forge C API - Stable ABI shared library
#
# This builds Forge as a shared library with a C interface for cross-compiler
# binary compatibility.
#
# Usage:
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   cmake --build .
#   cmake --install . --prefix /path/to/install
#
# The resulting package contains:
#   - lib/forge_capi.dll (or .so/.dylib)
#   - include/forge_c_api.h
#
# SPDX-License-Identifier: Zlib
#
##############################################################################

cmake_minimum_required(VERSION 3.16)
project(ForgeCAPI VERSION 0.1.0 LANGUAGES CXX C)

# Options
option(FORGE_CAPI_BUILD_TESTS "Build C API tests" ON)
option(FORGE_CAPI_USE_STATIC_RUNTIME "Use static runtime library (/MT) instead of dynamic (/MD) on MSVC" OFF)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Position independent code for shared library
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# On Windows, configure runtime library (static or dynamic)
if(MSVC)
    if(FORGE_CAPI_USE_STATIC_RUNTIME)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
        message(STATUS "Forge C API: Using static MSVC runtime (/MT, /MTd)")
    else()
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
        message(STATUS "Forge C API: Using dynamic MSVC runtime (/MD, /MDd)")
    endif()
    # Pass the runtime setting to Forge core (included as subdirectory)
    set(FORGE_USE_STATIC_RUNTIME ${FORGE_CAPI_USE_STATIC_RUNTIME} CACHE BOOL "" FORCE)
endif()

# Symbol visibility - export C++ symbols so loadable backends can link against them
# (Changed from 'hidden' to 'default' to allow backends to use X86InstructionSetBase etc.)
set(CMAKE_CXX_VISIBILITY_PRESET default)
set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF)

# Include directories
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

##############################################################################
# Find or build Forge
##############################################################################

# Use FetchContent cache to avoid re-downloading
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../.deps-cache" CACHE PATH "FetchContent cache")

# Note: api/c is two levels deep from forge root (api/c -> api -> forge)

# Disable tests for dependencies
set(SLEEF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FORGE_SKIP_TESTS ON CACHE BOOL "" FORCE)

# Add forge as subdirectory (go up from api/c to forge root)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. ${CMAKE_CURRENT_BINARY_DIR}/forge-build EXCLUDE_FROM_ALL)

##############################################################################
# Build shared library
##############################################################################

add_library(forge_capi SHARED
    forge_c_api.cpp
)

target_compile_definitions(forge_capi PRIVATE
    FORGE_CAPI_BUILDING
    FORGE_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    FORGE_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    FORGE_VERSION_PATCH=${PROJECT_VERSION_PATCH}
    $<$<BOOL:${FORGE_BUNDLE_AVX2}>:FORGE_BUNDLE_AVX2>
)

# Include directories for building (PRIVATE = internal forge headers)
target_include_directories(forge_capi PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    ${CMAKE_CURRENT_SOURCE_DIR}/..           # For <native/fdouble.hpp> etc.
    ${CMAKE_CURRENT_SOURCE_DIR}/../../tools
)

# Public include directory for consumers (forge_c_api.h)
target_include_directories(forge_capi PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Link forge and its dependencies statically into the shared library
target_link_libraries(forge_capi PRIVATE forge)

##############################################################################
# Per-file compile options for instruction-set-specific implementations
##############################################################################
#
# If you add C API files that use SIMD intrinsics directly, apply the
# appropriate compile flags to those specific files:
#
# Example for a file that needs AVX2 intrinsics:
#   set_source_files_properties(
#       forge_avx2_c_api.cpp
#       PROPERTIES COMPILE_OPTIONS "-mavx2"
#   )
#
# This ensures only that file is compiled with the special flags, avoiding
# performance regressions in other code paths. See the root CMakeLists.txt
# for the OBJECT library pattern used for the main Forge library.
#

# NOTE: We do NOT hide internal C++ symbols (--exclude-libs,ALL removed)
# This allows loadable backends to link against libforge_capi.so
# and share C++ class implementations, avoiding ABI issues with duplicate code.

# Set output name
set_target_properties(forge_capi PROPERTIES
    OUTPUT_NAME "forge_capi"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# On Windows, also generate import library
if(WIN32)
    set_target_properties(forge_capi PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS OFF
    )
endif()

##############################################################################
# Public header (C API only)
##############################################################################

# Create interface library for consumers
add_library(forge_capi_headers INTERFACE)
target_include_directories(forge_capi_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

##############################################################################
# Tests
##############################################################################

if(FORGE_CAPI_BUILD_TESTS)
    enable_testing()

    # Basic C API test
    add_executable(forge_capi_test test_capi.c)
    target_link_libraries(forge_capi_test PRIVATE forge_capi)
    target_include_directories(forge_capi_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    add_test(NAME forge_capi_basic_test COMMAND forge_capi_test)

    # Set library path for tests
    if(WIN32)
        set_tests_properties(forge_capi_basic_test PROPERTIES
            ENVIRONMENT "PATH=$<TARGET_FILE_DIR:forge_capi>;$ENV{PATH}"
        )
    else()
        set_tests_properties(forge_capi_basic_test PROPERTIES
            ENVIRONMENT "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:forge_capi>:$ENV{LD_LIBRARY_PATH}"
        )
    endif()
endif()

##############################################################################
# Installation
##############################################################################

install(TARGETS forge_capi
    EXPORT ForgeCAPITargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES forge_c_api.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export targets
install(EXPORT ForgeCAPITargets
    FILE ForgeCAPITargets.cmake
    NAMESPACE Forge::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ForgeCAPI
)

# Config files
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/ForgeCAPIConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ForgeCAPIConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ForgeCAPI
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ForgeCAPIConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ForgeCAPIConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ForgeCAPIConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ForgeCAPI
)
