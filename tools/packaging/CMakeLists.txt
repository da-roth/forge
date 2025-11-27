cmake_minimum_required(VERSION 3.20)
project(ForgePackage VERSION 0.1.0 LANGUAGES CXX)

# Include GNUInstallDirs for standard installation directories
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(FetchContent)

# Set C++ standard to match QuantLib-Risks
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Match MSVC runtime settings from configure-ci.cmd and quantlib-risks-cpp
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    message(STATUS "Using static MSVC runtime")
endif()

# Use a shared FetchContent cache directory to avoid re-downloading dependencies
# This persists across clean builds
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../.deps-cache" CACHE PATH "FetchContent dependency cache" FORCE)
message(STATUS "FetchContent cache directory: ${FETCHCONTENT_BASE_DIR}")

# Disable SLEEF tests BEFORE adding forge subdirectory
# This prevents linker errors with iutdsp128.exe test utilities on Windows
set(SLEEF_BUILD_TESTS OFF CACHE BOOL "Disable SLEEF tests" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "Build SLEEF tests" FORCE)

# Build the existing forge target as-is
# From forge/tools/packaging, go up 2 levels to reach forge/
# Disable forge tests when building the package
set(FORGE_SKIP_TESTS ON CACHE BOOL "Skip building Forge tests when packaging" FORCE)
message(STATUS "Adding forge subdirectory (tests disabled for packaging)...")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. ${CMAKE_CURRENT_BINARY_DIR}/forge-build)

# Note: quantlib-template has been moved to quantlib-risks-cpp/quantlib-template
# It is no longer part of the Forge package

# The forge target should now be available
# We need to make it export-friendly by fixing include directories and dependencies

message(STATUS "Making forge target export-friendly...")

# Get the current include directories
get_target_property(FORGE_INCLUDE_DIRS forge INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "Original forge include dirs: ${FORGE_INCLUDE_DIRS}")

# Clear the INTERFACE_INCLUDE_DIRECTORIES and rebuild them with generator expressions
set_target_properties(forge PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")

# Add include directories with BUILD_INTERFACE and INSTALL_INTERFACE
# For build: point to forge/src and forge/tools directly
# For install: point to include/src and include/tools (preserving structure)
target_include_directories(forge INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../tools>
    $<BUILD_INTERFACE:${FETCHCONTENT_BASE_DIR}/asmjit-src/src>
    $<BUILD_INTERFACE:${FETCHCONTENT_BASE_DIR}/sleef-build/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/tools>
)

# Change third-party dependencies from PUBLIC to PRIVATE to avoid export issues
# We need to get the current link libraries and separate them
get_target_property(FORGE_LINK_LIBS forge INTERFACE_LINK_LIBRARIES)
message(STATUS "Original forge link libs: ${FORGE_LINK_LIBS}")

# Clear the interface link libraries
set_target_properties(forge PROPERTIES INTERFACE_LINK_LIBRARIES "")

# For STATIC libraries, CMake requires dependencies in the export even if PRIVATE
# So we need to keep them and include them in our export set
target_link_libraries(forge PUBLIC asmjit nlohmann_json::nlohmann_json sleefavx2 sleef)

# Install the third-party dependencies that forge needs
# These need to be in the export set for static library linking
install(TARGETS asmjit sleefavx2 sleef
    EXPORT ForgeTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# nlohmann_json is header-only, handle it separately if needed
if(TARGET nlohmann_json::nlohmann_json)
    # Get the actual target name (might be nlohmann_json not nlohmann_json::nlohmann_json)
    if(TARGET nlohmann_json)
        install(TARGETS nlohmann_json
            EXPORT ForgeTargets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )

        # Install the natvis file if it exists
        if(EXISTS "${FETCHCONTENT_BASE_DIR}/json-src/nlohmann_json.natvis")
            install(FILES "${FETCHCONTENT_BASE_DIR}/json-src/nlohmann_json.natvis"
                DESTINATION ${CMAKE_INSTALL_PREFIX}
            )
        endif()
    endif()
endif()

# Set the installation directories
# We need to preserve the src/ and tools/ directory structure for relative includes
set(FORGE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR})
set(FORGE_INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR})
set(FORGE_INSTALL_BINDIR ${CMAKE_INSTALL_BINDIR})
set(FORGE_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/Forge)

# Install the forge library target
install(TARGETS forge
    EXPORT ForgeTargets
    LIBRARY DESTINATION ${FORGE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${FORGE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${FORGE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${FORGE_INSTALL_INCLUDEDIR}
)

# Install header files preserving the src/ and tools/ directory structure
# This is necessary because headers use relative includes like "../../src/graph/graph.hpp"
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../src/
    DESTINATION ${FORGE_INSTALL_INCLUDEDIR}/src
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/
    DESTINATION ${FORGE_INSTALL_INCLUDEDIR}/tools
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Export the targets for use from the build tree (optional, for development)
export(EXPORT ForgeTargets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/ForgeTargets.cmake
    NAMESPACE Forge::
)

# Install the export set
install(EXPORT ForgeTargets
    FILE ForgeTargets.cmake
    NAMESPACE Forge::
    DESTINATION ${FORGE_INSTALL_CMAKEDIR}
)

# Create the ForgeConfig.cmake file
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/ForgeConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ForgeConfig.cmake
    INSTALL_DESTINATION ${FORGE_INSTALL_CMAKEDIR}
    PATH_VARS FORGE_INSTALL_INCLUDEDIR
)

# Create the ForgeConfigVersion.cmake file
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ForgeConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Install the config files
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ForgeConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ForgeConfigVersion.cmake
    DESTINATION ${FORGE_INSTALL_CMAKEDIR}
)

message(STATUS "Forge package configured for installation to: ${CMAKE_INSTALL_PREFIX}")
