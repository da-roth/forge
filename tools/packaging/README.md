# Forge CMake Package

This directory contains the CMake packaging infrastructure for Forge, enabling it to be built and installed as a standalone CMake package.
## Purpose

Forge is packaged separately to solve CMake export issues when QuantLib tries to export its targets. By making Forge a proper CMake package, QuantLib sees it as a clean `IMPORTED` target rather than an in-tree dependency with complex third-party dependencies.

## Files

- **CMakeLists.txt**: Main packaging logic
  - Builds the forge library from `../../..` (forge root)
  - Configures include directories for both build and install
  - Exports ForgeTargets.cmake and ForgeConfig.cmake
  - Handles third-party dependencies (asmjit, sleef, nlohmann_json)

- **ForgeConfig.cmake.in**: Template for the package configuration file
  - Used by `find_package(Forge CONFIG REQUIRED)`
  - Provides the `Forge::forge` target to consumers

- **build-forge.cmd**: Build script
  - Configures, builds, and installs the Forge package
  - Called automatically by `configure-ci.cmd` when needed

## Usage

### Automatic (via configure-ci.cmd)

The Forge package is built automatically when you run the main build script:

```cmd
configure-ci.cmd
```

This will:
1. Check if Forge is already installed at `install/`
2. If not, build and install it by calling `build-forge.cmd`
3. Then build QuantLib-Risks with `CMAKE_PREFIX_PATH` pointing to the installed Forge

### Manual Build

To manually rebuild the Forge package:

```cmd
build-forge.cmd
```

Or for a clean build:

```cmd
configure-ci.cmd clean
```

## What Gets Installed

The Forge package includes:

**Libraries:**
- `forge.lib` - Main Forge library
- `asmjit.lib` - JIT assembly library (dependency)
- `sleef.lib` - SIMD math library (dependency)

**Headers:**
- `include/src/*` - Core Forge headers (fdouble, graph, compiler)
- `include/tools/*` - Tools including quantlib-template

**CMake Files:**
- `lib/cmake/Forge/ForgeConfig.cmake` - Package configuration
- `lib/cmake/Forge/ForgeTargets.cmake` - Exported targets

## Integration with QuantLib-Risks

`quantlib-risks-cpp/CMakeLists.txt` now uses:

```cmake
find_package(Forge CONFIG REQUIRED)
target_link_libraries(QuantLib-Risks INTERFACE Forge::forge)
```

This provides access to:
- Core Forge library (types/fdouble, graph recording, JIT compiler)
- quantlib-template expression templates
- All necessary include directories and dependencies

## Dependency Caching

Third-party dependencies (asmjit, sleef, nlohmann_json) are cached in `.deps-cache/` at the repository root. This cache persists across clean builds to avoid re-downloading dependencies.