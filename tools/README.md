# Forge Tools

This directory contains utilities, tooling, and packaging infrastructure for the Forge library.

## Directory Structure

### `capi/`
Stable C API wrapper for Forge. Provides an `extern "C"` interface that enables binary compatibility across different compilers and C++ standard library implementations. Use this for distributing precompiled Forge packages that work with any C++ compiler on a given platform.

### `packaging/`
CMake packaging infrastructure for building and installing Forge as a CMake package. This exports the full C++ API and requires ABI compatibility between the Forge build and consumer code (same compiler, standard library, etc.).

### `benchmarkTool/`
Performance benchmarking utilities for measuring Forge compilation and execution performance.

### `corruptionDetection/`
Tools for detecting memory and data corruption during graph compilation and execution.

### `graphSerialization/`
Utilities for serializing and deserializing Forge graphs to/from various formats.

### `sanityTool/`
Validation and sanity checking tools for verifying graph correctness and compiler output.

### `tape_interpreter/`
Interpreter for executing computation graphs without JIT compilation. Useful for debugging and platforms without JIT support.

### `testFunctions/`
Helper functions and utilities used by the test suite.

### `types/`
Type definitions and utilities shared across tools.

## Packaging Options

| Approach | Location | Use Case |
|----------|----------|----------|
| **C API** | `capi/` | Cross-compiler binary distribution (one package per platform) |
| **CMake Package** | `packaging/` | Source-compatible distribution (requires matching compiler) |
