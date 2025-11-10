#!/bin/bash
# Forge Examples - Build and Run Script
# This script builds only the examples and runs them

set -e  # Exit on error

# Parse arguments
DEBUG=0
CLEAN=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            DEBUG=1
            shift
            ;;
        --clean)
            CLEAN=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--debug] [--clean]"
            exit 1
            ;;
    esac
done

# Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  FORGE Examples - Build & Run${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Get script directory (forge root)
FORGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$FORGE_ROOT/build"

# Configuration
if [ $DEBUG -eq 1 ]; then
    BUILD_TYPE="Debug"
else
    BUILD_TYPE="Release"
fi

echo -e "${YELLOW}Configuration:${NC}"
echo "  Build Type: $BUILD_TYPE"
echo ""

# Clean if requested
if [ $CLEAN -eq 1 ] && [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf "$BUILD_DIR"
    echo -e "${GREEN}✓ Clean complete${NC}"
    echo ""
fi

# Create build directory if needed
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Creating build directory...${NC}"
    mkdir -p "$BUILD_DIR"
fi

# Navigate to build directory
cd "$BUILD_DIR"

# Configure with CMake if needed
if [ ! -f "CMakeCache.txt" ]; then
    echo -e "${YELLOW}Configuring with CMake...${NC}"

    cmake .. -DCMAKE_BUILD_TYPE=$BUILD_TYPE

    echo -e "${GREEN}✓ Configuration complete${NC}"
    echo ""
fi

# Build forge library first (includes SLEEF dependencies)
echo -e "${YELLOW}Building forge library and dependencies...${NC}"
cmake --build . --target forge --config $BUILD_TYPE

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to build forge library${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Forge library built${NC}"
echo ""

# Now build the examples (can use parallel builds safely)
echo -e "${YELLOW}Building examples...${NC}"
echo ""

EXAMPLE_TARGETS=("basic_gradient" "multi_variable" "performance_demo")

for target in "${EXAMPLE_TARGETS[@]}"; do
    echo -e "${GRAY}  Building $target...${NC}"
    cmake --build . --target $target --config $BUILD_TYPE -j

    if [ $? -ne 0 ]; then
        echo ""
        echo -e "${RED}Build failed for $target${NC}"
        exit 1
    fi
done

echo ""
echo -e "${GREEN}✓ Build completed successfully!${NC}"
echo ""

# Run the examples
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Running Examples${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

EXAMPLE_PATH="$BUILD_DIR/bin"

# Run basic_gradient
echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Example 1: Basic Gradient Computation                        ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

EXE="$EXAMPLE_PATH/basic_gradient"
if [ -f "$EXE" ]; then
    "$EXE"
    echo ""
else
    echo -e "${RED}  ERROR: basic_gradient not found${NC}"
fi

# Run multi_variable
echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Example 2: Multi-Variable Gradients                          ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

EXE="$EXAMPLE_PATH/multi_variable"
if [ -f "$EXE" ]; then
    "$EXE"
    echo ""
else
    echo -e "${RED}  ERROR: multi_variable not found${NC}"
fi

# Run performance_demo
echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Example 3: Performance Demonstration                         ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

EXE="$EXAMPLE_PATH/performance_demo"
if [ -f "$EXE" ]; then
    "$EXE"
    echo ""
else
    echo -e "${RED}  ERROR: performance_demo not found${NC}"
fi

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✓ All examples completed!${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

exit 0
