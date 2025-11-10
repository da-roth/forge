# Forge Examples - Build and Run Script
# This script builds only the examples and runs them

param(
    [switch]$Debug = $false,
    [switch]$Clean = $false
)

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  FORGE Examples - Build & Run" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Get the script directory (forge root)
$ForgeRoot = $PSScriptRoot
$BuildDir = Join-Path $ForgeRoot "build"

# Configuration
$BuildType = if ($Debug) { "Debug" } else { "Release" }

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Build Type: $BuildType"
Write-Host ""

# Clean if requested
if ($Clean -and (Test-Path $BuildDir)) {
    Write-Host "Cleaning build directory..." -ForegroundColor Yellow
    Remove-Item -Path $BuildDir -Recurse -Force
    Write-Host "✓ Clean complete" -ForegroundColor Green
    Write-Host ""
}

# Create build directory if needed
if (!(Test-Path $BuildDir)) {
    Write-Host "Creating build directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

# Navigate to build directory
Push-Location $BuildDir

try {
    # Configure with CMake if needed
    if (!(Test-Path "CMakeCache.txt")) {
        Write-Host "Configuring with CMake..." -ForegroundColor Yellow

        # Detect Visual Studio version
        $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
        if (Test-Path $vsWhere) {
            $vsVersion = & $vsWhere -latest -property catalog_productLineVersion
            if ($vsVersion -eq "2022") {
                $generator = "Visual Studio 17 2022"
            } elseif ($vsVersion -eq "2019") {
                $generator = "Visual Studio 16 2019"
            } else {
                $generator = "Visual Studio 15 2017"
            }
        } else {
            $generator = "Visual Studio 17 2022"
        }

        cmake .. -G "$generator" -A x64

        if ($LASTEXITCODE -ne 0) {
            throw "CMake configuration failed"
        }

        Write-Host "✓ Configuration complete" -ForegroundColor Green
        Write-Host ""
    }

    # Build forge library first (includes SLEEF dependencies)
    Write-Host "Building forge library and dependencies..." -ForegroundColor Yellow
    cmake --build . --target forge --config $BuildType

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to build forge library"
    }

    Write-Host "✓ Forge library built" -ForegroundColor Green
    Write-Host ""

    # Now build the examples (can use parallel builds safely)
    Write-Host "Building examples..." -ForegroundColor Yellow
    Write-Host ""

    $exampleTargets = @("basic_gradient", "multi_variable", "performance_demo")

    foreach ($target in $exampleTargets) {
        Write-Host "  Building $target..." -ForegroundColor Gray
        cmake --build . --target $target --config $BuildType -- /m

        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "Build failed for $target with exit code $LASTEXITCODE" -ForegroundColor Red
            Write-Host "Run 'cmake --build . --target $target --config $BuildType' manually for full output" -ForegroundColor Yellow
            throw "Build failed for $target"
        }
    }

    Write-Host ""
    Write-Host "✓ Build completed successfully!" -ForegroundColor Green
    Write-Host ""

    # Run the examples
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  Running Examples" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""

    $examplePath = Join-Path $BuildDir "bin\$BuildType"

    # Run basic_gradient
    Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║  Example 1: Basic Gradient Computation                        ║" -ForegroundColor Cyan
    Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""

    $exe = Join-Path $examplePath "basic_gradient.exe"
    if (Test-Path $exe) {
        & $exe
        Write-Host ""
    } else {
        Write-Host "  ERROR: basic_gradient.exe not found" -ForegroundColor Red
    }

    # Run multi_variable
    Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║  Example 2: Multi-Variable Gradients                          ║" -ForegroundColor Cyan
    Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""

    $exe = Join-Path $examplePath "multi_variable.exe"
    if (Test-Path $exe) {
        & $exe
        Write-Host ""
    } else {
        Write-Host "  ERROR: multi_variable.exe not found" -ForegroundColor Red
    }

    # Run performance_demo
    Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║  Example 3: Performance Demonstration                         ║" -ForegroundColor Cyan
    Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""

    $exe = Join-Path $examplePath "performance_demo.exe"
    if (Test-Path $exe) {
        & $exe
        Write-Host ""
    } else {
        Write-Host "  ERROR: performance_demo.exe not found" -ForegroundColor Red
    }

    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  ✓ All examples completed!" -ForegroundColor Green
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""

} catch {
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
} finally {
    Pop-Location
}

exit 0
