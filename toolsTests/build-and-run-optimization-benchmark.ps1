# Forge Optimization Benchmark - Build and Run Script
# This script builds the Forge tests and runs the optimization benchmark

# Get the directory where this script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ToolsTestsDir = $ScriptDir
$ForgeRoot = Split-Path -Parent $ToolsTestsDir

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Forge Optimization Benchmark - Build & Run" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Set build directory (within forge directory)
$BuildDir = Join-Path $ForgeRoot "build-vs"

# Check if build directory exists, create if not
if (-not (Test-Path $BuildDir)) {
    Write-Host "Creating build directory: $BuildDir" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
}

# Navigate to build directory
Push-Location $BuildDir

try {
    # Configure with CMake if needed
    if (-not (Test-Path "CMakeCache.txt")) {
        Write-Host "Configuring project with CMake..." -ForegroundColor Yellow
        cmake .. -G "Visual Studio 17 2022" -A x64
        if ($LASTEXITCODE -ne 0) {
            throw "CMake configuration failed"
        }
    }

    # Build the test executable in Release mode
    Write-Host ""
    Write-Host "Building forge_tools_tests in Release mode..." -ForegroundColor Yellow
    cmake --build . --target forge_tools_tests --config Release -- /m

    if ($LASTEXITCODE -ne 0) {
        throw "Build failed"
    }

    Write-Host "✓ Build completed successfully!" -ForegroundColor Green
    Write-Host ""

    # Check if the executable exists
    $TestExe = Join-Path $BuildDir "bin\Release\forge_tools_tests.exe"
    
    if (-not (Test-Path $TestExe)) {
        throw "Test executable not found at: $TestExe"
    }

    # Run the optimization benchmark test
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  Running Optimization Benchmark Test" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    
    # Run the specific test (OptimizationBenchmark test suite)
    & $TestExe --gtest_filter="OptimizationBenchmark.*"
    
    $TestExitCode = $LASTEXITCODE
    
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    
    if ($TestExitCode -eq 0) {
        Write-Host "  ✓ Benchmark completed successfully!" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Benchmark failed with exit code: $TestExitCode" -ForegroundColor Red
    }
    
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    
    # Return the test exit code
    exit $TestExitCode
    
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
} finally {
    # Return to original directory
    Pop-Location
}