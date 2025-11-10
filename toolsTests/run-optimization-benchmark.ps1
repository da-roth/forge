# Forge Optimization Benchmark - Run Only Script
# This script runs the already built optimization benchmark test

# Get the directory where this script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ToolsTestsDir = $ScriptDir
$ForgeRoot = Split-Path -Parent $ToolsTestsDir

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Forge Optimization Benchmark - Run Only" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Check multiple possible locations for the test executable (prefer Release builds)
$PossiblePaths = @(
    (Join-Path $ForgeRoot "build-vs\bin\Release\forge_tools_tests.exe"),      # VS Release build (preferred)
    (Join-Path $ForgeRoot "build-vs\bin\Debug\forge_tools_tests.exe"),        # VS Debug build
    (Join-Path $ForgeRoot "cmake-build-windows\bin\Release\forge_tools_tests.exe"), # CMake Windows Release
    (Join-Path $ForgeRoot "cmake-build-windows\bin\Debug\forge_tools_tests.exe"),   # CMake Windows Debug
    (Join-Path $ForgeRoot "build-fast\bin\Release\forge_tools_tests.exe"),    # Alternative Release build
    (Join-Path $ForgeRoot "build-fast\bin\Debug\forge_tools_tests.exe")       # Alternative Debug build
)

$TestExe = $null
$BuildType = ""

foreach ($Path in $PossiblePaths) {
    if (Test-Path $Path) {
        $TestExe = $Path
        if ($Path -match "Release") {
            $BuildType = "Release"
        } else {
            $BuildType = "Debug"
        }
        Write-Host "✓ Found test executable ($BuildType build):" -ForegroundColor Green
        Write-Host "  $TestExe" -ForegroundColor Gray
        break
    }
}

if (-not $TestExe) {
    Write-Host "✗ Error: forge_tools_tests.exe not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Searched in:" -ForegroundColor Yellow
    foreach ($Path in $PossiblePaths) {
        Write-Host "  - $Path" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "Please run 'build-and-run-optimization-benchmark.ps1' first to build the test executable." -ForegroundColor Yellow
    exit 1
}

# Check if we're using a Debug build and warn the user
if ($BuildType -eq "Debug") {
    Write-Host ""
    Write-Host "⚠️  Warning: Using Debug build. Performance results may not be representative." -ForegroundColor Yellow
    Write-Host "   For accurate benchmarks, build in Release mode." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Running Optimization Benchmark Test" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Run the specific test (OptimizationBenchmark test suite)
# You can modify the filter to run specific tests:
#   --gtest_filter="OptimizationBenchmark.*"              # All optimization benchmarks
#   --gtest_filter="OptimizationBenchmark.BasicGraph"     # Specific test
#   --gtest_filter="*"                                    # All tests in the executable

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