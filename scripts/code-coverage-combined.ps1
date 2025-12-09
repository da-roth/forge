# Combined Code Coverage Script for Forge
# Generates HTML coverage report for BOTH forge_tests AND forge_tools_tests over /src
# This gives a comprehensive view of what code is actually used across all tests
# Requires: OpenCppCoverage (https://github.com/OpenCppCoverage/OpenCppCoverage)
#
# Install OpenCppCoverage:
#   winget install OpenCppCoverage.OpenCppCoverage
#   -- or --
#   choco install opencppcoverage
#   -- or --
#   Download from: https://github.com/OpenCppCoverage/OpenCppCoverage/releases

param(
    [string]$BuildType = "Debug",
    [string]$BuildDir = "",
    [switch]$NoBuild,
    [switch]$OpenReport,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# Script location and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# Show help
if ($Help) {
    Write-Host @"
Forge Combined Code Coverage Script
====================================

This script runs BOTH forge_tests and forge_tools_tests and generates
a single unified coverage report for /src. This helps identify:
1. Overall code coverage from all test sources
2. Unused code that could potentially be removed

Usage: .\code-coverage-combined.ps1 [options]

Options:
    -BuildType <type>   Build configuration: Debug or Release (default: Debug)
    -BuildDir <path>    Custom build directory (default: cmake-build-debug or build)
    -NoBuild            Skip building, use existing binaries
    -OpenReport         Automatically open the HTML report in browser
    -Fast               Skip long-running tests for faster coverage analysis
    -Help               Show this help message

Examples:
    .\code-coverage-combined.ps1
    .\code-coverage-combined.ps1 -BuildType Release -OpenReport
    .\code-coverage-combined.ps1 -NoBuild -OpenReport

Requirements:
    - OpenCppCoverage must be installed and in PATH
    - Install via: winget install OpenCppCoverage.OpenCppCoverage

Output:
    Coverage report is generated in: <project>/coverage_report_combined/index.html
"@
    exit 0
}

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Forge Combined Code Coverage Report (All Tests)" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check if OpenCppCoverage is installed
$openCppCoverage = Get-Command "OpenCppCoverage.exe" -ErrorAction SilentlyContinue
if (-not $openCppCoverage) {
    # Try common installation paths
    $commonPaths = @(
        "$env:ProgramFiles\OpenCppCoverage\OpenCppCoverage.exe",
        "${env:ProgramFiles(x86)}\OpenCppCoverage\OpenCppCoverage.exe",
        "$env:LOCALAPPDATA\Programs\OpenCppCoverage\OpenCppCoverage.exe"
    )

    foreach ($path in $commonPaths) {
        if (Test-Path $path) {
            $openCppCoverage = @{ Source = $path }
            break
        }
    }

    if (-not $openCppCoverage) {
        Write-Host "ERROR: OpenCppCoverage not found!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please install OpenCppCoverage using one of these methods:" -ForegroundColor Yellow
        Write-Host "  winget install OpenCppCoverage.OpenCppCoverage" -ForegroundColor White
        Write-Host "  choco install opencppcoverage" -ForegroundColor White
        Write-Host "  Download: https://github.com/OpenCppCoverage/OpenCppCoverage/releases" -ForegroundColor White
        exit 1
    }
}

$openCppCoveragePath = if ($openCppCoverage.Source) { $openCppCoverage.Source } else { "OpenCppCoverage.exe" }
Write-Host "Found OpenCppCoverage: $openCppCoveragePath" -ForegroundColor Green

# Determine build directory
if ($BuildDir -eq "") {
    # Try cmake-build-debug first (CLion), then build (VS/CMake)
    $cmakeBuildDebug = Join-Path $ProjectRoot "cmake-build-debug"
    $buildFolder = Join-Path $ProjectRoot "build"

    if (Test-Path $cmakeBuildDebug) {
        $BuildDir = $cmakeBuildDebug
    } elseif (Test-Path $buildFolder) {
        $BuildDir = $buildFolder
    } else {
        Write-Host "ERROR: No build directory found!" -ForegroundColor Red
        Write-Host "Please build the project first or specify -BuildDir" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host "Build directory: $BuildDir" -ForegroundColor Cyan
Write-Host "Build type: $BuildType" -ForegroundColor Cyan

# Find both test executables
$testExePaths = @(
    (Join-Path $BuildDir "bin\forge_tests.exe"),
    (Join-Path $BuildDir "bin\$BuildType\forge_tests.exe"),
    (Join-Path $BuildDir "tests\$BuildType\forge_tests.exe"),
    (Join-Path $BuildDir "tests\forge_tests.exe")
)

$toolsTestExePaths = @(
    (Join-Path $BuildDir "bin\forge_tools_tests.exe"),
    (Join-Path $BuildDir "bin\$BuildType\forge_tools_tests.exe"),
    (Join-Path $BuildDir "toolsTests\$BuildType\forge_tools_tests.exe"),
    (Join-Path $BuildDir "toolsTests\forge_tools_tests.exe")
)

$testExe = $null
$toolsTestExe = $null

foreach ($path in $testExePaths) {
    if (Test-Path $path) {
        $testExe = $path
        break
    }
}

foreach ($path in $toolsTestExePaths) {
    if (Test-Path $path) {
        $toolsTestExe = $path
        break
    }
}

# Build if needed
if (-not $NoBuild) {
    Write-Host ""
    Write-Host "Building test executables..." -ForegroundColor Yellow

    Push-Location $BuildDir
    try {
        # Build forge_tests
        Write-Host "  Building forge_tests..." -ForegroundColor Gray
        $buildResult = & cmake --build . --config $BuildType --target forge_tests 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Build failed for forge_tests!" -ForegroundColor Red
            Write-Host $buildResult
            exit 1
        }

        # Build forge_tools_tests
        Write-Host "  Building forge_tools_tests..." -ForegroundColor Gray
        $buildResult = & cmake --build . --config $BuildType --target forge_tools_tests 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Build failed for forge_tools_tests!" -ForegroundColor Red
            Write-Host $buildResult
            exit 1
        }

        Write-Host "Build successful!" -ForegroundColor Green
    } finally {
        Pop-Location
    }

    # Re-check for executables after build
    foreach ($path in $testExePaths) {
        if (Test-Path $path) {
            $testExe = $path
            break
        }
    }

    foreach ($path in $toolsTestExePaths) {
        if (Test-Path $path) {
            $toolsTestExe = $path
            break
        }
    }
}

# Verify both executables exist
$missingExes = @()
if (-not $testExe -or -not (Test-Path $testExe)) {
    $missingExes += "forge_tests.exe"
}
if (-not $toolsTestExe -or -not (Test-Path $toolsTestExe)) {
    $missingExes += "forge_tools_tests.exe"
}

if ($missingExes.Count -gt 0) {
    Write-Host "ERROR: Missing test executables: $($missingExes -join ', ')" -ForegroundColor Red
    Write-Host ""
    if (-not $testExe) {
        Write-Host "forge_tests.exe searched in:" -ForegroundColor Yellow
        foreach ($path in $testExePaths) {
            Write-Host "  $path" -ForegroundColor Gray
        }
        Write-Host ""
    }
    if (-not $toolsTestExe) {
        Write-Host "forge_tools_tests.exe searched in:" -ForegroundColor Yellow
        foreach ($path in $toolsTestExePaths) {
            Write-Host "  $path" -ForegroundColor Gray
        }
        Write-Host ""
    }
    Write-Host "Please build the project first: cmake --build <build-dir>" -ForegroundColor Yellow
    exit 1
}

Write-Host "Test executables found:" -ForegroundColor Green
Write-Host "  forge_tests:       $testExe" -ForegroundColor Cyan
Write-Host "  forge_tools_tests: $toolsTestExe" -ForegroundColor Cyan

# Setup coverage output directory
$coverageDir = Join-Path $ProjectRoot "coverage_report_combined"
$tempCoverageDir1 = Join-Path $ProjectRoot "coverage_temp_unit"
$tempCoverageDir2 = Join-Path $ProjectRoot "coverage_temp_tools"

# Clean up previous reports
foreach ($dir in @($coverageDir, $tempCoverageDir1, $tempCoverageDir2)) {
    if (Test-Path $dir) {
        Remove-Item -Recurse -Force $dir
    }
}

# Source directory to measure coverage for
$srcDir = Join-Path $ProjectRoot "src"

Write-Host ""
Write-Host "Running combined coverage analysis..." -ForegroundColor Yellow
Write-Host "Source directory: $srcDir" -ForegroundColor Cyan
Write-Host ""

# Step 1: Run forge_tests and save binary coverage
Write-Host "[1/3] Running forge_tests..." -ForegroundColor Yellow
$coverageArgs1 = @(
    "--sources", $srcDir,
    "--modules", $BuildDir,
    "--export_type", "binary:$tempCoverageDir1\coverage1.bin",
    "--", $testExe,
    "--gtest_filter=-OptimizationBenchmark.*:BridgeWorkflowBenchmark.*"
)

& $openCppCoveragePath @coverageArgs1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Coverage analysis failed for forge_tests (exit code: $LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "  forge_tests coverage collected" -ForegroundColor Green

# Step 2: Run forge_tools_tests and save binary coverage
Write-Host ""
Write-Host "[2/3] Running forge_tools_tests..." -ForegroundColor Yellow
$coverageArgs2 = @(
    "--sources", $srcDir,
    "--modules", $BuildDir,
    "--export_type", "binary:$tempCoverageDir2\coverage2.bin",
    "--", $toolsTestExe,
    "--gtest_filter=-OptimizationBenchmark.*:BridgeWorkflowBenchmark.*"
)

& $openCppCoveragePath @coverageArgs2 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Coverage analysis failed for forge_tools_tests (exit code: $LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "  forge_tools_tests coverage collected" -ForegroundColor Green

# Step 3: Merge both coverage reports into HTML
Write-Host ""
Write-Host "[3/3] Merging coverage reports..." -ForegroundColor Yellow

$mergeArgs = @(
    "--input_coverage", "$tempCoverageDir1\coverage1.bin",
    "--input_coverage", "$tempCoverageDir2\coverage2.bin",
    "--export_type", "html:$coverageDir"
)

& $openCppCoveragePath @mergeArgs | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Coverage merge failed (exit code: $LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "  Coverage reports merged successfully" -ForegroundColor Green

# Clean up temporary directories
Remove-Item -Recurse -Force $tempCoverageDir1 -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force $tempCoverageDir2 -ErrorAction SilentlyContinue

# Check if report was generated
$indexHtml = Join-Path $coverageDir "index.html"
if (Test-Path $indexHtml) {
    Write-Host ""
    Write-Host "================================================" -ForegroundColor Green
    Write-Host "  Combined coverage report generated!" -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "This report includes coverage from:" -ForegroundColor Cyan
    Write-Host "  - forge_tests (unit tests)" -ForegroundColor White
    Write-Host "  - forge_tools_tests (integration/benchmark tests)" -ForegroundColor White
    Write-Host ""
    Write-Host "Report location: $indexHtml" -ForegroundColor Cyan

    if ($OpenReport) {
        Write-Host "Opening report in browser..." -ForegroundColor Yellow
        Start-Process $indexHtml
    } else {
        Write-Host ""
        Write-Host "To view the report, run:" -ForegroundColor Yellow
        Write-Host "  Start-Process `"$indexHtml`"" -ForegroundColor White
        Write-Host "Or use -OpenReport flag next time" -ForegroundColor Gray
    }
} else {
    Write-Host ""
    Write-Host "WARNING: Report index.html not found at expected location" -ForegroundColor Yellow
    Write-Host "Check the coverage_report_combined directory: $coverageDir" -ForegroundColor Yellow
}

Write-Host ""
