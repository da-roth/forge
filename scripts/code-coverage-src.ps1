# Code Coverage Script for Forge
# Generates HTML coverage report for forge_tests over /src
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
Forge Code Coverage Script
==========================

Usage: .\code-coverage-src.ps1 [options]

Options:
    -BuildType <type>   Build configuration: Debug or Release (default: Debug)
    -BuildDir <path>    Custom build directory (default: cmake-build-debug or build)
    -NoBuild            Skip building, use existing binaries
    -OpenReport         Automatically open the HTML report in browser
    -Help               Show this help message

Examples:
    .\code-coverage-src.ps1
    .\code-coverage-src.ps1 -BuildType Release -OpenReport
    .\code-coverage-src.ps1 -NoBuild -OpenReport

Requirements:
    - OpenCppCoverage must be installed and in PATH
    - Install via: winget install OpenCppCoverage.OpenCppCoverage

Output:
    Coverage report is generated in: <project>/coverage_report/index.html
"@
    exit 0
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Forge Code Coverage Report Generator" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
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
    # Try build first (VS/CMake), then cmake-build-debug (CLion)
    $buildFolder = Join-Path $ProjectRoot "build"
    $cmakeBuildDebug = Join-Path $ProjectRoot "cmake-build-debug"

    if (Test-Path $buildFolder) {
        $BuildDir = $buildFolder
    } elseif (Test-Path $cmakeBuildDebug) {
        $BuildDir = $cmakeBuildDebug
    } else {
        Write-Host "ERROR: No build directory found!" -ForegroundColor Red
        Write-Host "Please build the project first or specify -BuildDir" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host "Build directory: $BuildDir" -ForegroundColor Cyan
Write-Host "Build type: $BuildType" -ForegroundColor Cyan

# Find forge_tests executable
$testExePaths = @(
    (Join-Path $BuildDir "bin\forge_tests.exe"),
    (Join-Path $BuildDir "bin\$BuildType\forge_tests.exe"),
    (Join-Path $BuildDir "tests\$BuildType\forge_tests.exe"),
    (Join-Path $BuildDir "tests\forge_tests.exe")
)

$testExe = $null
foreach ($path in $testExePaths) {
    if (Test-Path $path) {
        $testExe = $path
        break
    }
}

# Build if needed
if (-not $NoBuild) {
    Write-Host ""
    Write-Host "Building forge_tests..." -ForegroundColor Yellow

    Push-Location $BuildDir
    try {
        # Try to build using cmake
        $buildResult = & cmake --build . --config $BuildType --target forge_tests 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Build failed!" -ForegroundColor Red
            Write-Host $buildResult
            exit 1
        }
        Write-Host "Build successful!" -ForegroundColor Green
    } finally {
        Pop-Location
    }

    # Re-check for executable after build
    foreach ($path in $testExePaths) {
        if (Test-Path $path) {
            $testExe = $path
            break
        }
    }
}

if (-not $testExe -or -not (Test-Path $testExe)) {
    Write-Host "ERROR: forge_tests.exe not found!" -ForegroundColor Red
    Write-Host "Searched in:" -ForegroundColor Yellow
    foreach ($path in $testExePaths) {
        Write-Host "  $path" -ForegroundColor Gray
    }
    Write-Host ""
    Write-Host "Please build the project first: cmake --build <build-dir> --target forge_tests" -ForegroundColor Yellow
    exit 1
}

Write-Host "Test executable: $testExe" -ForegroundColor Green

# Setup coverage output directory
$coverageDir = Join-Path $ProjectRoot "coverage_report"
if (Test-Path $coverageDir) {
    Write-Host "Cleaning previous coverage report..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $coverageDir
}

# Source directory to measure coverage for
$srcDir = Join-Path $ProjectRoot "src"

Write-Host ""
Write-Host "Running coverage analysis..." -ForegroundColor Yellow
Write-Host "Source directory: $srcDir" -ForegroundColor Cyan
Write-Host ""

# Run OpenCppCoverage
$coverageArgs = @(
    "--sources", $srcDir,
    "--modules", $BuildDir,
    "--export_type", "html:$coverageDir",
    "--", $testExe
)

Write-Host "Executing: $openCppCoveragePath $($coverageArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

& $openCppCoveragePath @coverageArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Coverage analysis failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

# Check if report was generated
$indexHtml = Join-Path $coverageDir "index.html"
if (Test-Path $indexHtml) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  Coverage report generated!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
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
    Write-Host "Check the coverage_report directory: $coverageDir" -ForegroundColor Yellow
}

Write-Host ""
