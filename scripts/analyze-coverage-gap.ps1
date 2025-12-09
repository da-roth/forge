# Coverage Gap Analyzer
# Compares line-by-line coverage between unit tests and combined tests

param([switch]$Help)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

if ($Help) {
    Write-Host @"
Coverage Gap Analyzer - Finds code tested by toolsTests but not by unit tests
Usage: .\analyze-coverage-gap.ps1
"@
    exit 0
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Coverage Gap Analysis" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$unitTestReport = Join-Path $ProjectRoot "coverage_report\Modules\forge_tests"
$combinedReport = Join-Path $ProjectRoot "coverage_report_combined\Modules"
$outputFile = Join-Path $ProjectRoot "coverage_gap_analysis.txt"

if (-not (Test-Path $unitTestReport)) {
    Write-Host "ERROR: Unit test report not found" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $combinedReport)) {
    Write-Host "ERROR: Combined report not found" -ForegroundColor Red
    exit 1
}

Write-Host "Analyzing coverage reports..." -ForegroundColor Yellow
Write-Host ""

# Get all source files from combined report (use forge_tests only to avoid duplicates)
$combinedFiles = Get-ChildItem -Path (Join-Path $combinedReport "forge_tests") -Filter "*.html" -ErrorAction SilentlyContinue

Write-Host "Found $($combinedFiles.Count) source files" -ForegroundColor Green
Write-Host ""

$gapFiles = @()
$totalGapLines = 0
$sourcePathMap = @{}  # Map filename to actual source path

foreach ($file in $combinedFiles) {
    $fileName = $file.Name
    $combinedPath = $file.FullName

    # Find corresponding unit test file
    $unitPath = Join-Path $unitTestReport $fileName

    if (-not (Test-Path $unitPath)) {
        continue  # Skip if not in unit tests
    }

    # Count covered lines (green background)
    $combinedHtml = Get-Content $combinedPath -Raw
    $unitHtml = Get-Content $unitPath -Raw

    # Extract actual source file path from HTML (look for href link to source)
    $actualSourcePath = ""
    if ($combinedHtml -match '<a href="[^"]+">([^<]+)</a>') {
        $actualSourcePath = $matches[1]
        # Clean up and make relative to project root
        $actualSourcePath = $actualSourcePath -replace '.*\\(src\\.*)', '$1'
        $actualSourcePath = $actualSourcePath -replace '.*\\(tools\\.*)', '$1'
    }

    # Count green lines: background-color:#dfd
    $combinedGreen = ([regex]::Matches($combinedHtml, 'background-color:#dfd')).Count
    $unitGreen = ([regex]::Matches($unitHtml, 'background-color:#dfd')).Count

    # Count red lines: background-color:#fdd
    $combinedRed = ([regex]::Matches($combinedHtml, 'background-color:#fdd')).Count
    $unitRed = ([regex]::Matches($unitHtml, 'background-color:#fdd')).Count

    $combinedTotal = $combinedGreen + $combinedRed
    $unitTotal = $unitGreen + $unitRed

    if ($combinedTotal -eq 0) { continue }

    $gap = $combinedGreen - $unitGreen

    if ($gap -gt 0) {
        $combinedPercent = [math]::Round(($combinedGreen / $combinedTotal) * 100)
        $unitPercent = [math]::Round(($unitGreen / $unitTotal) * 100)

        # Extract source file name (without .html)
        $sourceFile = $fileName -replace '\.html$', ''

        # Store actual path for reference
        if ($actualSourcePath) {
            $sourcePathMap[$sourceFile] = $actualSourcePath
        }

        $gapFiles += [PSCustomObject]@{
            SourceFile = $sourceFile
            ActualPath = $actualSourcePath
            UnitCovered = $unitGreen
            UnitPercent = $unitPercent
            CombinedCovered = $combinedGreen
            CombinedPercent = $combinedPercent
            Gap = $gap
            PercentGap = $combinedPercent - $unitPercent
            TotalLines = $combinedTotal
        }

        $totalGapLines += $gap
    }
}

# Sort by gap size
$gapFiles = $gapFiles | Sort-Object -Property Gap -Descending

# Generate report
$report = @"
================================================================================
                         COVERAGE GAP ANALYSIS
================================================================================

Summary:
--------
Files with coverage gap: $($gapFiles.Count)
Total lines covered by toolsTests but NOT by unit tests: $totalGapLines

This shows which /src code is tested by /toolsTests but missing from /tests.
Add unit tests for these areas to close the gap.

================================================================================
                    GAP BY FILE (Sorted by Impact)
================================================================================

{0,-50} {1,8} {2,8} {3,8} {4,8} {5,8}
{6,-50} {7,8} {8,8} {9,8} {10,8} {11,8}
"@ -f "Source File", "Unit %", "All %", "Gap %", "+Lines", "Total",
      ("-" * 50), "-------", "-------", "-------", "-------", "-------"

foreach ($item in $gapFiles) {
    $report += "`n{0,-50} {1,7}% {2,7}% {3,7}% {4,8} {5,8}" -f `
        $item.SourceFile,
        $item.UnitPercent,
        $item.CombinedPercent,
        $item.PercentGap,
        "+$($item.Gap)",
        $item.TotalLines
}

$report += "`n`nCoverage Report Locations (for comparing unit vs combined):"
$report += "`n" + ("-" * 100)
foreach ($item in $gapFiles) {
    $unitHtmlPath = "coverage_report\Modules\forge_tests\$($item.SourceFile).html"
    $combinedHtmlPath = "coverage_report_combined\Modules\forge_tests\$($item.SourceFile).html"
    $report += "`n`n$($item.SourceFile):"
    $report += "`n  Unit tests only:  file:///$($ProjectRoot -replace '\\', '/')/$($unitHtmlPath -replace '\\', '/')"
    $report += "`n  All tests:        file:///$($ProjectRoot -replace '\\', '/')/$($combinedHtmlPath -replace '\\', '/')"
}

$report += @"


================================================================================
                      TOP 10 PRIORITY RECOMMENDATIONS
================================================================================

Add unit tests for these files (highest impact first):

"@

$top10 = $gapFiles | Select-Object -First 10
$index = 1
foreach ($item in $top10) {
    $report += "$index. $($item.SourceFile)`n"
    $report += "   Gap: $($item.Gap) lines not covered by unit tests (+$($item.PercentGap)%)`n"
    $report += "   Current: $($item.UnitPercent)% (unit) -> $($item.CombinedPercent)% (all tests)`n"
    $report += "`n"
    $index++
}

$report += @"
================================================================================
                              NEXT STEPS
================================================================================

1. Review /toolsTests to see what tests cover the gap files
2. Create equivalent unit tests in /tests
3. Re-run coverage to verify gap is closing

Commands:
    .\scripts\code-coverage-src.ps1
    .\scripts\code-coverage-combined.ps1
    .\scripts\analyze-coverage-gap.ps1

================================================================================
"@

Set-Content -Path $outputFile -Value $report -Encoding UTF8

Write-Host "Analysis complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Gap Summary:" -ForegroundColor Cyan
Write-Host "  Files with gaps: $($gapFiles.Count)" -ForegroundColor White
Write-Host "  Total gap lines: $totalGapLines" -ForegroundColor White
Write-Host ""
Write-Host "Top 5 files needing unit tests:" -ForegroundColor Yellow
$top5 = $gapFiles | Select-Object -First 5
foreach ($item in $top5) {
    Write-Host ("  {0,-45} +{1,3} lines ({2,2}% -> {3,2}%)" -f $item.SourceFile, $item.Gap, $item.UnitPercent, $item.CombinedPercent) -ForegroundColor Gray
}
Write-Host ""
Write-Host "Full report: $outputFile" -ForegroundColor Cyan
Write-Host ""
