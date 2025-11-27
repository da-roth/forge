@echo off
REM build-forge.cmd
REM Builds and installs the Forge package for use by QuantLib-Risks

echo ========================================
echo Building Forge Package
echo ========================================

REM Get the root directory (3 levels up from forge/tools/packaging)
set SCRIPT_DIR=%~dp0
set FORGE_PACKAGING_DIR=%SCRIPT_DIR%
set INSTALL_DIR=%SCRIPT_DIR%..\..\..\install
set BUILD_DIR=%FORGE_PACKAGING_DIR%build
set DEPS_CACHE=%SCRIPT_DIR%..\..\..\..deps-cache

echo.
echo Paths:
echo   Packaging Dir: %FORGE_PACKAGING_DIR%
echo   Install Dir: %INSTALL_DIR%
echo   Build Dir: %BUILD_DIR%
echo   Deps Cache: %DEPS_CACHE%
echo.

REM Clean previous build directory
if exist "%BUILD_DIR%" (
    echo Removing previous build directory...
    rmdir /s /q "%BUILD_DIR%"
)

mkdir "%BUILD_DIR%"

REM Configure with CMake
echo Configuring Forge package...
cd /d "%BUILD_DIR%"

REM Convert paths to forward slashes for CMake
set INSTALL_CMAKE=%INSTALL_DIR:\=/%
set DEPS_CACHE_CMAKE=%DEPS_CACHE:\=/%
set PACKAGING_CMAKE=%FORGE_PACKAGING_DIR:\=/%

cmake -S %PACKAGING_CMAKE% -B . -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=%INSTALL_CMAKE% -DFETCHCONTENT_BASE_DIR=%DEPS_CACHE_CMAKE%

if %ERRORLEVEL% neq 0 (
    echo.
    echo ========================================
    echo Forge package configuration failed!
    echo ========================================
    exit /b %ERRORLEVEL%
)

REM Build Forge
echo.
echo Building Forge...
cmake --build .

if %ERRORLEVEL% neq 0 (
    echo.
    echo ========================================
    echo Forge package build failed!
    echo ========================================
    exit /b %ERRORLEVEL%
)

REM Install Forge
echo.
echo Installing Forge package...
cmake --install .

if %ERRORLEVEL% neq 0 (
    echo.
    echo ========================================
    echo Forge package installation failed!
    echo ========================================
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================
echo Forge package built successfully!
echo ========================================
echo Installed to: %INSTALL_DIR%
echo.

exit /b 0
