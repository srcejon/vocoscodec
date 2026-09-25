@echo off
rem Configure and build vocoscodec with MSVC 2022 and Ninja.
rem Usage: build.bat [Release|Debug] [extra CMake arguments, e.g. -DVOCOSCODEC_BUILD_TESTS=OFF]
rem Output goes to build\<config>; run the tests with test.bat.
setlocal
set CONFIG=Release
set EXTRA=%*
if /i "%~1"=="Release" goto have_config
if /i "%~1"=="Debug" goto have_config
goto build
:have_config
set CONFIG=%~1
set EXTRA=
rem Everything after the first word, with any "=" kept intact.
for /f "tokens=1,*" %%a in ("%*") do set EXTRA=%%b

:build
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1

set BUILD=%~dp0build\%CONFIG%
cmake -S "%~dp0." -B "%BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=%CONFIG% ^
  -DVOCOSCODEC_BUILD_TESTS=ON -DVOCOSCODEC_BUILD_TOOLS=ON %EXTRA%
if errorlevel 1 exit /b 1
cmake --build "%BUILD%"
if errorlevel 1 exit /b 1
