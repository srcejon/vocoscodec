@echo off
rem Run the vocoscodec tests of a build made by build.bat.
rem Usage: test.bat [Release|Debug]
setlocal
set CONFIG=%~1
if "%CONFIG%"=="" set CONFIG=Release
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
ctest --test-dir "%~dp0build\%CONFIG%" --output-on-failure
if errorlevel 1 exit /b 1
