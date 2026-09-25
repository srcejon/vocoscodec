@echo off
rem Run vocoscodec_benchmark (build.bat Release first) pinned to one logical
rem CPU: set BENCH_CPU to choose it, default 2. On hybrid Intel parts CPUs
rem 0-15 are performance cores; an unpinned run can land on an efficiency
rem core and take twice as long.
rem Usage: benchmark.bat [benchmark options, e.g. --group 9 --codebooks 2]
setlocal
if "%BENCH_CPU%"=="" set BENCH_CPU=2
for /f %%h in ('powershell -NoProfile -Command "[Convert]::ToString([int64]1 -shl %BENCH_CPU%, 16)"') do set MASK=%%h
start /b /wait /affinity %MASK% "" "%~dp0build\Release\vocoscodec\vocoscodec_benchmark.exe" "%~dp0model" %*
