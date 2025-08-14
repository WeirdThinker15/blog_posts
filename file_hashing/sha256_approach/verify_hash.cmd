@echo off
setlocal enabledelayedexpansion

:: Input arguments
set "INPUT_FILE=%~1"
set "HASH_FILE=%~2"

:: Validate input
if "%INPUT_FILE%"=="" (
    echo Usage: verify_hash.cmd input_file hash_file
    exit /b 1
)
if "%HASH_FILE%"=="" (
    echo Usage: verify_hash.cmd input_file hash_file
    exit /b 1
)

if not exist "%INPUT_FILE%" (
    echo Input file "%INPUT_FILE%" not found.
    exit /b 1
)
if not exist "%HASH_FILE%" (
    echo Hash file "%HASH_FILE%" not found.
    exit /b 1
)

:: Compute SHA256 hash of input file
for /f "skip=1 tokens=1" %%i in ('certutil -hashfile "%INPUT_FILE%" SHA256 ^| findstr /r /v "^$" ^| findstr /v /i "certutil"') do (
    set "COMPUTED_HASH=%%i"
    goto done_hash
)
:done_hash

:: Read expected hash from file (first line only)
set /p EXPECTED_HASH=<"%HASH_FILE%"

:: Clean up both values (trim spaces)
for /f %%a in ("!COMPUTED_HASH!") do set "COMPUTED_HASH=%%a"
for /f %%a in ("!EXPECTED_HASH!") do set "EXPECTED_HASH=%%a"

:: Compare (case-insensitive)
if /i "!COMPUTED_HASH!"=="!EXPECTED_HASH!" (
    echo Hash verified successfully.
    exit /b 0
) else (
    echo Hash mismatch!
    echo Expected: !EXPECTED_HASH!
    echo Actual:   !COMPUTED_HASH!
    exit /b 2
)

endlocal
