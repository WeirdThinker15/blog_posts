@echo off
setlocal enabledelayedexpansion

:: Input arguments
set "INPUT_FILE=%~1"
set "HASH_FILE=%~2"

:: Validate input
if "%INPUT_FILE%"=="" (
    echo Usage: generate_hash.cmd input_file [hash_output_file]
    exit /b 1
)

if not exist "%INPUT_FILE%" (
    echo ❌ Input file "%INPUT_FILE%" not found.
    exit /b 1
)

:: Set default hash file name if not provided
if "%HASH_FILE%"=="" (
    set "HASH_FILE=%INPUT_FILE%.sha256"
)

:: Compute SHA256 hash and capture only the actual hash line
for /f "skip=1 tokens=1" %%i in ('certutil -hashfile "%INPUT_FILE%" SHA256 ^| findstr /r /v "^$" ^| findstr /v /i "certutil"') do (
    set "HASH=%%i"
    goto done
)
:done

:: Write hash to output file
echo !HASH! > "%HASH_FILE%"
echo ✅ Hash written to "%HASH_FILE%"

endlocal
