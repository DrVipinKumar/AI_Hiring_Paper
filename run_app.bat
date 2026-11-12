@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================
echo  Hiring-pipeline analysis toolkit - launcher
echo ============================================
echo.

REM --- Check Python ---
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found on PATH.
    echo         Install Python 3.10+ from https://www.python.org/ and re-run.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo [INFO] %%v

REM --- Check/install dependencies ---
echo.
echo [INFO] Checking dependencies from requirements.txt ...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

REM --- Self-test ---
echo.
echo [INFO] Running metric self-tests ...
python test_metrics.py
if errorlevel 1 (
    echo [ERROR] Self-tests failed. Fix issues before running analysis.
    pause
    exit /b 1
)

REM --- Resolve data file (optional first argument) ---
if "%~1"=="" (
    set "DATA_FILE=testing_data.csv"
    set "OUTDIR=result_output"
    set "GROUPS=gender region"
) else (
    set "DATA_FILE=%~1"
    set "OUTDIR=results"
    if "%~2"=="" (set "GROUPS=gender region") else (set "GROUPS=%~2 %~3 %~4 %~5")
)

if not exist "%DATA_FILE%" (
    if "%DATA_FILE%"=="testing_data.csv" (
        echo [INFO] testing_data.csv not found - generating it ...
        python generate_testing_data.py --out testing_data.csv
    ) else (
        echo [ERROR] Data file not found: %DATA_FILE%
        pause
        exit /b 1
    )
)

REM --- Run analysis ---
echo.
echo [INFO] Running analysis on %DATA_FILE% ...
python analyze_hiring.py --data "%DATA_FILE%" --outdir "%OUTDIR%" --group-cols %GROUPS%
if errorlevel 1 (
    echo [ERROR] Analysis failed.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] All checks passed. Results written to result_output\
pause
endlocal