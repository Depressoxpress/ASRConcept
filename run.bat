@echo off
:: ================================================================
:: Real-Time ASR Overlay — Portable Launcher
:: Place this alongside main.py and a venv/ folder on a USB stick.
:: ================================================================

:: Suppress HuggingFace warnings (set BEFORE Python starts)
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "HF_HUB_DISABLE_IMPLICIT_TOKEN=1"
set "TOKENIZERS_PARALLELISM=false"

:: Resolve the directory this .bat lives in (works from any drive)
set "APP_DIR=%~dp0"

:: Try portable venv first, then fall back to system Python
if exist "%APP_DIR%venv\Scripts\python.exe" (
    set "PYTHON=%APP_DIR%venv\Scripts\python.exe"
) else if exist "%APP_DIR%.venv\Scripts\python.exe" (
    set "PYTHON=%APP_DIR%.venv\Scripts\python.exe"
) else if exist "%APP_DIR%python\python.exe" (
    set "PYTHON=%APP_DIR%python\python.exe"
) else (
    set "PYTHON=python"
)

:: ----------------------------------------------------------------
:: CLEANUP: Kill any previous instances to free Port 8501 & Overlay
:: ----------------------------------------------------------------
echo Cleaning up previous sessions...
:: 1. Kill any process listening on Port 8501 (Streamlit)
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8501" ^| find "LISTENING"') do taskkill /f /pid %%a >nul 2>&1

:: 2. Kill any lingering overlay.py processes
wmic process where "CommandLine like '%%src%%overlay.py%%'" call terminate >nul 2>&1

:: 3. Kill any lingering streamlit processes by name just in case
taskkill /F /IM "streamlit.exe" >nul 2>&1

timeout /t 1 /nobreak >nul

echo ============================================
echo   Real-Time ASR Overlay
echo   Starting Streamlit dashboard...
echo ============================================
echo.
echo Using Python: %PYTHON%
echo.

:: ----------------------------------------------------------------
:: DEPENDENCY CHECK: Ensure packages are installed in this environment
:: ----------------------------------------------------------------
"%PYTHON%" -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ------------------------------------------------------------
    echo   Streamlit not found in this environment.
    echo   Installing dependencies from requirements.txt...
    echo ------------------------------------------------------------
    "%PYTHON%" -m pip install -r "%APP_DIR%requirements.txt"
    if %errorlevel% neq 0 (
        echo.
        echo FAIL: Could not install dependencies.
        pause
        exit /b
    )
    echo.
    echo Dependencies installed successfully.
    echo.
)

"%PYTHON%" -m streamlit run "%APP_DIR%main.py" --server.headless true --server.port 8501

pause
