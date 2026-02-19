@echo off
:: ================================================================
:: Real-Time ASR Overlay — Portable Launcher
:: Place this alongside main.py and a venv/ folder on a USB stick.
:: ================================================================

:: Resolve the directory this .bat lives in (works from any drive)
set "APP_DIR=%~dp0"

:: Try portable venv first, then fall back to system Python
if exist "%APP_DIR%venv\Scripts\python.exe" (
    set "PYTHON=%APP_DIR%venv\Scripts\python.exe"
) else if exist "%APP_DIR%python\python.exe" (
    set "PYTHON=%APP_DIR%python\python.exe"
) else (
    set "PYTHON=python"
)

echo ============================================
echo   Real-Time ASR Overlay
echo   Starting Streamlit dashboard...
echo ============================================
echo.
echo Using Python: %PYTHON%
echo.

"%PYTHON%" -m streamlit run "%APP_DIR%main.py" --server.headless true --server.port 8501

pause
