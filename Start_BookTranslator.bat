@echo off
chcp 65001 >nul
title BookTranslator
echo ===================================================
echo              BookTranslator Launcher
echo ===================================================
echo.

set HF_HOME=%~dp0hf_cache
echo [i] Hugging Face cache redirected to: %HF_HOME%
echo.

if not exist ".venv\" (
    echo [1/2] First run detected. Creating virtual environment...
    python -m venv .venv
    
    echo [2/2] Installing required dependencies...
    .venv\Scripts\python.exe -m pip install -q -e .
    echo.
)

echo [i] Starting application...
start "" .venv\Scripts\pythonw.exe -m src.launcher.gui
exit
