@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONDONTWRITEBYTECODE=1
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
    python main.py %*
) else (
    py -3 main.py %*
)
if errorlevel 1 pause
