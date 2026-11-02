@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run: powershell -ExecutionPolicy Bypass -File setup.ps1
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m src.app %*
if errorlevel 1 pause
