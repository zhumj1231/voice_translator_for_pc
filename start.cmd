@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Run setup.cmd first to create the Python 3.12 environment.
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -X utf8 "app.py"
