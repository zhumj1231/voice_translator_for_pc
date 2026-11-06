@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-xai.ps1"
if errorlevel 1 pause
