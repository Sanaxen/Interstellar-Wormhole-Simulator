@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found. Running setup first...
  call setup.bat
)

".venv\Scripts\python.exe" -m wormhole_app
pause

