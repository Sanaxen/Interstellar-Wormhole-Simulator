@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found. Running setup first...
  call setup.bat
)

echo Example:
echo   render_cli.bat entrance.jpg exit.jpg output
echo.

if "%~3"=="" (
  echo Usage: render_cli.bat ENTRANCE_PANORAMA EXIT_PANORAMA OUTPUT_DIR
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -m wormhole_app.render_cli --entrance "%~1" --exit "%~2" --output "%~3"
pause

