@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found. Running setup first...
  call setup.bat
)

echo Installing OpenGL GPU preview/render dependencies...
".venv\Scripts\python.exe" -m pip install moderngl
if errorlevel 1 goto fail

echo.
echo OpenGL setup complete. Select "OpenGL GPU" in the app.
pause
exit /b 0

:fail
echo.
echo OpenGL setup failed. Check Python, pip, and graphics driver support.
pause
exit /b 1

