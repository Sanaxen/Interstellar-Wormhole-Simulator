@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  set PYTHON=py -3
) else (
  where python >nul 2>nul
  if %errorlevel%==0 (
    set PYTHON=python
  ) else (
    set PYTHON="%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
  )
)

if not exist ".venv\Scripts\python.exe" (
  %PYTHON% -m venv .venv
  if errorlevel 1 (
    echo Standard venv bootstrap failed. Retrying without bundled ensurepip...
    if exist ".venv" rmdir /s /q ".venv"
    %PYTHON% -m venv --without-pip .venv
    if errorlevel 1 goto fail
    %PYTHON% -m pip --python ".venv\Scripts\python.exe" install pip
    if errorlevel 1 goto fail
  )
)

".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto fail

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail

echo.
echo Setup complete. Run run_app.bat to start the simulator.
pause
exit /b 0

:fail
echo.
echo Setup failed. Please install Python 3.10 or newer and try again.
pause
exit /b 1
