@echo off
rem Double click this file to open today's problem.
rem It calls start.py, which installs what it needs the first time.

cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  py start.py %*
  goto done
)

where python >nul 2>nul
if %errorlevel%==0 (
  python start.py %*
  goto done
)

echo.
echo Python was not found on this computer.
echo Install it from https://www.python.org/downloads/ and tick
echo "Add python.exe to PATH" on the first screen of the installer.
echo.

:done
echo.
pause
