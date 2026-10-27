@echo off
title RufusIdent - Gulmohur High School
cd /d "%~dp0"
set PORT=8765

python --version >nul 2>nul
if %errorlevel%==0 (
  start "RufusIdent server" /min python server.py %PORT%
) else (
  start "RufusIdent server" /min powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0serve.ps1" -Port %PORT%
)

if exist "backend\.venv\Scripts\python.exe" (
  rem /k keeps the window open if the backend stops, so its error message stays visible
  start "RufusIdent backend" /min cmd /k "cd /d backend && .venv\Scripts\python.exe server.py"
)

timeout /t 2 /nobreak >nul
start "" msedge --app=http://localhost:%PORT%/ --start-maximized
