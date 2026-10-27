@echo off
title RufusIdent backend setup
cd /d "%~dp0"
python --version >nul 2>nul
if errorlevel 1 (
  echo Python 3.10 or newer is needed: https://www.python.org/downloads/  ^(tick "Add python.exe to PATH"^)
  pause
  exit /b 1
)
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || (pause & exit /b 1)
python setup_models.py || (pause & exit /b 1)
python benchmark.py
echo.
echo Setup done. Next: put photos in photos\^<admission_no^>\ and run enroll.bat
pause
