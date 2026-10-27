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
rem Use the laptop's built-in graphics (DirectML) when possible; the backend checks it is faster and gives identical results.
pip uninstall -y onnxruntime >nul 2>nul
pip install onnxruntime-directml || pip install onnxruntime
python setup_models.py || (pause & exit /b 1)
python benchmark.py
echo.
echo Setup done. Next: put photos in photos\^<admission_no^>\ and run enroll.bat
pause
