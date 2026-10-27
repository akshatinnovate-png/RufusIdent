@echo off
title RufusIdent recognition backend
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python server.py
pause
