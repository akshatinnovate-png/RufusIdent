@echo off
title RufusIdent enrolment
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python enroll.py %*
pause
