@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -u engine.py
exit /b %ERRORLEVEL%
