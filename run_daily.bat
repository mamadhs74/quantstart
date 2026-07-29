@echo off
cd /d "%~dp0"
echo ======================================================= >> engine_task.log
echo RUN STARTED %DATE% %TIME% >> engine_task.log
".venv\Scripts\python.exe" -u engine.py >> engine_task.log 2>&1
echo RUN FINISHED %DATE% %TIME% exit code %ERRORLEVEL% >> engine_task.log
exit /b %ERRORLEVEL%
