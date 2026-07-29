@echo off
REM Wrapper for Windows Task Scheduler.
REM
REM Calling python.exe directly from a scheduled task means fighting cmd's
REM quoting rules around the redirect. A batch file sidesteps that entirely:
REM the scheduler runs this, and this handles its own paths and logging.
REM
REM %~dp0 is the directory this file lives in, with a trailing backslash,
REM so the task works regardless of what working directory it inherits.

cd /d "%~dp0"

echo. >> engine_task.log
echo ======================================================== >> engine_task.log
echo RUN STARTED %DATE% %TIME% >> engine_task.log

REM -u disables Python's output buffering, so the log fills as the run
REM proceeds instead of only when the process exits.
".venv\Scripts\python.exe" -u engine.py >> engine_task.log 2>&1

echo RUN FINISHED %DATE% %TIME% (exit code %ERRORLEVEL%) >> engine_task.log

exit /b %ERRORLEVEL%
