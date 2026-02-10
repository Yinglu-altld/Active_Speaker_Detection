@echo off
setlocal

REM Runs the 2-speaker DOA test using the venv Python (no activation required).

set ROOT=%~dp0
set PY=%ROOT%.venv\Scripts\python.exe

if not exist "%PY%" (
  echo ERROR: venv python not found: "%PY%"
  echo Create/activate your venv first.
  pause
  exit /b 1
)

"%PY%" "%ROOT%run_2speaker_test.py"
pause

