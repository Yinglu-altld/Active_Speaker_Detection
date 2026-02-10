@echo off
setlocal

REM Runs the VAD test using the venv Python (no activation required).

set ROOT=%~dp0
set PY=%ROOT%.venv\Scripts\python.exe
set CFG=%ROOT%config.json

if not exist "%PY%" (
  echo ERROR: venv python not found: "%PY%"
  echo Activate your venv or create it first.
  pause
  exit /b 1
)

if not exist "%CFG%" (
  echo ERROR: config not found: "%CFG%"
  pause
  exit /b 1
)

"%PY%" -m furhat_asd.tools.vad_test --config "%CFG%" --interval-ms 250
pause
