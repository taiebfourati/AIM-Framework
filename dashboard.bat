@echo off
REM ---------------------------------------------------------------------------
REM dashboard.bat - one-shot launcher for the AIMP x Simu5G dashboard.
REM
REM Usage (from anywhere, or double-click in Explorer):
REM   dashboard.bat                       default (Live RAN, port 8765)
REM   dashboard.bat --port 9000           override port
REM   dashboard.bat --skip-safety         bypass the boot-time preflight
REM
REM Sequence:
REM   1. Switch into the repo root (this script's directory) so Python can
REM      import the dashboard package.
REM   2. Invoke the project venv's interpreter with -m dashboard.live, which
REM      runs the safety preflight first and then starts uvicorn.
REM ---------------------------------------------------------------------------
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -m dashboard.live %*
