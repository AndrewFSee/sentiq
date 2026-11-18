@echo off
rem Live 8-K collector window: runs scripts\collect_service.py, which loops `newsfeed collect
rem --once` + health check + 120 s pause and holds data\state\collector.lock. The boot task
rem \Sentiq\Collector (scripts\install_boot_collector.ps1) normally holds that lock already, since
rem it starts at boot before anyone logs in; this window then stands by, shows the active
rem collector's last cycle in its title, and takes over if the boot collector stops.
rem   scripts\collect_window.cmd          run in a labelled window
rem   scripts\collect_window.cmd once     one cycle, then exit (smoke test; refused if active)
rem To stop ALL collection: close this window AND, in an admin PowerShell,
rem   Stop-ScheduledTask -TaskPath \Sentiq\ -TaskName Collector
rem Logs: data\collect.log (cycles), data\collector-service.log (start / take-over / failures).
rem Uses uv's *base* Python (see scripts\nf.py for why not the venv's python.exe).
setlocal
set "PY=%APPDATA%\uv\python\cpython-3.11.12-windows-x86_64-none\python.exe"
title sentiq - collector
cd /d "%~dp0.."
if not exist "%PY%" (
    echo Base Python not found: %PY%
    echo Install it with: uv python install 3.11.12
    title sentiq - collector [NO PYTHON]
    pause
    goto :eof
)
if /i "%~1"=="once" (
    "%PY%" scripts\collect_service.py --mode window --once
    goto :eof
)
"%PY%" scripts\collect_service.py --mode window
echo Collector loop exited (code %errorlevel%) - see data\collector-service.log
title sentiq - collector [STOPPED]
pause
