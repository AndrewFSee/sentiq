@echo off
rem Live 8-K collector for the holdout: one `newsfeed collect --once` cycle, then 120 s pause
rem (= collect.interval_seconds), forever. Each cycle is a fresh process, so one crash doesn't
rem stop collection. Close the window (or Ctrl+C) to stop. Full log: data\collect.log.
rem   scripts\collect_window.cmd          run in a labelled window
rem   scripts\collect_window.cmd once     one cycle, then exit (smoke test)
rem Runs the *base* Python uv installed + scripts\nf.py (adds the venv's packages): Windows Smart
rem App Control blocks the venv's own newsfeed.exe / python.exe (no reputation), not this one.
setlocal
set "LABEL=sentiq - COLLECTING 8-Ks"
set "PY=%APPDATA%\uv\python\cpython-3.11.12-windows-x86_64-none\python.exe"
title %LABEL%
cd /d "%~dp0.."
if not exist "%PY%" (
    echo Base Python not found: %PY%
    echo Install it with: uv python install 3.11.12
    title %LABEL% [NO PYTHON]
    pause
    goto :eof
)

:loop
echo [%date% %time:~0,8%] cycle...
"%PY%" scripts\nf.py collect --once >> data\collect.log 2>&1
if errorlevel 1 (
    echo     FAILED - see data\collect.log
    title %LABEL% [last cycle FAILED]
) else (
    "%PY%" scripts\nf.py collect --check > nul 2>&1
    if errorlevel 1 (
        echo     ok, but health check reports a problem - run: "%PY%" scripts\nf.py collect --check
        title %LABEL% [UNHEALTHY]
    ) else (
        echo     ok
        title %LABEL%
    )
)
if /i "%~1"=="once" goto :eof
"%SystemRoot%\System32\timeout.exe" /t 120 /nobreak > nul
goto loop
