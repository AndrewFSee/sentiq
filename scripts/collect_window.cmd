@echo off
rem Live 8-K collector for the holdout: one `newsfeed collect --once` cycle, then 120 s pause
rem (= collect.interval_seconds), forever. Each cycle is a fresh process, so one crash doesn't
rem stop collection. Close the window (or Ctrl+C) to stop. Full log: data\collect.log.
rem   scripts\collect_window.cmd          run in a labelled window
rem   scripts\collect_window.cmd once     one cycle, then exit (smoke test)
setlocal
set "LABEL=sentiq - COLLECTING 8-Ks"
title %LABEL%
cd /d "%~dp0.."

:loop
echo [%date% %time:~0,8%] cycle...
uv run python -m newsfeed collect --once >> data\collect.log 2>&1
if errorlevel 1 (
    echo     FAILED - see data\collect.log
    title %LABEL% [last cycle FAILED]
) else (
    uv run python -m newsfeed collect --check > nul 2>&1
    if errorlevel 1 (
        echo     ok, but health check reports a problem - run: uv run python -m newsfeed collect --check
        title %LABEL% [UNHEALTHY]
    ) else (
        echo     ok
        title %LABEL%
    )
)
if /i "%~1"=="once" goto :eof
"%SystemRoot%\System32\timeout.exe" /t 120 /nobreak > nul
goto loop
