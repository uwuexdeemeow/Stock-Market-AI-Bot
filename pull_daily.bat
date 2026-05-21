@echo off
REM pull_daily.bat - Windows cmd.exe equivalent of pull_daily.sh.
REM Mirrors the same three-step pull so you can double-click from
REM File Explorer instead of opening a bash shell.  See pull_daily.sh
REM for the inline explanations of each step.

setlocal

echo [pull_daily] fetching origin...
git fetch origin
if errorlevel 1 goto :error

echo [pull_daily] fast-forwarding main...
git checkout main
if errorlevel 1 goto :error
git pull --ff-only origin main
if errorlevel 1 goto :error

echo [pull_daily] copying signals/latest into working tree...
git checkout origin/signals/latest -- signals/ logs/
if errorlevel 1 goto :error

echo [pull_daily] un-staging signal/log files...





]\
git reset HEAD signals/ logs/ >nul 2>&1

echo.
echo [pull_daily] done. Latest dashboard inputs:
dir signals\alpaca_paper_log.csv signals\monitor_heartbeat.json signals\factor_data_health.json 2>nul
dir /b /o-d logs\daily_run_*.json 2>nul

endlocal
exit /b 0

:error
echo.
echo [pull_daily] FAILED — see git error above.
echo   Common cause: local uncommitted changes on main blocking the fast-forward.
echo   Fix: `git status` to see what's modified, then commit or stash, then re-run.
endlocal
exit /b 1
