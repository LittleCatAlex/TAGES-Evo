@echo off
setlocal
cd /d "%~dp0"

set "PY_EXE="
if exist ".venv\Scripts\python.exe" set "PY_EXE=.venv\Scripts\python.exe"
if "%PY_EXE%"=="" if exist "..\.venv\Scripts\python.exe" set "PY_EXE=..\.venv\Scripts\python.exe"
if "%PY_EXE%"=="" set "PY_EXE=python"

%PY_EXE% scripts\show_summary.py --header test
if %errorlevel% neq 0 goto :error_exit

%PY_EXE% scripts\show_summary.py --test-step 1
%PY_EXE% scripts\audit_physics_loss_activation.py
if %errorlevel% neq 0 goto :error_exit

%PY_EXE% scripts\show_summary.py --test-step 2
%PY_EXE% scripts\run_ssr_analysis.py
if %errorlevel% neq 0 goto :error_exit

%PY_EXE% scripts\show_summary.py --test-step 3
%PY_EXE% scripts\plot_site_response_facies.py
if %errorlevel% neq 0 goto :error_exit

%PY_EXE% scripts\show_summary.py --test-step 4
%PY_EXE% scripts\validate_basin_structures.py
if %errorlevel% neq 0 goto :error_exit

%PY_EXE% scripts\show_summary.py --test-step 5
%PY_EXE% scripts\update_manuscript_verified.py
if %errorlevel% neq 0 goto :error_exit

%PY_EXE% scripts\show_summary.py --test-step 6
%PY_EXE% -m pytest tests\test_manuscript_consistency.py -v
if %errorlevel% neq 0 goto :error_exit

%PY_EXE% -m pytest tests\ -q
if %errorlevel% neq 0 goto :error_exit

%PY_EXE% scripts\show_summary.py --report

if /i "%~1"=="--no-pause" goto :skip_pause
if /i "%~1"=="-n" goto :skip_pause
if defined NO_PAUSE goto :skip_pause
if defined CI goto :skip_pause
pause
:skip_pause
exit /b 0

:error_exit
%PY_EXE% scripts\show_summary.py --error %errorlevel%
if /i "%~1"=="--no-pause" goto :skip_err_pause
if /i "%~1"=="-n" goto :skip_err_pause
if defined NO_PAUSE goto :skip_err_pause
if defined CI goto :skip_err_pause
pause
:skip_err_pause
exit /b %errorlevel%
