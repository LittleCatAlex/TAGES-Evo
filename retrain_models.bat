@echo off
setlocal
cd /d "%~dp0"

set "PY_EXE="
if exist ".venv\Scripts\python.exe" set "PY_EXE=.venv\Scripts\python.exe"
if "%PY_EXE%"=="" if exist "..\.venv\Scripts\python.exe" set "PY_EXE=..\.venv\Scripts\python.exe"
if "%PY_EXE%"=="" set "PY_EXE=python"

%PY_EXE% scripts\show_summary.py --header train
if %errorlevel% neq 0 goto :error_exit

%PY_EXE% scripts\show_summary.py --train-step 1
%PY_EXE% scripts\analyze_walther_transitions.py
if %errorlevel% neq 0 goto :error_exit
%PY_EXE% scripts\build_petrophysics_dataset.py
if %errorlevel% neq 0 goto :error_exit
%PY_EXE% scripts\analyze_rock_physics_and_logs.py
if %errorlevel% neq 0 goto :error_exit

%PY_EXE% scripts\show_summary.py --train-step 2
%PY_EXE% scripts\train_well_seq_model.py
if %errorlevel% neq 0 goto :error_exit

%PY_EXE% scripts\show_summary.py --train-step 3
%PY_EXE% scripts\integrate_school_boreholes.py
if %errorlevel% neq 0 goto :error_exit

%PY_EXE% scripts\show_summary.py --train-step 4
%PY_EXE% scripts\train_distilled_model.py
if %errorlevel% neq 0 goto :error_exit

%PY_EXE% scripts\show_summary.py --train-step 5
%PY_EXE% scripts\generate_taiwan_basin_voxels.py
if %errorlevel% neq 0 goto :error_exit

%PY_EXE% scripts\show_summary.py --train-step 6
%PY_EXE% scripts\build_taiwan_portal_data.py
if %errorlevel% neq 0 goto :error_exit

%PY_EXE% scripts\show_summary.py --train

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
