@echo off
REM Multi-Objective SAC Testing Script for Windows
REM This script runs comprehensive tests on all available test environments
REM 
REM Usage: 
REM   run_test_mo_sac.bat                    - Run comprehensive tests on all environments
REM   run_test_mo_sac.bat CartPole           - Test specific environment (partial name match)
REM   run_test_mo_sac.bat Pendulum           - Test specific environment (partial name match)

echo ==========================================
echo Multi-Objective SAC Testing Script
echo ==========================================

REM Set the working directory to the script location
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if required packages are installed
echo Checking dependencies...
python -c "import torch, numpy, gymnasium, matplotlib" >nul 2>&1
if errorlevel 1 (
    echo Warning: Some required packages might be missing. Installing...
    pip install -r requirements.txt
)

REM Create necessary directories
echo Creating output directories...
if not exist "models" mkdir models
if not exist "plots" mkdir plots
if not exist "logs" mkdir logs

REM Set environment variables for better performance (optional)
set CUDA_VISIBLE_DEVICES=0
set OMP_NUM_THREADS=4

REM Environment selection
if "%1"=="" (
    echo Running comprehensive tests on all environments...
    echo This will take a while ^(30-60 minutes depending on hardware^)
    echo.
    
    REM Run comprehensive tests with default parameters
    python test_mo_sac.py
    
) else (
    REM Run test on specific environment (partial name matching)
    set ENV_NAME=%1
    echo Running test on environment containing: %ENV_NAME%
    echo.
    
    REM Available environments:
    REM - MultiObjectiveContinuousCartPole-v0
    REM - MultiObjectiveMountainCarContinuous-v0  
    REM - MultiObjectivePendulum-v0
    REM - MultiObjectiveLunarLander-v0
    
    REM Find matching environment
    if /i "%ENV_NAME:CartPole=%" neq "%ENV_NAME%" (
        set FULL_ENV_NAME=MultiObjectiveContinuousCartPole-v0
    ) else if /i "%ENV_NAME:Mountain=%" neq "%ENV_NAME%" (
        set FULL_ENV_NAME=MultiObjectiveMountainCarContinuous-v0
    ) else if /i "%ENV_NAME:car=%" neq "%ENV_NAME%" (
        set FULL_ENV_NAME=MultiObjectiveMountainCarContinuous-v0
    ) else if /i "%ENV_NAME:Pendulum=%" neq "%ENV_NAME%" (
        set FULL_ENV_NAME=MultiObjectivePendulum-v0
    ) else if /i "%ENV_NAME:Lunar=%" neq "%ENV_NAME%" (
        set FULL_ENV_NAME=MultiObjectiveLunarLander-v0
    ) else if /i "%ENV_NAME:lander=%" neq "%ENV_NAME%" (
        set FULL_ENV_NAME=MultiObjectiveLunarLander-v0
    ) else (
        echo Unknown environment: %ENV_NAME%
        echo Available environments:
        echo   - CartPole ^(MultiObjectiveContinuousCartPole-v0^)
        echo   - Mountain ^(MultiObjectiveMountainCarContinuous-v0^)
        echo   - Pendulum ^(MultiObjectivePendulum-v0^)
        echo   - Lunar ^(MultiObjectiveLunarLander-v0^)
        pause
        exit /b 1
    )
    
    echo Testing environment: %FULL_ENV_NAME%
    python test_mo_sac.py "%FULL_ENV_NAME%"
)

REM Check if the script completed successfully
if errorlevel 1 (
    echo.
    echo ==========================================
    echo Testing failed with errors!
    echo ==========================================
    echo Check the error messages above for troubleshooting.
    pause
    exit /b 1
) else (
    echo.
    echo ==========================================
    echo Testing completed successfully!
    echo ==========================================
    echo Results saved in:
    echo   - models\     ^(trained model files^)
    echo   - plots\      ^(training plots and analysis^)
    echo   - runs\       ^(tensorboard logs^)
    echo.
    echo To view tensorboard logs:
    echo   tensorboard --logdir runs/
    echo.
    echo To view plots:
    echo   dir plots\*.png
    echo.
    pause
)
