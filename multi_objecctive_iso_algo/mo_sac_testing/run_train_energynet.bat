@echo off
REM Multi-Objective SAC Training Script for EnergyNet MoISO Environment (Windows)
REM This script trains the MO-SAC algorithm on the EnergyNet Multi-Objective ISO environment
REM
REM Usage: run_train_energynet.bat [OPTIONS]
REM 
REM Examples:
REM   run_train_energynet.bat                    - Run with default parameters
REM   run_train_energynet.bat quick-test         - Quick test run (shorter training)
REM   run_train_energynet.bat cost-priority      - Prioritize cost reduction over stability
REM   run_train_energynet.bat stability-priority - Prioritize stability over cost

echo ==========================================
echo MO-SAC EnergyNet Training Script
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
python -c "import torch, numpy, gymnasium, matplotlib, tensorboard" >nul 2>&1
if errorlevel 1 (
    echo Warning: Some required packages might be missing. Installing...
    pip install -r requirements.txt
)

REM Create necessary directories
echo Creating output directories...
if not exist "energynet_experiments" mkdir energynet_experiments
if not exist "energynet_experiments\models" mkdir energynet_experiments\models
if not exist "energynet_experiments\logs" mkdir energynet_experiments\logs
if not exist "energynet_experiments\plots" mkdir energynet_experiments\plots

REM Set environment variables for better performance (optional)
set CUDA_VISIBLE_DEVICES=0
set OMP_NUM_THREADS=4

REM Parse command line arguments for quick configurations
set QUICK_TEST=false
set COST_PRIORITY=false
set STABILITY_PRIORITY=false

if /i "%1"=="quick-test" set QUICK_TEST=true
if /i "%1"=="cost-priority" set COST_PRIORITY=true
if /i "%1"=="stability-priority" set STABILITY_PRIORITY=true

REM Set base experiment name with timestamp
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%" & set "Sec=%dt:~12,2%"
set "TIMESTAMP=%YYYY%%MM%%DD%_%HH%%Min%%Sec%"
set "BASE_EXP_NAME=mo_sac_energynet_%TIMESTAMP%"

REM Configure training parameters based on flags
if "%QUICK_TEST%"=="true" (
    echo Running quick test configuration...
    set TOTAL_TIMESTEPS=50000
    set LEARNING_STARTS=2000
    set EVAL_FREQ=10000
    set SAVE_FREQ=25000
    set WEIGHTS=0.6 0.4
    set EXP_NAME=%BASE_EXP_NAME%_quick
    
) else if "%COST_PRIORITY%"=="true" (
    echo Running cost-priority configuration...
    set TOTAL_TIMESTEPS=1000000
    set LEARNING_STARTS=10000
    set EVAL_FREQ=20000
    set SAVE_FREQ=100000
    set WEIGHTS=0.8 0.2
    set EXP_NAME=%BASE_EXP_NAME%_cost_priority
    
) else if "%STABILITY_PRIORITY%"=="true" (
    echo Running stability-priority configuration...
    set TOTAL_TIMESTEPS=1000000
    set LEARNING_STARTS=10000
    set EVAL_FREQ=20000
    set SAVE_FREQ=100000
    set WEIGHTS=0.3 0.7
    set EXP_NAME=%BASE_EXP_NAME%_stability_priority
    
) else (
    echo Running default configuration...
    set TOTAL_TIMESTEPS=1000000
    set LEARNING_STARTS=10000
    set EVAL_FREQ=20000
    set SAVE_FREQ=100000
    set WEIGHTS=0.6 0.4
    set EXP_NAME=%BASE_EXP_NAME%_default
)

echo Training configuration:
echo   Experiment name: %EXP_NAME%
echo   Total timesteps: %TOTAL_TIMESTEPS%
echo   Weights: %WEIGHTS%
echo.

REM Run the training with all configurable parameters
python train_energynet.py ^
    --experiment-name "%EXP_NAME%" ^
    --save-dir "energynet_experiments" ^
    ^
    --total-timesteps %TOTAL_TIMESTEPS% ^
    --learning-starts %LEARNING_STARTS% ^
    --eval-freq %EVAL_FREQ% ^
    --save-freq %SAVE_FREQ% ^
    ^
    --weights %WEIGHTS% ^
    ^
    --actor-lr 3e-4 ^
    --critic-lr 3e-4 ^
    --alpha-lr 3e-4 ^
    ^
    --gamma 0.99 ^
    --tau 0.005 ^
    ^
    --buffer-size 1000000 ^
    --batch-size 256 ^
    ^
    --dispatch-strategy "PROPORTIONAL" ^
    ^
    --verbose

REM Store the exit code
set EXIT_CODE=%errorlevel%

REM Report results
if %EXIT_CODE% equ 0 (
    echo.
    echo ==========================================
    echo Training completed successfully!
    echo ==========================================
    echo Results saved in: energynet_experiments\
    echo.
    echo Files created:
    echo   - energynet_experiments\%EXP_NAME%_config.json    ^(configuration^)
    echo   - energynet_experiments\%EXP_NAME%_results.json   ^(training results^)
    echo   - energynet_experiments\models\%EXP_NAME%_final.pth ^(trained model^)
    echo   - energynet_experiments\logs\                      ^(tensorboard logs^)
    echo.
    echo To view training progress:
    echo   tensorboard --logdir energynet_experiments\logs\
    echo.
    echo Parameter Tuning Tips:
    echo   - If learning is slow: increase learning rates ^(--actor-lr, --critic-lr^)
    echo   - If training is unstable: decrease learning rates or increase --tau
    echo   - For different objectives: adjust --weights [cost_weight stability_weight]
    echo   - For longer training: increase --total-timesteps
    echo   - For more evaluation: decrease --eval-freq
    echo.
    echo Analysis Suggestions:
    echo 1. Compare different weight configurations:
    echo    run_train_energynet.bat cost-priority
    echo    run_train_energynet.bat stability-priority
    echo.
    echo 2. Monitor training in real-time:
    echo    tensorboard --logdir energynet_experiments\logs\
    echo.
    
) else (
    echo.
    echo ==========================================
    echo Training failed with errors!
    echo ==========================================
    echo Exit code: %EXIT_CODE%
    echo.
    echo Common troubleshooting:
    echo   1. Check that EnergyNet environment is properly installed
    echo   2. Verify all dependencies are installed: pip install -r requirements.txt
    echo   3. Check CUDA availability if using GPU: python -c "import torch; print(torch.cuda.is_available())"
    echo   4. Try running with quick-test flag for faster debugging
    echo   5. Check the error messages above for specific issues
)

pause
exit /b %EXIT_CODE%

REM ==============================================
REM PARAMETER REFERENCE FOR MANUAL CUSTOMIZATION
REM ==============================================
REM
REM Experiment Configuration:
REM   --experiment-name       : Name for this training run
REM   --save-dir             : Directory to save all results
REM
REM Training Parameters:
REM   --total-timesteps      : Total training steps (1M = ~2-4 hours)
REM   --learning-starts      : Random exploration steps before learning
REM   --eval-freq            : How often to evaluate (lower = more evaluation)
REM   --save-freq            : How often to save model checkpoints
REM
REM Multi-Objective Weights:
REM   --weights              : [cost_weight stability_weight] (must sum to 1.0)
REM                           Examples:
REM                           - 1.0 0.0  (only optimize cost)
REM                           - 0.0 1.0  (only optimize stability)  
REM                           - 0.5 0.5  (equal importance)
REM                           - 0.8 0.2  (prioritize cost)
REM                           - 0.3 0.7  (prioritize stability)
REM
REM Network Learning Rates:
REM   --actor-lr             : Actor network learning rate (1e-5 to 1e-3)
REM   --critic-lr            : Critic network learning rate (1e-5 to 1e-3)
REM   --alpha-lr             : Entropy coefficient learning rate (1e-5 to 1e-3)
REM
REM SAC Algorithm Parameters:
REM   --gamma                : Discount factor (0.9-0.999, higher = more long-term)
REM   --tau                  : Target network update rate (0.001-0.01, lower = more stable)
REM
REM Experience Replay:
REM   --buffer-size          : Replay buffer size (larger = more stable, more memory)
REM   --batch-size           : Training batch size (64-512, larger = more stable)
REM
REM Environment Configuration:
REM   --dispatch-strategy    : How to dispatch energy (PROPORTIONAL, EQUAL, etc.)
REM   --use-dispatch-action  : Enable dispatch actions (add this flag to enable)
REM
REM Logging:
REM   --verbose              : Enable detailed logging (add this flag to enable)
REM   --no-tensorboard       : Disable tensorboard logging (add this flag to disable)
