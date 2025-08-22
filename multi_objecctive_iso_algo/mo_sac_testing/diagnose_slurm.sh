#!/bin/bash

#SBATCH --job-name=mo_sac_diagnostic
#SBATCH --output=slurm_diagnostic_%j.out
#SBATCH --error=slurm_diagnostic_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

# SLURM Diagnostic Script for MO-SAC Environment
# This script helps diagnose common issues with SLURM cluster setup
#
# Usage: sbatch ./diagnose_slurm.sh

echo "=========================================="
echo "SLURM Diagnostic Script for MO-SAC"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "=========================================="

# Get the original script location
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    ORIG_DIR="$SLURM_SUBMIT_DIR"
else
    ORIG_DIR="$(dirname "${BASH_SOURCE[0]}")"
    ORIG_DIR="$(cd "$ORIG_DIR" && pwd)"
fi

echo "1. DIRECTORY INFORMATION:"
echo "   Submit Directory: $SLURM_SUBMIT_DIR"
echo "   Original Directory: $ORIG_DIR"
echo "   Current Directory: $(pwd)"
echo "   SLURM_TMPDIR: $SLURM_TMPDIR"
echo "   USER: $USER"
echo "   HOME: $HOME"
echo ""

echo "2. PYTHON ENVIRONMENT:"
# Try different Python commands
for cmd in python3 python python3.8 python3.9 python3.10 python3.11; do
    if command -v $cmd &> /dev/null; then
        echo "   Found: $cmd at $(which $cmd)"
        echo "   Version: $($cmd --version 2>&1)"
    fi
done
echo ""

echo "3. PYTHON PACKAGES CHECK:"
PYTHON_CMD=""
for cmd in python3 python; do
    if command -v $cmd &> /dev/null; then
        PYTHON_CMD=$cmd
        break
    fi
done

if [ -n "$PYTHON_CMD" ]; then
    echo "   Using: $PYTHON_CMD"
    
    # Check essential packages
    packages=("torch" "numpy" "gymnasium" "matplotlib" "tensorboard")
    for pkg in "${packages[@]}"; do
        if $PYTHON_CMD -c "import $pkg" 2>/dev/null; then
            version=$($PYTHON_CMD -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo "unknown")
            echo "   ✓ $pkg ($version)"
        else
            echo "   ✗ $pkg (missing)"
        fi
    done
else
    echo "   No Python interpreter found!"
fi
echo ""

echo "4. GPU INFORMATION:"
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "   CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    if command -v nvidia-smi &> /dev/null; then
        echo "   GPU Status:"
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,temperature.gpu --format=csv,noheader,nounits | sed 's/^/     /'
        echo ""
        echo "   CUDA Version:"
        nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1 | sed 's/^/     Driver: /'
        if command -v nvcc &> /dev/null; then
            nvcc --version | grep release | sed 's/^/     /'
        fi
    else
        echo "   nvidia-smi not available"
    fi
    
    # Test PyTorch CUDA
    if [ -n "$PYTHON_CMD" ] && $PYTHON_CMD -c "import torch" 2>/dev/null; then
        cuda_available=$($PYTHON_CMD -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
        cuda_count=$($PYTHON_CMD -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
        echo "   PyTorch CUDA Available: $cuda_available"
        echo "   PyTorch CUDA Devices: $cuda_count"
    fi
else
    echo "   No GPU devices allocated"
fi
echo ""

echo "5. FILE SYSTEM PERMISSIONS:"
echo "   Original directory permissions:"
ls -la "$ORIG_DIR" | head -10 | sed 's/^/     /'

if [ -n "$SLURM_TMPDIR" ]; then
    echo "   SLURM_TMPDIR permissions:"
    ls -la "$SLURM_TMPDIR" | head -5 | sed 's/^/     /'
    
    echo "   Testing write access to SLURM_TMPDIR:"
    if mkdir -p "$SLURM_TMPDIR/test_write" 2>/dev/null; then
        echo "     ✓ Can create directories in SLURM_TMPDIR"
        rmdir "$SLURM_TMPDIR/test_write" 2>/dev/null
    else
        echo "     ✗ Cannot create directories in SLURM_TMPDIR"
    fi
fi

echo "   Testing write access to /tmp:"
if mkdir -p "/tmp/test_write_$$" 2>/dev/null; then
    echo "     ✓ Can create directories in /tmp"
    rmdir "/tmp/test_write_$$" 2>/dev/null
else
    echo "     ✗ Cannot create directories in /tmp"
fi

echo "   Testing write access to current directory:"
if mkdir -p "./test_write" 2>/dev/null; then
    echo "     ✓ Can create directories in current directory"
    rmdir "./test_write" 2>/dev/null
else
    echo "     ✗ Cannot create directories in current directory"
fi
echo ""

echo "6. PROJECT FILES CHECK:"
echo "   Files in original directory:"
ls -la "$ORIG_DIR" | sed 's/^/     /'
echo ""

# Check for specific required files
required_files=(
    "test_mo_sac.py"
    "train_energynet.py"
    "test_environments.py"
    "../multi_objective_sac.py"
    "../EnergyNetMoISO/MoISOEnv.py"
)

echo "   Required files status:"
for file in "${required_files[@]}"; do
    if [ -f "$ORIG_DIR/$file" ]; then
        echo "     ✓ $file"
    else
        echo "     ✗ $file (missing)"
    fi
done
echo ""

echo "7. ENVIRONMENT VARIABLES:"
echo "   PATH: $PATH"
echo "   LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "   PYTHONPATH: $PYTHONPATH"
echo "   OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo ""

echo "8. LOADED MODULES:"
if command -v module &> /dev/null; then
    echo "   Currently loaded modules:"
    module list 2>&1 | sed 's/^/     /'
    echo ""
    echo "   Available Python modules:"
    module avail python 2>&1 | grep -i python | sed 's/^/     /'
    echo ""
    echo "   Available CUDA modules:"
    module avail cuda 2>&1 | grep -i cuda | sed 's/^/     /'
else
    echo "   Module system not available"
fi
echo ""

echo "9. SYSTEM INFORMATION:"
echo "   Hostname: $(hostname)"
echo "   OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2 2>/dev/null || uname -a)"
echo "   Kernel: $(uname -r)"
echo "   Architecture: $(uname -m)"
echo "   Memory: $(free -h | grep Mem | awk '{print $2}' 2>/dev/null || echo "unknown")"
echo "   CPU: $(nproc 2>/dev/null || echo "unknown") cores"
echo ""

echo "10. QUICK FUNCTIONALITY TEST:"
if [ -n "$PYTHON_CMD" ] && [ -f "$ORIG_DIR/test_environments.py" ]; then
    echo "   Testing import of test environments..."
    if $PYTHON_CMD -c "
import sys
sys.path.insert(0, '$ORIG_DIR')
try:
    from test_environments import MultiObjectiveContinuousCartPole
    print('     ✓ Test environments import successfully')
except Exception as e:
    print(f'     ✗ Test environments import failed: {e}')
" 2>/dev/null; then
        :
    else
        echo "     ✗ Test environments import failed"
    fi
    
    echo "   Testing MO-SAC algorithm import..."
    if $PYTHON_CMD -c "
import sys
sys.path.insert(0, '$ORIG_DIR/..')
try:
    from multi_objective_sac import MultiObjectiveSAC
    print('     ✓ MO-SAC algorithm imports successfully')
except Exception as e:
    print(f'     ✗ MO-SAC algorithm import failed: {e}')
" 2>/dev/null; then
        :
    else
        echo "     ✗ MO-SAC algorithm import failed"
    fi
else
    echo "   Cannot run functionality tests (missing Python or files)"
fi
echo ""

echo "=========================================="
echo "Diagnostic Complete"
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Check slurm_diagnostic_${SLURM_JOB_ID}.out for full output"
echo "Check slurm_diagnostic_${SLURM_JOB_ID}.err for any errors"
echo ""
echo "Common solutions based on diagnostic results:"
echo ""
echo "If Python packages are missing:"
echo "  pip install torch gymnasium matplotlib tensorboard numpy"
echo ""
echo "If file permissions are an issue:"
echo "  Use the v2 scripts (run_test_mo_sac_v2.sh, run_train_energynet_v2.sh)"
echo ""
echo "If GPU is not available:"
echo "  Check SLURM allocation: sbatch --gres=gpu:1 ..."
echo "  Load CUDA module if needed: module load cuda"
echo ""
echo "If imports fail:"
echo "  Check file paths and PYTHONPATH"
echo "  Ensure all required files are present"
echo ""
echo "=========================================="
