#!/usr/bin/env python3
"""
Quick script to test GPU availability and PyTorch CUDA support.
Run this on SLURM nodes to diagnose CUDA issues.
"""

import sys
import os

print("="*60)
print("GPU and CUDA Availability Test")
print("="*60)

# System info
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print()

# Environment variables
print("CUDA Environment Variables:")
print(f"  CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
print(f"  CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")
print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
ld_lib_path = os.environ.get('LD_LIBRARY_PATH', 'Not set')
if len(ld_lib_path) > 100:
    ld_lib_path = ld_lib_path[:100] + "..."
print(f"  LD_LIBRARY_PATH: {ld_lib_path}")
print()

# Check for nvidia-smi
try:
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used', '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        print("GPU Status (nvidia-smi):")
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 4:
                    print(f"  GPU {parts[0]}: {parts[1]}, {parts[2]}MB total, {parts[3]}MB used")
    else:
        print("nvidia-smi failed or no GPUs detected")
except Exception as e:
    print(f"nvidia-smi not available: {e}")
print()

# Check PyTorch
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"PyTorch CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f}GB")
        
        # Test tensor creation on GPU
        try:
            device = torch.device('cuda:0')
            test_tensor = torch.randn(100, 100).to(device)
            result = torch.matmul(test_tensor, test_tensor.T)
            print("✓ GPU tensor operations working")
        except Exception as e:
            print(f"✗ GPU tensor operations failed: {e}")
    else:
        print("PyTorch CUDA not available")
        
        # Diagnostic information
        try:
            # Check if CUDA toolkit is available
            cuda_version = torch.version.cuda
            if cuda_version:
                print(f"  PyTorch was compiled with CUDA {cuda_version}")
            else:
                print("  PyTorch was compiled without CUDA support")
        except:
            pass
            
except ImportError as e:
    print(f"PyTorch not available: {e}")

print()
print("="*60)
print("Test completed")
print("="*60)
