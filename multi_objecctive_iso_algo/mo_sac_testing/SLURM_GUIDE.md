# SLURM Cluster Guide for MO-SAC Scripts

## Running on SLURM Clusters

The bash scripts have been updated with SLURM directives and cluster-specific configurations.

### Basic Usage

```bash
# Test environments
sbatch -c 4 --gres=gpu:1 ./run_test_mo_sac.sh
sbatch -c 4 --gres=gpu:1 ./run_test_mo_sac.sh CartPole
sbatch -c 4 --gres=gpu:1 ./run_test_mo_sac.sh Pendulum

# Train on EnergyNet
sbatch -c 4 --gres=gpu:1 ./run_train_energynet.sh
sbatch -c 4 --gres=gpu:1 ./run_train_energynet.sh --quick-test
sbatch -c 4 --gres=gpu:1 ./run_train_energynet.sh --cost-priority
```

### SLURM Directives in Scripts

The scripts now include these SLURM directives:

#### Test Script (`run_test_mo_sac.sh`):
```bash
#SBATCH --job-name=mo_sac_test
#SBATCH --output=slurm_test_%j.out
#SBATCH --error=slurm_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
```

#### Training Script (`run_train_energynet.sh`):
```bash
#SBATCH --job-name=mo_sac_energynet
#SBATCH --output=slurm_train_%j.out
#SBATCH --error=slurm_train_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
```

### Customizing for Your Cluster

You may need to modify the scripts for your specific cluster setup:

#### 1. Module Loading
Uncomment and modify the module loading section in both scripts:
```bash
# Add your cluster's modules
module load python/3.8
module load cuda/11.8
module load pytorch/1.12.0
module load gcc/9.3.0
```

#### 2. Resource Requirements
Adjust SLURM directives based on your cluster's resources:
```bash
# For longer training
#SBATCH --time=24:00:00

# For more memory
#SBATCH --mem=64G

# For multiple GPUs
#SBATCH --gres=gpu:2

# For specific partition
#SBATCH --partition=gpu

# For specific account
#SBATCH --account=your_account
```

#### 3. Python Environment
The scripts automatically detect Python interpreters in this order:
- `python3`
- `python`
- `python3.8`
- `python3.9` 
- `python3.10`

If your cluster uses a different Python command, modify the detection loop in the scripts.

### Job Management Commands

#### Submit Jobs:
```bash
# Submit with custom resources
sbatch -c 8 --gres=gpu:2 --mem=64G ./run_train_energynet.sh

# Submit with time limit
sbatch -c 4 --gres=gpu:1 --time=12:00:00 ./run_train_energynet.sh
```

#### Monitor Jobs:
```bash
# Check job queue
squeue -u $USER

# Check specific job
squeue -j JOB_ID

# Check job details
scontrol show job JOB_ID

# Check job efficiency (after completion)
seff JOB_ID

# Check job accounting
sacct -j JOB_ID --format=JobID,JobName,MaxRSS,Elapsed,CPUTime
```

#### Cancel Jobs:
```bash
# Cancel specific job
scancel JOB_ID

# Cancel all your jobs
scancel -u $USER
```

### Output Files

#### SLURM Output Files:
- `slurm_test_JOBID.out` - Standard output for test jobs
- `slurm_test_JOBID.err` - Error output for test jobs
- `slurm_train_JOBID.out` - Standard output for training jobs
- `slurm_train_JOBID.err` - Error output for training jobs

#### Results Files:
- `models/` - Test environment models
- `plots/` - Training plots and analysis
- `runs/` - TensorBoard logs for tests
- `energynet_experiments/` - All EnergyNet training results

### Cluster-Specific Features

#### GPU Detection:
Scripts automatically use SLURM-allocated GPUs via `CUDA_VISIBLE_DEVICES`

#### CPU Threading:
Scripts set `OMP_NUM_THREADS` to match `SLURM_CPUS_PER_TASK`

#### Memory Management:
Scripts display allocated memory from `SLURM_MEM_PER_NODE`

#### Job Information:
Scripts display job ID, node, and resource allocation at startup

### Troubleshooting SLURM Issues

#### Common Problems:

1. **Job Stuck in Queue**
   ```bash
   # Check queue status
   squeue -u $USER
   
   # Check available resources
   sinfo -N -l
   
   # Check partition limits
   scontrol show partition
   ```

2. **Out of Memory**
   ```bash
   # Increase memory request
   sbatch --mem=64G ./script.sh
   
   # Or reduce batch size in training
   # Edit script: --batch-size 128
   ```

3. **Time Limit Exceeded**
   ```bash
   # Increase time limit
   sbatch --time=24:00:00 ./script.sh
   
   # Or use quick test for debugging
   sbatch ./run_train_energynet.sh --quick-test
   ```

4. **GPU Not Available**
   ```bash
   # Check GPU availability
   sinfo -o "%20N %10c %10m %25f %10G" | grep gpu
   
   # Request specific GPU type
   sbatch --gres=gpu:v100:1 ./script.sh
   ```

5. **Module Loading Issues**
   ```bash
   # List available modules
   module avail
   
   # Check loaded modules
   module list
   
   # Load required modules manually
   module load python/3.8 cuda/11.8
   ```

### Performance Optimization

#### For Faster Training:
```bash
# Use more CPUs
sbatch -c 8 --gres=gpu:1 ./run_train_energynet.sh

# Use multiple GPUs (if code supports it)
sbatch --gres=gpu:2 ./run_train_energynet.sh

# Use high-memory nodes
sbatch --mem=64G ./run_train_energynet.sh
```

#### For Debugging:
```bash
# Quick test runs
sbatch ./run_train_energynet.sh --quick-test

# Interactive session
srun -c 4 --gres=gpu:1 --pty bash
# Then run scripts interactively
```

### File Transfer

#### Download Results:
```bash
# Download all results
scp -r username@cluster:/path/to/energynet_experiments ./

# Download specific files
scp username@cluster:/path/to/energynet_experiments/exp_results.json ./

# Download logs
scp username@cluster:/path/to/slurm_train_*.out ./
```

#### Upload Code Changes:
```bash
# Upload modified scripts
scp ./run_train_energynet.sh username@cluster:/path/to/scripts/

# Upload entire directory
scp -r ./mo_sac_testing/ username@cluster:/path/to/project/
```

### Example Job Submission Scripts

#### High-Priority Training:
```bash
#!/bin/bash
#SBATCH --job-name=mo_sac_priority
#SBATCH --partition=gpu
#SBATCH --qos=high
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=128G

module load python/3.8 cuda/11.8 pytorch/1.12.0
./run_train_energynet.sh --total-timesteps 5000000
```

#### Hyperparameter Sweep:
```bash
#!/bin/bash
for weight1 in 0.3 0.5 0.7 0.9; do
    weight2=$(echo "1.0 - $weight1" | bc -l)
    sbatch --job-name="sweep_${weight1}" \
           -c 4 --gres=gpu:1 \
           --export=WEIGHTS="$weight1 $weight2" \
           ./run_train_energynet.sh
done
```

### Best Practices

1. **Start Small**: Always test with `--quick-test` first
2. **Monitor Resources**: Check `seff JOB_ID` after jobs complete
3. **Save Intermediate Results**: Scripts auto-save checkpoints
4. **Use Screen/Tmux**: For monitoring long-running jobs
5. **Check Disk Space**: Ensure adequate space for results
6. **Backup Results**: Download important results regularly
