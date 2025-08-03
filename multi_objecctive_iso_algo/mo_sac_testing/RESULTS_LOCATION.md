# EnergyNet Training Results Location Guide

## ✅ Your Results Are Correctly Saved!

After running the training script, your results are saved in these locations:

### 📁 **Main Directory**
- `energynet_training_config.json` - Training configuration
- `energynet_training_results.json` - Training metrics and final performance

### 📁 **models/** Directory  
- `energynet_training_final.pth` - Final trained model
- `energynet_training_[timesteps].pth` - Intermediate checkpoints

### 📁 **logs/** Directory
- `energynet_training_[timestamp]/` - TensorBoard logs for visualization

### 📁 **plots/** Directory
- Training plots and visualizations (if generated)

## 🔍 **How to View Results**

### Quick Summary
```bash
./show_latest_results.sh
```

### Detailed Results
```bash
./view_results.sh
```

### TensorBoard Visualization  
```bash
./view_results.sh --tensorboard
# or manually:
tensorboard --logdir=logs --port=6006 --host=0.0.0.0
```

### Manual File Inspection
```bash
# View latest results
cat energynet_training_results.json | python3 -m json.tool

# List all models  
ls -la models/*.pth

# Check TensorBoard logs
ls -la logs/energynet_training_*/
```

## 🚀 **Running New Training**

```bash
# Default training (100k timesteps)
sbatch --gres=gpu:1 -c 4 run_train_energynet_v2.sh

# Custom training (e.g., 50k timesteps, learning rate 0.001, batch size 128)
sbatch --gres=gpu:1 -c 4 run_train_energynet_v2.sh 50000 0.001 128
```

## 📊 **Understanding Results**

The `energynet_training_results.json` contains:
- **final_evaluation**: Performance metrics on 50 test episodes
- **training_stats**: Number of episodes and final episode reward  
- **config**: All training parameters used
- **model_path**: Location of the saved model

## ✅ **All Files Are There!**

If you don't see results, check:
1. SLURM job completed successfully: `squeue -u $USER`
2. No errors in: `cat slurm_energynet_[job_id].err`
3. Results copied back: Look for "copied back" messages in `slurm_energynet_[job_id].out`
