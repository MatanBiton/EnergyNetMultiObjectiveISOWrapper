#!/bin/bash
# Utility script to view EnergyNet training results and start tensorboard.

echo "EnergyNet Training Results Viewer"
echo "================================="

# Check for results files
if ls *_results.json 1> /dev/null 2>&1; then
    echo "Available results files:"
    for file in *_results.json; do
        if [ -f "$file" ]; then
            echo "  - $file"
            # Extract key metrics
            if command -v jq > /dev/null 2>&1; then
                experiment=$(jq -r '.experiment_name' "$file" 2>/dev/null || echo "unknown")
                training_time=$(jq -r '.training_time' "$file" 2>/dev/null || echo "unknown")
                timesteps=$(jq -r '.total_timesteps' "$file" 2>/dev/null || echo "unknown")
                final_performance=$(jq -r '.final_evaluation.scalarized_mean' "$file" 2>/dev/null || echo "unknown")
                echo "    Experiment: $experiment"
                echo "    Timesteps: $timesteps"
                echo "    Training time: ${training_time}s"
                echo "    Final performance: $final_performance"
                echo ""
            fi
        fi
    done
else
    echo "No results files found in current directory."
fi

# Check for model files
echo "Available trained models:"
if [ -d "models" ]; then
    find models -name "*.pth" -type f | while read model; do
        echo "  - $model"
    done
else
    echo "  No models directory found."
fi

# Check for tensorboard logs
echo ""
echo "Tensorboard logs:"
tb_dirs=""
for dir in logs/*/; do
    if [ -d "$dir" ]; then
        echo "  - $dir"
        tb_dirs="$tb_dirs $dir"
    fi
done

for dir in models/logs/*/; do
    if [ -d "$dir" ]; then
        echo "  - $dir"
        tb_dirs="$tb_dirs $dir"
    fi
done

for dir in runs/*/; do
    if [ -d "$dir" ]; then
        echo "  - $dir" 
        tb_dirs="$tb_dirs $dir"
    fi
done

if [ -n "$tb_dirs" ]; then
    echo ""
    echo "To view tensorboard logs, run:"
    echo "  tensorboard --logdir=logs --port=6006 --host=0.0.0.0"
    echo "  # or for all logs:"
    echo "  tensorboard --logdir=. --port=6006 --host=0.0.0.0"
    echo ""
    echo "To start tensorboard automatically, run:"
    echo "  $0 --tensorboard"
fi

# Auto-start tensorboard if requested
if [ "$1" = "--tensorboard" ] || [ "$1" = "-tb" ]; then
    echo "Starting tensorboard..."
    if command -v tensorboard > /dev/null 2>&1; then
        echo "Tensorboard will be available at: http://localhost:6006"
        echo "Press Ctrl+C to stop"
        tensorboard --logdir=. --port=6006 --host=0.0.0.0
    else
        echo "Error: tensorboard command not found"
        echo "Install with: pip install tensorboard"
    fi
fi
