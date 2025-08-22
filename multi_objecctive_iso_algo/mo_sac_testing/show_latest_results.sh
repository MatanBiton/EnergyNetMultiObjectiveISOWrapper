#!/bin/bash
# Quick script to show the latest EnergyNet training results

echo "Latest EnergyNet Training Results Summary"
echo "========================================"

# Find the most recent results file
latest_results=$(ls -t *_results.json 2>/dev/null | head -1)
if [ -n "$latest_results" ]; then
    echo "Latest results file: $latest_results"
    
    if command -v jq > /dev/null 2>&1; then
        echo ""
        echo "Training Summary:"
        echo "=================="
        
        experiment=$(jq -r '.experiment_name' "$latest_results" 2>/dev/null)
        timesteps=$(jq -r '.total_timesteps' "$latest_results" 2>/dev/null)
        training_time=$(jq -r '.training_time' "$latest_results" 2>/dev/null)
        
        echo "Experiment: $experiment"
        echo "Total timesteps: $timesteps"
        echo "Training time: ${training_time}s ($(echo "scale=2; $training_time/60" | bc 2>/dev/null || echo "")min)"
        
        echo ""
        echo "Final Performance:"
        echo "=================="
        
        cost_reward=$(jq -r '.final_evaluation.mean_rewards[0]' "$latest_results" 2>/dev/null)
        stability_reward=$(jq -r '.final_evaluation.mean_rewards[1]' "$latest_results" 2>/dev/null)
        scalarized=$(jq -r '.final_evaluation.scalarized_mean' "$latest_results" 2>/dev/null)
        
        echo "Cost reward: $cost_reward"
        echo "Stability reward: $stability_reward"
        echo "Scalarized reward: $scalarized"
        
        echo ""
        echo "Training Stats:"
        echo "==============="
        
        episodes=$(jq -r '.training_stats.num_episodes' "$latest_results" 2>/dev/null)
        final_episode_reward=$(jq -r '.training_stats.final_episode_reward' "$latest_results" 2>/dev/null)
        
        echo "Total episodes: $episodes"
        echo "Final episode reward: $final_episode_reward"
        
    else
        echo "Install 'jq' for detailed JSON parsing: sudo apt install jq"
        echo "Raw results file content:"
        head -20 "$latest_results"
    fi
else
    echo "No results files found."
fi

echo ""
echo "Available Files:"
echo "================"
echo "Config files: $(ls *_config.json 2>/dev/null | wc -l)"
echo "Results files: $(ls *_results.json 2>/dev/null | wc -l)"
echo "Model files: $(find models -name "*.pth" 2>/dev/null | wc -l)"
echo "Tensorboard logs: $(find logs -name "events.out.tfevents*" 2>/dev/null | wc -l)"

echo ""
echo "To view detailed results: ./view_results.sh"
echo "To start tensorboard: ./view_results.sh --tensorboard"
