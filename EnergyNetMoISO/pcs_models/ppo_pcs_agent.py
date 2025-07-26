from .generic_pcs_agent import GenericPCSAgent
from stable_baselines3 import PPO
from gymnasium.spaces import Box, Discrete
import numpy as np

config = {
    "observation_space": {
        "battery_level": {
            "min": "from_battery_config",  # Will use battery min from battery config
            "max": "from_battery_config",  # Will use battery max from battery config
        },
        "time": {
            "min": 0.0,
            "max": 1.0,
        },
        "iso_buy_price": {
            "min": 0.0,
            "max": 100.0,
        },
        "iso_sell_price": {
            "min": 0.0,
            "max": 100.0,
        },
    }
}


class PPOPCSAgent(GenericPCSAgent):
    def __init__(self, ppo_path):
        super().__init__()
        pcs_obs_config = config
        energy_config = self.pcs_unit_config['battery']['model_parameters']
        
        # Get battery level bounds from battery config if specified
        battery_level_config = pcs_obs_config.get('battery_level', {})
        battery_min = energy_config['min'] if battery_level_config.get('min') == "from_battery_config" else battery_level_config.get('min', energy_config['min'])
        battery_max = energy_config['max'] if battery_level_config.get('max') == "from_battery_config" else battery_level_config.get('max', energy_config['max'])
        
        # Get other observation space bounds from config
        pcs_time_config = pcs_obs_config.get('time', {})
        buy_price_config = pcs_obs_config.get('iso_buy_price', {})
        sell_price_config = pcs_obs_config.get('iso_sell_price', {})
        
        self.pcs_observation_space = Box(
            low=np.array([
                battery_min,
                pcs_time_config.get('min', 0.0),
                buy_price_config.get('min', 0.0),
                sell_price_config.get('min', 0.0)
            ], dtype=np.float32),
            high=np.array([
                battery_max,
                pcs_time_config.get('max', 1.0),
                buy_price_config.get('max', 100.0),
                sell_price_config.get('max', 100.0)
            ], dtype=np.float32),
            dtype=np.float32
        )
        
        # Load the trained PPO model
        self.ppo_model = PPO.load(ppo_path)
        
        # Set the observation space for the loaded model
        self.ppo_model.observation_space = self.pcs_observation_space
    
    def predict(self, obs, deterministic=True, **kwargs):
        """
        Predict action using the trained PPO model.
        
        Args:
            obs: Observation from the environment
            deterministic: Whether to use deterministic policy
            **kwargs: Additional arguments
            
        Returns:
            Action predicted by the PPO model
        """
        # Ensure observation is in the correct format
        obs = np.array(obs, dtype=np.float32)
        
        # Use PPO model to predict action
        action, _ = self.ppo_model.predict(obs, deterministic=deterministic)
        
        return action
