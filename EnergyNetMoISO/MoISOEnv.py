from gymnasium.spaces import Box
from energy_net.env.iso_env import make_iso_env_zoo
from energy_net.env.iso_v0 import ISOEnv
from energy_net.market.pricing_policy import PricingPolicy
from energy_net.market.iso.demand_patterns import DemandPattern
from energy_net.market.iso.cost_types import CostType
import numpy as np


class MultiObjectiveISOEnv(ISOEnv):
    def __init__(self, use_dispatch_action=False,
                 dispatch_strategy="PORPRTIONAL",
                 trained_pcs_model=None,
                 **kwargs):
        env_kwargs = {
            "dispatch_config": {
                "use_dispatch_action": use_dispatch_action,
                "default_strategy": dispatch_strategy
            }
        }
        kwargs.update(env_kwargs)

        super().__init__(pricing_policy=PricingPolicy.ONLINE, 
                         cost_type=CostType.CONSTANT,
                         num_pcs_agents=1,
                         demand_pattern=DemandPattern.CONSTANT,
                         trained_pcs_model=trained_pcs_model, 
                         **kwargs)
        self.reward_space = Box(low=np.array([-np.inf, -np.inf], dtype=np.float32),
                                high=np.array([np.inf, np.inf], dtype=np.float32),
                                dtype=np.float32)
        self.reward_dim = 2

    def _calculate_cost_reward(self, info):
        return (-1)*(info["dispatch_cost"]+info["reserve_cost"])
    
    def _calculate_stability_reward(self, info):
        return (-1) * info["shortfall"]

    def step(self, action):
        state, _, terminated, truncated, info = super().step(action)
        cost_reward = self._calculate_cost_reward(info)
        stability_reward = self._calculate_stability_reward(info)
        return state, np.array([cost_reward, stability_reward]), terminated, truncated, info
        



