from .generic_pcs_agent import GenericPCSAgent
import torch

class ConstantPCSAgent(GenericPCSAgent):
    def __init__(self, action):
        super().__init__()
        self.action = action

    def predict(self, obs, deterministic=True, **kwargs):
        return torch.tensor(self.action), None