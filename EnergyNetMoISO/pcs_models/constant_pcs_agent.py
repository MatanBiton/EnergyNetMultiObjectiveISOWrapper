from generic_pcs_agent import GenericPCSAgent

class ConstantPCSAgent(GenericPCSAgent):
    def __init__(self, action):
        super().__init__()
        self.action = action

    def predict(self, obs, deterministice, **kwargs):
        return self.action