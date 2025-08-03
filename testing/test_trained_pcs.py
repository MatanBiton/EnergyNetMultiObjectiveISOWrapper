from EnergyNetMoISO.MoISOEnv import MultiObjectiveISOEnv
from EnergyNetMoISO.pcs_models.generic_pcs_agent import GenericPCSAgent
from EnergyNetMoISO.pcs_models.ppo_pcs_agent import PPOPCSAgent
import numpy as np

# class TestAgent(GenericPCSAgent):

#     def predict(obs, deterministice, **kwargs):
#         return np.array([0])

# agent = TestAgent()

# env = MultiObjectiveISOEnv(trained_pcs_model=agent)
# env.step(np.array([0,0]))

ppo_path = "C:\\Technion\\EnergySystems\\EnergyNetMultiObjectiveISOWrapper\\best_model.zip"

env = MultiObjectiveISOEnv()
print(env.step(np.array([0,0]))[1])