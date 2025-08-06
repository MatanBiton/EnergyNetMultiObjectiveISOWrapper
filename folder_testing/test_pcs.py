
import sys
import os
sys.path.append('..')

from energy_net.env.pcs_unit_v0 import PCSUnitEnv
from multi_objecctive_iso_algo.multi_objective_sac import MultiObjectiveSAC
path = "/home/matan.biton/EnergyNetMultiObjectiveISOWrapper/multi_objecctive_iso_algo/mo_sac_testing/models/train_energynet_full_opt_0_action_with_disp_1754339131_final.pth"
class testC:
    def predict(obs, deterministic):
        return [(0,0)]
agent = MultiObjectiveSAC(3, 3, 2)
agent.load(path)
env = PCSUnitEnv(trained_iso_model_instance=agent)
env.reset()
for i in range(10):
    action = env.action_space.sample()

    print(env.step(action))