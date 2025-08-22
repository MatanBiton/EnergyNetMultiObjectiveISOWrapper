import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from EnergyNetMoISOWrapper.MoISOWrapper import MoISOEnv



env = MoISOEnv(use_dispatch_action=True)
print(env.action_space.sample())
print(env.step(env.action_space.sample()))
# import os
# import energy_net

# installed_dir = os.path.dirname(energy_net.__file__)
# print("Package installed at:", installed_dir)

# print("Configs folder contents:")
# print(os.listdir(os.path.join(installed_dir, "configs")))
