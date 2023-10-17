import gymnasium as gym
from stable_baselines3.common.env_checker import check_env

import env

carla_env = env.CarlaEnv()
check_env(carla_env, warn=True)
