from stable_baselines3 import TD3

from env import CarlaEnvContinuous

with CarlaEnvContinuous() as carla_env:
    # Load the trained agent
    model = TD3.load("checkpoints/model.zip", carla_env)

    # If you want to test the learned policy after training:
    obs, _ = carla_env.reset()
    for _ in range(10000):
        try:
            action, _states = model.predict(obs.copy(), deterministic=True)
        except Exception as e:
            print(e)
        obs, rewards, terminated, truncated, info = carla_env.step(action)
        if terminated or truncated:
            obs, _ = carla_env.reset()
