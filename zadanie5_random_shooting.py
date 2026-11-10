# Zadanie 5 - random shooting
import os
import gymnasium as gym
import numpy as np

from controllers.random_shooting import RandomShootingController

# wizualizacja domyślnie włączona, RENDER=0 ją wyłącza (np. w run_all.sh)
RENDER = os.environ.get("RENDER", "1") != "0"
N_ROLLOUTS = 200
HORIZON = 20

env = gym.make("Pendulum-v1", render_mode="human" if RENDER else None, g=9.81)
controller = RandomShootingController(env, N_ROLLOUTS, HORIZON)
env.reset(seed=0)
env.unwrapped.state = np.array([np.pi, 0.0])

episode_reward = 0.
for i in range(200):
    _, reward, _, _, _ = env.step(controller.compute_control())
    episode_reward += reward
print(f"n={N_ROLLOUTS}, T={HORIZON}: reward = {episode_reward:.1f}")
env.close()
