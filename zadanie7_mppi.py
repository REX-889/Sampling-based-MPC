# Zadanie 7 - MPPI na wahadle
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from controllers.mppi import MPPIController
from controllers.pd import angle_normalize

# wizualizacja domyślnie włączona, RENDER=0 ją wyłącza (np. w run_all.sh)
RENDER = os.environ.get("RENDER", "1") != "0"
os.makedirs("results", exist_ok=True)

env = gym.make("Pendulum-v1", render_mode="human" if RENDER else None, g=9.81)
model = gym.make("Pendulum-v1", g=9.81).unwrapped
model.reset(seed=0)
# 3 iteracje na krok (pętla po k z Algorytmu 1) dają ok. 10 pkt lepiej niż 1
controller = MPPIController(env, model, n_samples=128, horizon=30, sigma=0.5, lam=1.0, n_iterations=3)
env.reset(seed=0)
env.unwrapped.state = np.array([np.pi, 0.0])

thetas, actions = [], []
episode_reward = 0.
for i in range(200):
    action = controller.compute_control()
    _, reward, _, _, _ = env.step(action)
    episode_reward += reward
    thetas.append(angle_normalize(env.unwrapped.state[0]))
    actions.append(action[0])
print(f"MPPI: reward = {episode_reward:.1f}, theta na końcu = {thetas[-1]:.4f}")

t = np.arange(200) * 0.05
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(t, thetas)
ax[0].set_ylabel("theta [rad]")
ax[1].plot(t, actions)
ax[1].set_ylabel("u [Nm]")
ax[1].set_xlabel("t [s]")
fig.suptitle(f"MPPI, reward = {episode_reward:.0f}")
plt.savefig("results/zadanie7_mppi.png")
env.close()
