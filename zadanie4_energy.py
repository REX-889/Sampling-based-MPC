# Zadanie 4 - sterowanie energią + PD w okolicy góry
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from controllers.energy import EnergyController
from controllers.pd import angle_normalize

# wizualizacja domyślnie włączona, RENDER=0 ją wyłącza (np. w run_all.sh)
RENDER = os.environ.get("RENDER", "1") != "0"
os.makedirs("results", exist_ok=True)

env = gym.make("Pendulum-v1", render_mode="human" if RENDER else None, g=9.81)
controller = EnergyController(env)
env.reset()
env.unwrapped.state = np.array([3.1, 0.0])

thetas, actions, energies = [], [], []
switch_step = None
episode_reward = 0.
for i in range(200):
    action = controller.compute_control()
    if controller.mode == "pd" and switch_step is None:
        switch_step = i
    _, reward, _, _, _ = env.step(action)
    episode_reward += reward
    theta, theta_dot = env.unwrapped.state
    thetas.append(angle_normalize(theta))
    actions.append(action[0])
    energies.append(controller.energy(theta, theta_dot))
print(f"reward = {episode_reward:.1f}, przełączenie na PD w kroku {switch_step}, theta na końcu = {thetas[-1]:.4f}")

t = np.arange(200) * 0.05
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
ax[0].plot(t, thetas)
ax[0].set_ylabel("theta [rad]")
ax[1].plot(t, actions)
ax[1].set_ylabel("u [Nm]")
ax[2].plot(t, energies)
ax[2].set_ylabel("E [J]")
ax[2].set_xlabel("t [s]")
for a in ax:
    a.axvline(switch_step * 0.05, color="gray", linestyle="--")
fig.suptitle("Swing-up energią + PD")
plt.savefig("results/zadanie4_energy.png")
env.close()
