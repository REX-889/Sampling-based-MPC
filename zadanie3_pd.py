# Zadanie 3 - regulator PD
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from controllers.pd import PDController, angle_normalize

# wizualizacja domyślnie włączona, RENDER=0 ją wyłącza (np. w run_all.sh)
RENDER = os.environ.get("RENDER", "1") != "0"
os.makedirs("results", exist_ok=True)

env = gym.make("Pendulum-v1", render_mode="human" if RENDER else None, g=9.81)
controller = PDController(env)

for start in [[0.1, 0.5], [3.1, 0.0]]:
    env.reset()
    env.unwrapped.state = np.array(start)
    thetas = []
    episode_reward = 0.
    for i in range(200):
        _, reward, _, _, _ = env.step(controller.compute_control())
        episode_reward += reward
        thetas.append(angle_normalize(env.unwrapped.state[0]))
    print(f"start {start}: reward = {episode_reward:.1f}, theta na końcu = {thetas[-1]:.3f}")
    plt.plot(np.arange(200) * 0.05, thetas, label=f"start {start}")

# z [3.1, 0] się nie da - max moment 2 Nm < m*g*l/2 = 4.9 Nm, PD nie podniesie wahadła
plt.xlabel("t [s]")
plt.ylabel("theta [rad]")
plt.title("Regulator PD")
plt.legend()
plt.savefig("results/zadanie3_pd.png")
env.close()
