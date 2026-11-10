# Zadanie 6 - heatmapa nagrody random shooting dla różnych n i T
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from controllers.random_shooting import RandomShootingController

os.makedirs("results", exist_ok=True)

ROLLOUTS = [5, 10, 25, 50, 100, 200, 500]
HORIZONS = [5, 10, 20, 30, 50]

env = gym.make("Pendulum-v1", g=9.81)
rewards = np.zeros((len(ROLLOUTS), len(HORIZONS)))
for i, n in enumerate(ROLLOUTS):
    for j, T in enumerate(HORIZONS):
        controller = RandomShootingController(env, n, T)
        env.reset(seed=0)
        env.unwrapped.state = np.array([np.pi, 0.0])
        for k in range(200):
            _, reward, _, _, _ = env.step(controller.compute_control())
            rewards[i, j] += reward
        print(f"n={n}, T={T}: {rewards[i, j]:.0f}")
np.save("results/zadanie6_heatmap.npy", rewards)

plt.imshow(rewards, origin="lower", aspect="auto")
plt.colorbar(label="nagroda epizodu")
plt.xticks(range(len(HORIZONS)), HORIZONS)
plt.yticks(range(len(ROLLOUTS)), ROLLOUTS)
for i in range(len(ROLLOUTS)):
    for j in range(len(HORIZONS)):
        plt.text(j, i, f"{rewards[i, j]:.0f}", ha="center", va="center", color="w")
plt.xlabel("długość rolloutu T")
plt.ylabel("liczba rolloutów n")
plt.title("Random shooting, start [pi, 0]")
plt.savefig("results/zadanie6_heatmap.png")
