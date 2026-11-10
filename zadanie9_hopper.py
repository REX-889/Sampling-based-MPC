# Zadanie 9 - to samo MPPI na Hopperze (i dowolnym innym środowisku)
import os
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from controllers.mppi import MPPIController

# wizualizacja domyślnie włączona, RENDER=0 ją wyłącza (np. w run_all.sh)
RENDER = os.environ.get("RENDER", "1") != "0"
ENV_ID = sys.argv[1] if len(sys.argv) > 1 else "Hopper-v5"
os.makedirs("results", exist_ok=True)

env = gym.make(ENV_ID, render_mode="human" if RENDER else None)
env.reset(seed=0)
if ENV_ID == "Pendulum-v1":
    env.unwrapped.state = np.array([np.pi, 0.0])
    controller = MPPIController(env, n_samples=128, horizon=30, sigma=0.5, n_iterations=3)
else:
    # Hopper: bez kary za upadek nagroda za prędkość wygrywa i robot się wywraca,
    # krótki horyzont (T<=100) też widzi upadek za późno
    # N=256: bez upadków na ziarnach 0-3 (średnio ok. 660 pkt); przy N=192 średnio ok. 790,
    # ale na ziarnie 0 Hopper przewraca się w kroku 469
    controller = MPPIController(env, n_samples=256, horizon=150, sigma=0.3, fall_penalty=10.0)

rewards = []
for i in range(500):
    _, reward, terminated, truncated, _ = env.step(controller.compute_control())
    rewards.append(reward)
    if i % 50 == 0:
        print(f"krok {i}, reward do tej pory {sum(rewards):.0f}")
    if terminated or truncated:
        print(f"koniec epizodu w kroku {i}")
        break
print(f"{ENV_ID}: reward = {sum(rewards):.1f}")

plt.plot(np.cumsum(rewards))
plt.xlabel("krok")
plt.ylabel("skumulowana nagroda")
plt.title(f"MPPI - {ENV_ID}")
plt.savefig(f"results/zadanie9_{ENV_ID.split('-')[0].lower()}.png")
env.close()
