# Zadanie 2 - ustawienie stanu początkowego
import os
import gymnasium as gym
import numpy as np

# wizualizacja domyślnie włączona, RENDER=0 ją wyłącza (np. w run_all.sh)
RENDER = os.environ.get("RENDER", "1") != "0"

env = gym.make("Pendulum-v1", render_mode="human" if RENDER else None, g=9.81)
env.reset()
# reset() losuje stan, więc nadpisujemy go ręcznie
env.unwrapped.state = np.array([3.14, 0.0])
print("stan:", env.unwrapped.state)

# bez sterowania wahadło powinno zostać na dole
for i in range(50):
    env.step(np.zeros(1))
print("stan po 50 krokach bez sterowania:", env.unwrapped.state)
env.close()
