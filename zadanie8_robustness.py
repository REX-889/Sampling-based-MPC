# Zadanie 8 - odporność MPPI na błędy modelu
# model w MPPI ma zawsze m=1, l=1, a zmieniamy parametry "prawdziwego" wahadła
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from controllers.mppi import MPPIController

os.makedirs("results", exist_ok=True)

MASSES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
LENGTHS = [0.5, 0.75, 1.0, 1.25, 1.5]
NOISES = [0.0, 0.25, 0.5, 1.0, 1.5]


def run(m=1.0, l=1.0, noise=0.0, model_knows=False):
    env = gym.make("Pendulum-v1", g=9.81)
    env.reset(seed=0)
    env.unwrapped.m = m
    env.unwrapped.l = l
    model = gym.make("Pendulum-v1", g=9.81).unwrapped
    model.reset(seed=0)
    if model_knows:
        # dla porównania: model zna prawdziwe parametry
        model.m = m
        model.l = l
    controller = MPPIController(env, model, n_samples=128, horizon=30, sigma=0.5, lam=0.1)
    rng = np.random.default_rng(1)
    env.unwrapped.state = np.array([np.pi, 0.0])
    total = 0.
    for i in range(200):
        action = controller.compute_control() + rng.normal(0, noise, size=1).astype(np.float32)
        total += env.step(action)[1]
    return total


fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

r_wrong = [run(m=m) for m in MASSES]
r_true = [run(m=m, model_knows=True) for m in MASSES]
print("masa:", MASSES, "\n  model m=1:", np.round(r_wrong), "\n  model zna m:", np.round(r_true))
ax[0].plot(MASSES, r_wrong, "o-", label="model m=1")
ax[0].plot(MASSES, r_true, "s--", label="model zna prawdziwe m")
ax[0].set_xlabel("prawdziwa masa [kg]")

r_wrong = [run(l=l) for l in LENGTHS]
r_true = [run(l=l, model_knows=True) for l in LENGTHS]
print("długość:", LENGTHS, "\n  model l=1:", np.round(r_wrong), "\n  model zna l:", np.round(r_true))
ax[1].plot(LENGTHS, r_wrong, "o-", label="model l=1")
ax[1].plot(LENGTHS, r_true, "s--", label="model zna prawdziwe l")
ax[1].set_xlabel("prawdziwa długość [m]")

r_noise = [run(noise=s) for s in NOISES]
print("szum na wejściu:", NOISES, "\n ", np.round(r_noise))
ax[2].plot(NOISES, r_noise, "o-", label="szum na sterowaniu")
ax[2].set_xlabel("std szumu [Nm]")

for a in ax:
    a.set_ylabel("nagroda epizodu")
    a.grid()
    a.legend()
plt.tight_layout()
plt.savefig("results/zadanie8_robustness.png")
