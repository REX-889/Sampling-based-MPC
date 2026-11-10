# Zadanie 1 - zapoznanie się ze środowiskiem Pendulum
import os
import gymnasium as gym

from controllers.random import RandomController

# wizualizacja domyślnie włączona, RENDER=0 ją wyłącza (np. w run_all.sh)
RENDER = os.environ.get("RENDER", "1") != "0"

env = gym.make("Pendulum-v1", render_mode="human" if RENDER else None, g=9.81)
p = env.unwrapped
print("obserwacja:", env.observation_space)  # [cos(theta), sin(theta), theta_dot]
print("akcja:", env.action_space)  # moment [Nm]
print(f"m={p.m}, l={p.l}, g={p.g}, dt={p.dt}, max_torque={p.max_torque}, max_speed={p.max_speed}")
# nagroda: -(theta^2 + 0.1*theta_dot^2 + 0.001*u^2), najlepiej 0 w górze

controller = RandomController(env)
env.reset(seed=0)
print("stan po reset():", p.state)  # [theta, theta_dot], theta=0 to góra

episode_reward = 0.
for i in range(200):
    action = controller.compute_control()
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
print(f"Episode reward (losowe sterowanie): {episode_reward:.1f}")
env.close()
