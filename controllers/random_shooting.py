import gymnasium as gym
import numpy as np

from controllers.interface import Controller


class RandomShootingController(Controller):
    def __init__(self, env, n_rollouts=100, horizon=30, seed=0):
        super().__init__(env)
        # model to druga kopia środowiska, na której symulujemy rollouty
        self.model = gym.make("Pendulum-v1", g=9.81).unwrapped
        self.model.reset(seed=seed)
        self.n_rollouts = n_rollouts
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)

    def compute_control(self):
        state = self.env.unwrapped.state.copy()
        best_reward = -np.inf
        best_action = None
        for _ in range(self.n_rollouts):
            actions = self.rng.uniform(-2.0, 2.0, size=(self.horizon, 1))
            self.model.state = state.copy()
            total = 0.0
            for a in actions:
                _, r, _, _, _ = self.model.step(a)
                total += r
            if total > best_reward:
                best_reward = total
                best_action = actions[0]
        return best_action.astype(np.float32)
