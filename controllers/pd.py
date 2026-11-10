import numpy as np

from controllers.interface import Controller


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class PDController(Controller):
    def __init__(self, env, kp=12.0, kd=2.5):
        super().__init__(env)
        self.kp = kp
        self.kd = kd

    def compute_control(self):
        theta, theta_dot = self.env.unwrapped.state
        u = -self.kp * angle_normalize(theta) - self.kd * theta_dot
        return np.clip([u], -2.0, 2.0).astype(np.float32)
