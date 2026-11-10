import numpy as np

from controllers.interface import Controller
from controllers.pd import PDController, angle_normalize


class EnergyController(Controller):
    """Swing-up wg Astrom, Furuta (2000): u = k * (E_ref - E) * sign(theta_dot),
    blisko góry przełączamy się na PD."""

    def __init__(self, env, k=1.0, switch_angle=0.5):
        super().__init__(env)
        p = env.unwrapped
        self.m, self.l, self.g = p.m, p.l, p.g
        self.J = self.m * self.l ** 2 / 3  # pręt obracany wokół końca
        self.k = k
        self.switch_angle = switch_angle
        self.pd = PDController(env)
        self.mode = "energy"

    def energy(self, theta, theta_dot):
        # E = 0 dla wahadła w górze w spoczynku
        return 0.5 * self.J * theta_dot ** 2 + self.m * self.g * self.l / 2 * (np.cos(theta) - 1)

    def compute_control(self):
        theta, theta_dot = self.env.unwrapped.state
        if abs(angle_normalize(theta)) < self.switch_angle:
            self.mode = "pd"
            return self.pd.compute_control()
        self.mode = "energy"
        direction = 1.0 if theta_dot >= 0 else -1.0
        u = self.k * (0.0 - self.energy(theta, theta_dot)) * direction
        return np.clip([u], -2.0, 2.0).astype(np.float32)
