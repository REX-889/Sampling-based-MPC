from controllers.interface import Controller


class RandomController(Controller):
    def compute_control(self):
        return self.env.action_space.sample()