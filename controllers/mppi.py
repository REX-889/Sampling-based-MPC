import os

import gymnasium as gym
import mujoco
import numpy as np
from mujoco import rollout

from controllers.interface import Controller


def get_state(env):
    env = env.unwrapped
    if hasattr(env, "data"):  # MuJoCo (np. Hopper)
        return env.data.qpos.copy(), env.data.qvel.copy()
    return env.state.copy()  # Pendulum


def set_state(env, state):
    env = env.unwrapped
    if hasattr(env, "data"):
        env.set_state(state[0], state[1])
    else:
        env.state = state.copy()


class MPPIController(Controller):
    def __init__(self, env, model=None, n_samples=64, horizon=30, sigma=1.0, lam=1.0,
                 fall_penalty=0.0, n_iterations=1, seed=0, parallel=True):
        super().__init__(env)
        if model is None:
            model = gym.make(env.spec.id).unwrapped
            model.reset(seed=seed)
        self.model = model
        self.n_samples = n_samples
        self.horizon = horizon
        self.lam = lam
        self.fall_penalty = fall_penalty
        self.n_iterations = n_iterations
        self.u_min = env.action_space.low.astype(np.float64)
        self.u_max = env.action_space.high.astype(np.float64)
        # sigma względem połowy zakresu akcji (Pendulum: 2 Nm, Hopper: 1)
        self.sigma = sigma * (self.u_max - self.u_min) / 2
        self.rng = np.random.default_rng(seed)
        self.u = np.zeros((horizon, len(self.u_min)))
        # Hopper: wszystkie próbki naraz przez mujoco.rollout na wielu wątkach;
        # Pendulum (i inne) liczy się po staremu, przez env.step
        self.parallel = parallel and env.spec.id.startswith("Hopper")
        if self.parallel:
            self._init_parallel()

    def _init_parallel(self):
        m = self.model
        self.mj_model = m.model
        self.frame_skip = m.frame_skip
        self.dt = m.dt
        # tyle wątków, ile rdzeni faktycznie dostępnych dla procesu (uwzględnia limity
        # Dockera/taskset); na 1 rdzeniu też działa, tylko bez zysku z równoległości
        try:
            n_threads = len(os.sched_getaffinity(0))
        except AttributeError:  # macOS/Windows nie mają sched_getaffinity
            n_threads = os.cpu_count() or 1
        n_threads = max(1, int(os.environ.get("MPPI_THREADS", n_threads)))
        # długość odcinka horyzontu (kroki env) między odrzucaniem upadłych próbek
        self.chunk = 10
        self.thread_data = [mujoco.MjData(self.mj_model) for _ in range(n_threads)]
        self.state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        # stan z rollout zaczyna się od czasu (1 liczba), potem qpos i qvel
        self.qpos_idx = 1 + np.arange(self.mj_model.nq)
        self.qvel_idx = 1 + self.mj_model.nq + np.arange(self.mj_model.nv)

    def _hopper_rewards(self, x_before, qpos, qvel, actions):
        """Nagroda i 'terminated' jak w Hopper-v5 step(), dla tablic (próbki x horyzont)."""
        m = self.model
        x_velocity = (qpos[..., 0] - x_before) / self.dt
        z, angle = qpos[..., 1], qpos[..., 2]
        rest = np.concatenate([qpos[..., 2:], qvel], axis=-1)
        min_state, max_state = m._healthy_state_range
        min_z, max_z = m._healthy_z_range
        min_angle, max_angle = m._healthy_angle_range
        healthy = (np.all((min_state < rest) & (rest < max_state), axis=-1)
                   & (min_z < z) & (z < max_z) & (min_angle < angle) & (angle < max_angle))
        reward = (m._forward_reward_weight * x_velocity
                  + m._healthy_reward * healthy
                  - m._ctrl_cost_weight * np.sum(actions ** 2, axis=-1))
        return reward, ~healthy & m._terminate_when_unhealthy

    def parallel_costs(self, u_samples):
        n, horizon = u_samples.shape[:2]
        data = self.env.unwrapped.data
        state0 = np.empty(mujoco.mj_stateSize(self.mj_model, self.state_spec))
        mujoco.mj_getState(self.mj_model, data, state0, self.state_spec)
        states = np.tile(state0, (n, 1))
        x_before = np.full(n, data.qpos[0])
        costs = np.zeros(n)
        alive = np.arange(n)  # próbki, które jeszcze stoją
        # horyzont w odcinkach: po każdym odrzucam próbki, które upadły,
        # tak jak stara pętla przerywa rollout po upadku (inaczej liczylibyśmy szamotanie)
        for t0 in range(0, horizon, self.chunk):
            if alive.size == 0:
                break
            t1 = min(t0 + self.chunk, horizon)
            # każda akcja trzymana przez frame_skip kroków fizyki, jak w do_simulation
            control = np.repeat(u_samples[alive, t0:t1], self.frame_skip, axis=1)
            out, _ = rollout.rollout(self.mj_model, self.thread_data, states[alive], control,
                                     persistent_pool=True)
            # stan po każdym pełnym kroku env (co frame_skip kroków fizyki)
            out = out[:, self.frame_skip - 1::self.frame_skip]
            qpos, qvel = out[..., self.qpos_idx], out[..., self.qvel_idx]
            xb = np.concatenate([x_before[alive, None], qpos[:, :-1, 0]], axis=1)
            reward, terminated = self._hopper_rewards(xb, qpos, qvel, u_samples[alive, t0:t1])
            # nagrody do pierwszego upadku włącznie, potem kara za niedożyte kroki
            fell = terminated.any(axis=1)
            t_fall = np.where(fell, terminated.argmax(axis=1), t1 - t0)
            counted = np.arange(t1 - t0)[None] <= t_fall[:, None]
            costs[alive] -= np.sum(reward * counted, axis=1)
            costs[alive[fell]] += self.fall_penalty * (horizon - (t0 + t_fall[fell]))
            states[alive] = out[:, -1]
            x_before[alive] = qpos[:, -1, 0]
            alive = alive[~fell]
        return costs

    def rollout_cost(self, state, actions):
        set_state(self.model, state)
        cost = 0.0
        for t, a in enumerate(actions):
            _, r, terminated, _, _ = self.model.step(a)
            cost -= r
            if terminated:
                # kara za każdy krok, którego rollout nie dożył (Hopper się przewrócił)
                cost += self.fall_penalty * (len(actions) - t)
                break
        return cost

    def compute_control(self):
        state = get_state(self.env)
        for k in range(self.n_iterations):
            noise = self.rng.normal(0.0, self.sigma, size=(self.n_samples, *self.u.shape))
            u_samples = np.clip(self.u + noise, self.u_min, self.u_max)
            du = u_samples - self.u  # efektywne zaburzenia po obcięciu
            if self.parallel:
                costs = self.parallel_costs(u_samples)
            else:
                costs = np.array([self.rollout_cost(state, u_n) for u_n in u_samples])
            # odejmuję minimum, żeby exp nie wyzerował wszystkich wag
            w = np.exp(-(costs - costs.min()) / self.lam)
            w /= w.sum()
            self.u = self.u + np.tensordot(w, du, axes=1)
        action = self.u[0].copy()
        # przesunięcie sekwencji o jedną próbkę, na końcu zero
        self.u = np.vstack([self.u[1:], np.zeros((1, self.u.shape[1]))])
        return action.astype(np.float32)
