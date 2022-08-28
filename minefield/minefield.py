import gym
from gym import spaces
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class Vehicle:
    x_pos: float = 0.0
    y_pos: float = 0.0
    direction: float = 0.0
    step: int = 0


@dataclass
class Target:
    x_pos: float = 0.0
    y_pos: float = 0.0


@dataclass
class Mine:
    x_pos: float = 0.0
    y_pos: float = 0.0


class MinefieldEnv(gym.Env):
    metadata = {}

    def __init__(
        self,
        max_mines: int = 10,
        max_steps: int = 200,
        width: float = 200.0,
    ):
        self.vehicle = Vehicle()
        self.target = Target()
        self.mines = []

        self.max_mines = max_mines
        self.max_steps = max_steps
        self.width = width

        self.observation_space = spaces.Dict(
            {
                "Vehicle": spaces.Box(
                    -float("inf"), float("inf"), shape=(4,), dtype=np.float32
                ),
                "Target": spaces.Box(
                    -float("inf"), float("inf"), shape=(2,), dtype=np.float32
                ),
                "Mines": spaces.Box(
                    -float("inf"), float("inf"), shape=(2,), dtype=np.float32
                ),
            }
        )

        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        def randpos() -> Tuple[float, float]:
            return (
                self.np_random.uniform(-self.width / 2, self.width / 2),
                self.np_random.uniform(-self.width / 2, self.width / 2),
            )

        self.vehicle.x_pos, self.vehicle.y_pos = randpos()
        self.target.x_pos, self.target.y_pos = randpos()
        mines: List[Mine] = []
        for _ in range(self.max_mines):
            x, y = randpos()
            # Check that the mine is not too close to the vehicle, target, or any other mine
            pos = [(m.x_pos, m.y_pos) for m in mines] + [
                (self.vehicle.x_pos, self.vehicle.y_pos),
                (self.target.x_pos, self.target.y_pos),
            ]
            if any(map(lambda p: (x - p[0]) ** 2 + (y - p[1]) ** 2 < 15 * 15, pos)):
                continue
            mines.append(Mine(x, y))
        self.vehicle.direction = self.np_random.uniform(0, 2 * np.pi)
        self._step = 0
        self.mines = mines
        obs, _, _ = self._get_obs_rew_done()
        return obs, {}

    def step(self, action):
        if action == 0:
            self.vehicle.direction -= np.pi / 8
        elif action == 1:
            self.vehicle.x_pos += 3 * np.cos(self.vehicle.direction)
            self.vehicle.y_pos += 3 * np.sin(self.vehicle.direction)
        elif action == 2:
            self.vehicle.direction += np.pi / 8
        else:
            raise ValueError(
                f"Invalid action {action}, must be 0, 1, or 2. (0: left, 1: forward, 2: right)"
            )
        self.vehicle.direction %= 2 * np.pi

        self._step += 1
        self.vehicle.step = self._step

        obs, reward, done = self._get_obs_rew_done()
        return obs, reward, done, {}

    def _get_obs_rew_done(self):
        if (self.target.x_pos - self.vehicle.x_pos) ** 2 + (
            self.target.y_pos - self.vehicle.y_pos
        ) ** 2 < 5 * 5:
            done = True
            reward = 1
        elif (
            any(
                map(
                    lambda m: (self.vehicle.x_pos - m.x_pos) ** 2
                    + (self.vehicle.y_pos - m.y_pos) ** 2
                    < 5 * 5,
                    self.mines,
                )
            )
            or self._step >= self.max_steps
        ):
            done = True
            reward = 0
        else:
            done = False
            reward = 0
        return (
            {
                "Vehicle": np.array(
                    [
                        self.vehicle.x_pos,
                        self.vehicle.y_pos,
                        self.vehicle.direction,
                        self._step,
                    ],
                    dtype=np.float32,
                ).reshape(1, 4),
                "Target": np.array(
                    [self.target.x_pos, self.target.y_pos],
                    dtype=np.float32,
                ).reshape(1, 2),
                "Mines": np.array(
                    [[m.x_pos, m.y_pos] for m in self.mines],
                    dtype=np.float32,
                ).reshape(len(self.mines), 2),
            },
            reward,
            done,
        )
