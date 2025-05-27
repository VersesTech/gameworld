import os
import numpy as np
from gymnasium import spaces

from PIL import Image
from gameworld.envs.base.base_env import GameworldEnv
from gameworld.envs.base.utils import parse_image, make_ball



class Explode(GameworldEnv):
    """ Based on the Atari Kaboom game.
    
        Player moves left right to catch bombs dropping from a spaceship.
    """

    def __init__(self, player_y=170, bomber_y=20, **kwargs):
        super().__init__()

        self.width = 160
        self.height = 210
        self.bucket_width = 30
        self.bucket_height = 12
        self.bomb_size = 5
        self.player_x = self.width // 2 - self.bucket_width // 2
        self.player_y = player_y
        self.player_speed = 8
        self.bomber_x = 10
        self.bomber_y = bomber_y
        self.bomber_dx = 2
        self.max_bombs = 1

        self.reset()
        self.action_space = spaces.Discrete(3)  # 0: stay, 1: left, 2: right
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(210, 160, 3), dtype=np.uint8
        )

    def reset(self):
        self.player_x = self.width // 2 - self.bucket_width // 2
        self.bombs = []

        return self._get_obs(), {}

    def step(self, action):
        if action == 1:
            self.player_x = max(self.player_x - self.player_speed, 0)
        elif action == 2:
            self.player_x = min(
                self.player_x + self.player_speed, self.width - self.bucket_width
            )

        # Bomb movement
        self.bomber_x += self.bomber_dx
        if (
            self.bomber_x <= 10
            or self.bomber_x >= self.width - 10
            or np.random.rand() < 0.02
        ):
            self.bomber_dx = -self.bomber_dx

        if len(self.bombs) < self.max_bombs and np.random.rand() < 0.05:
            self.bombs.append(
                [int(self.bomber_x), int(self.bomber_y), 2]
            )  # (x, y, speed)

        for bomb in self.bombs:
            bomb[1] += bomb[2]

        # Catching bomb
        reward = 0
        done = False
        for bomb in self.bombs:
            if (
                self.player_y - self.bomb_size
                <= bomb[1]
                <= self.player_y + max(self.bucket_height, bomb[2] + self.bomb_size)
                and self.player_x - self.bomb_size
                <= bomb[0]
                <= self.player_x + self.bucket_width
            ):
                reward = 1  # Successful catch
                self.bombs.remove(bomb)

            bomb[2] += 0.5  # Increase speed
        for bomb in self.bombs:
            if bomb[1] >= self.height:
                reward -= 1  # Bomb missed
                self.bombs.remove(bomb)

        return (
            self._get_obs(),
            reward,
            done,
            False,
            {},
        )

    def _get_obs(self):
        obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        obs[:, :, 0] = 50  # Background color (dark blue)
        obs[:, :, 1] = 50
        obs[:, :, 2] = 100
        obs[
            self.player_y : self.player_y + self.bucket_height,
            self.player_x : self.player_x + self.bucket_width,
        ] = [
            255,
            255,
            0,
        ]  # Yellow bucket
        for bomb in self.bombs:
            obs[
                int(bomb[1]) : int(bomb[1]) + self.bomb_size,
                int(bomb[0]) : int(bomb[0]) + self.bomb_size,
            ] = [
                255,
                0,
                0,
            ]  # Red bomb
        obs[
            int(self.bomber_y) : int(self.bomber_y) + 10,
            int(self.bomber_x) - 10 : int(self.bomber_x) + 10,
        ] = [
            0,
            255,
            0,
        ]  # Green bomber

        return obs