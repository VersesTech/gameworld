import numpy as np
from gymnasium import spaces
from gameworld.envs.base.base_env import GameworldEnv


class Jump(GameworldEnv):
    """Player needs to jump or dodge incoming obstacles."""

    def __init__(self, **kwargs):
        super().__init__()

        self.width = 160
        self.height = 210
        self.ground_y = 180
        self.runner_x = 20
        self.runner_y = self.ground_y - 20
        self.runner_width = 10
        self.runner_height = 20
        self.gravity = 3
        self.jump_speed = -20

        self.runner_vel_y = 0
        self.is_jumping = False

        self.top_margin = 20

        self.obstacles = []
        self.obstacle_spawn_prob = 0.05

        self.action_space = spaces.Discrete(2)  # 0: do nothing, 1: jump
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(210, 160, 3), dtype=np.uint8
        )

        self.reset()

    def reset(self):
        self.runner_y = self.ground_y - self.runner_height
        self.runner_vel_y = 0
        self.is_jumping = False
        self.obstacles = []
        return self._get_obs(), {}

    def step(self, action):
        reward = 0
        done = False

        if action == 1 and not self.is_jumping:
            self.runner_vel_y = self.jump_speed
            self.is_jumping = True

        self.runner_y += self.runner_vel_y
        self.runner_vel_y += self.gravity

        if self.runner_y >= self.ground_y - self.runner_height:
            self.runner_y = self.ground_y - self.runner_height
            self.runner_vel_y = 0
            self.is_jumping = False

        # Spawn new obstacles

        if np.random.rand() < self.obstacle_spawn_prob and (
            len(self.obstacles) == 0 or self.obstacles[-1]["x"] < self.width - 50
        ):
            height = 20
            width = 15
            floating = np.random.rand() < 0.3  # 30% chance to float
            y_pos = (
                self.ground_y - height
                if not floating
                else self.ground_y - height - np.random.randint(40, 70)
            )
            self.obstacles.append(
                {"x": self.width, "y": y_pos, "width": width, "height": height}
            )

        # Move and remove off-screen obstacles
        for obs in self.obstacles:
            obs["x"] -= 3
        self.obstacles = [obs for obs in self.obstacles if obs["x"] + obs["width"] > 0]

        # Check collision
        for obs in self.obstacles:
            if (
                self.runner_x < obs["x"] + obs["width"]
                and self.runner_x + self.runner_width > obs["x"]
                and self.runner_y < obs["y"] + obs["height"]
                and self.runner_y + self.runner_height > obs["y"]
            ):
                reward = -1
                self.runner_y = self.height + 1
                done = True
                break

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        obs[:, :, 0] = 50  # Background color (dark blue)
        obs[:, :, 1] = 50
        obs[:, :, 2] = 100

        # Ground
        obs[self.ground_y : self.ground_y + (self.height - self.ground_y), :] = [
            150,
            150,
            255,
        ]

        # Celing
        obs[0 : self.top_margin, :] = [150, 150, 255]

        # runner
        obs[
            int(self.runner_y) : int(self.runner_y) + self.runner_height,
            self.runner_x : self.runner_x + self.runner_width,
        ] = [255, 255, 0]

        # Obstacles
        for obs_ in self.obstacles:
            y0 = max(0, obs_["y"])
            y1 = min(self.height, obs_["y"] + obs_["height"])
            x0 = max(0, obs_["x"])
            x1 = min(self.width, obs_["x"] + obs_["width"])
            obs[y0:y1, x0:x1] = [255, 0, 0]

        return obs
