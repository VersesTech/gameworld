import numpy as np
from gymnasium import spaces
from gameworld.envs.base.base_env import GameworldEnv


class Aviate(GameworldEnv):
    """Based on the FlappyBird game

    Player needs to flap the birds wings to navigate through the pipes.
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.width = 160
        self.height = 210
        self.gravity = 1
        self.jump_speed = -10

        self.bird_x = 30
        self.bird_y = 100
        self.bird_radius = 5
        self.bird_vel_y = 0

        self.pipe_gap = 100
        self.pipe_width = 20
        self.pipe_speed = 2
        self.pipe_spawn_prob = 0.03

        self.bg_color = (50, 50, 100)
        self.bird_color = (255, 255, 0)
        self.pipe_color_lower = (0, 255, 30)
        self.pipe_color_upper = (30, 255, 0)

        self.pipes = []

        self.action_space = spaces.Discrete(2)  # 0: do nothing, 1: flap
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(210, 160, 3), dtype=np.uint8
        )

        self.reset()

    def reset(self):
        self.bird_y = 160
        self.bird_vel_y = 0
        self.pipes = []
        gap_y = np.random.randint(40, self.height - 40 - self.pipe_gap)
        self.pipes.append({"x": self.width, "gap_y": gap_y})
        return self._get_obs(), {}

    def step(self, action):
        reward = 0
        done = False

        if action == 1 and self.bird_vel_y > 2:
            self.bird_vel_y = self.jump_speed

        self.bird_y += self.bird_vel_y
        self.bird_vel_y += self.gravity

        if self.bird_y < 0 or self.bird_y > self.height:
            reward = -1
            done = True

        if np.random.rand() < self.pipe_spawn_prob and (
            len(self.pipes) == 0 or self.pipes[-1]["x"] < self.width - 80
        ):
            gap_y = np.random.randint(40, self.height - 40 - self.pipe_gap)
            self.pipes.append({"x": self.width, "gap_y": gap_y})

        for pipe in self.pipes:
            pipe["x"] -= self.pipe_speed

        self.pipes = [pipe for pipe in self.pipes if pipe["x"] + self.pipe_width > 0]

        for pipe in self.pipes:
            if (
                self.bird_x + self.bird_radius > pipe["x"]
                and self.bird_x - self.bird_radius < pipe["x"] + self.pipe_width
            ):
                if not (pipe["gap_y"] < self.bird_y < pipe["gap_y"] + self.pipe_gap):
                    reward = -1
                    self.bird_y = self.height + 1
                    done = True
                    break

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        obs[:, :, :] = self.bg_color

        # Bird
        if self.bird_y > -self.bird_radius and self.bird_y < self.height:
            y0 = max(0, int(self.bird_y - self.bird_radius))
            y1 = min(self.height, int(self.bird_y + self.bird_radius))
            x0 = max(0, self.bird_x - self.bird_radius)
            x1 = min(self.width, self.bird_x + self.bird_radius)
            obs[y0:y1, x0:x1] = self.bird_color

        for pipe in self.pipes:
            x0 = max(0, pipe["x"])
            x1 = pipe["x"] + self.pipe_width
            obs[0 : pipe["gap_y"], x0:x1] = self.pipe_color_upper
            obs[pipe["gap_y"] + self.pipe_gap : self.height, x0:x1] = (
                self.pipe_color_lower
            )

        return obs
