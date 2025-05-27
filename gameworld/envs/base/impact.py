import numpy as np
from gymnasium import spaces
from gameworld.envs.base.base_env import GameworldEnv


class Impact(GameworldEnv):
    """ Based on the Atari Breakout game.
    
        Player moves paddle left right to bounce a ball and destroy bricks.
    """

    def __init__(self, paddle_y=190, **kwargs):
        super().__init__()

        self.width = 160
        self.height = 210
        self.paddle_width = 30
        self.paddle_height = 8
        self.paddle_speed = 12
        self.ball_size = 4

        self.paddle_y = paddle_y
        self.paddle_x = self.width // 2 - self.paddle_width // 2

        self.ball_x = self.width // 2
        self.ball_y = self.height // 2

        self.ball_speed = 4
        self.ball_dx = self.ball_speed
        self.ball_dy = -self.ball_speed
        self.ball_start_y = 100

        self.brick_rows = 5
        self.brick_cols = 10
        self.brick_width = self.width // self.brick_cols
        self.brick_height = 10
        self.bricks = np.ones((self.brick_rows, self.brick_cols), dtype=bool)
        self.lives = 3
        self.reset()

        self.action_space = spaces.Discrete(3)  # 0: stay, 1: left, 2: right
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(210, 160, 3), dtype=np.uint8
        )

    def reset(self):
        self.lives = 3
        # self.paddle_x = self.width // 2 - self.paddle_width // 2
        self.ball_x = self.width // 2
        self.ball_y = self.ball_start_y
        self.ball_dx = np.random.choice([self.ball_speed, -self.ball_speed])
        self.ball_dy = self.ball_speed
        self.bricks[:, :] = True
        return self._get_obs(), {}

    def step(self, action):
        if action == 1:
            self.paddle_x = max(0, self.paddle_x - self.paddle_speed)
        elif action == 2:
            self.paddle_x = min(
                self.width - self.paddle_width, self.paddle_x + self.paddle_speed
            )

        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        # Ball-wall collision
        if self.ball_x <= 0 or self.ball_x + self.ball_size >= self.width:
            self.ball_x = 0 if self.ball_x <= 0 else self.width - self.ball_size
            self.ball_dx *= -1
        if self.ball_y <= 0:
            self.ball_y = 0
            self.ball_dy *= -1

        # Ball-paddle collision
        if (
            self.paddle_y
            <= self.ball_y + self.ball_size
            <= self.paddle_y + self.paddle_height
            and self.paddle_x - self.ball_size
            < self.ball_x
            <= self.paddle_x + self.paddle_width
        ):
            self.ball_dy *= -1
            self.ball_y = self.paddle_y - self.ball_size

        # Ball-brick collision
        reward = 0
        collided = False
        for row in range(self.brick_rows):
            for col in range(self.brick_cols):
                if self.bricks[row, col]:
                    bx = col * self.brick_width
                    by = row * self.brick_height + 20
                    if (
                        bx <= self.ball_x <= bx + self.brick_width
                        and by <= self.ball_y <= by + self.brick_height
                        and not collided
                    ):
                        self.bricks[row, col] = False

                        overlap_left = self.ball_x + self.ball_size - bx
                        overlap_right = bx + self.brick_width - self.ball_x
                        overlap_top = self.ball_y + self.ball_size - by
                        overlap_bottom = by + self.brick_height - self.ball_y

                        min_overlap_x = min(overlap_left, overlap_right)
                        min_overlap_y = min(overlap_top, overlap_bottom)

                        if min_overlap_x < min_overlap_y:
                            self.ball_dx *= -1
                        else:
                            self.ball_dy *= -1

                        collided = True  # only hit one brick at a time
                        reward = 1

        # Ball missed
        done = False
        if self.ball_y >= self.height:
            self.lives -= 1
            if self.lives > 0:
                self.ball_x = self.width // 2
                self.ball_y = self.ball_start_y
                self.ball_dx = np.random.choice([self.ball_speed, -self.ball_speed])
                self.ball_dy = self.ball_speed
            else:
                done = True
            reward = -1

        # All bricks destroyed
        if self.bricks.sum() == 0:
            done = True

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        obs[:, :] = [50, 50, 100]  # Dark blue background

        # Paddle
        obs[
            self.paddle_y : self.paddle_y + self.paddle_height,
            self.paddle_x : self.paddle_x + self.paddle_width,
        ] = [
            255,
            255,
            0,
        ]  # Yellow paddle

        # Ball
        if self.ball_y < self.height:
            obs[
                self.ball_y : self.ball_y + self.ball_size,
                self.ball_x : self.ball_x + self.ball_size,
            ] = [255, 0, 0]

        # Bricks
        for row in range(self.brick_rows):
            for col in range(self.brick_cols):
                if self.bricks[row, col]:
                    bx = col * self.brick_width
                    by = row * self.brick_height + 20
                    obs[by : by + self.brick_height, bx : bx + self.brick_width] = [
                        0,
                        255,
                        0,
                    ]
        return obs
