import numpy as np
from gymnasium import spaces
from gameworld.envs.base.base_env import GameworldEnv


def sign(x):
    return x / np.abs(x)


class Bounce(GameworldEnv):
    """Based on the Atari Pong game.

    Player moves a paddle up and down to bounce the ball back to the enemy.
    """

    def __init__(self, player_x=135, opponent_x=15, **kwargs):
        super().__init__()

        self.bg_color = (50, 50, 100)
        self.player_color = (255, 255, 0)
        self.opponent_color = (0, 255, 0)
        self.ball_color = (255, 0, 0)
        self.wall_color = [150, 150, 255]

        self.width = 160
        self.height = 210
        self.wall_width = 15
        self.paddle_width = 10
        self.paddle_height = 40
        self.ball_size = 5
        self.player_x = player_x
        self.player_y = self.height // 2 - self.paddle_height // 2
        self.player_speed = 10
        self.opponent_x = opponent_x
        self.opponent_y = self.height // 2 - self.paddle_height // 2
        self.reset()

        self.action_space = spaces.Discrete(3)  # 0: stay, 1: up, 2: down
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(210, 160, 3), dtype=np.uint8
        )

    def reset(self):
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        self.opponent_y = self.height // 2 - self.paddle_height // 2
        self.ball_dx = np.random.choice([-3, 3])
        self.ball_dy = np.random.choice([-3, 3])
        return self._get_obs(), {}

    def step(self, action):
        if action == 1:
            self.player_y = max(self.player_y - self.player_speed, 0)
        elif action == 2:
            self.player_y = min(
                self.player_y + self.player_speed, self.height - self.paddle_height
            )

        # Simple AI for opponent
        opponent_action = 0
        if self.ball_y > self.opponent_y + self.paddle_height - 2:
            self.opponent_y = min(self.opponent_y + 4, self.height - self.paddle_height)
            opponent_action = 1
        elif self.ball_y < self.opponent_y + 2:
            self.opponent_y = max(self.opponent_y - 4, 0)
            opponent_action = 2

        # Ball movement
        self.ball_x = int(self.ball_x + self.ball_dx)
        self.ball_y = int(self.ball_y + self.ball_dy)

        # Ball collision with walls
        if (
            self.ball_y < self.wall_width
            or self.ball_y > self.height - self.ball_size - self.wall_width
        ):
            self.ball_dy = -self.ball_dy
            # don't bounce of wall with 0 velocity
            self.ball_dy = np.sign(self.ball_dy) * np.ceil(np.abs(self.ball_dy))
            self.ball_y = np.clip(
                self.ball_y,
                a_min=self.wall_width,
                a_max=self.height - self.ball_size - self.wall_width,
            )

        # Ball collision with paddles
        if (
            self.ball_x >= self.player_x - self.ball_size
            and self.ball_x <= self.player_x + self.paddle_width - self.ball_size
            and self.player_y <= self.ball_y <= self.player_y + self.paddle_height
        ):
            # Keep going in the same direction, with a fixed velocit of 1.5
            base_speed = sign(self.ball_dy) * 1.5

            # If you are moving up, you push the ball up & vice versa
            # If you hit the ball stationary, nothing changes
            action_impact = (
                3 if action == 1 else -3 if action == 2 else sign(self.ball_dy) * 6
            )

            self.ball_dy = base_speed + action_impact

            self.ball_x = self.player_x - self.ball_size
            self.ball_dx = -self.ball_dx
        elif (
            self.ball_x <= self.opponent_x + self.paddle_width
            and self.ball_x >= self.opponent_x
            and self.opponent_y <= self.ball_y <= self.opponent_y + self.paddle_height
        ):
            # Keep going in the same direction, with a fixed velocit of 1.5
            base_speed = sign(self.ball_dy) * 1.5

            # If you are moving up, you push the ball up & vice versa
            # If you hit the ball stationary, nothing changes
            action_impact = (
                3
                if opponent_action == 1
                else -3
                if opponent_action == 2
                else sign(self.ball_dy) * 6
            )

            self.ball_dy = base_speed + action_impact

            self.ball_dx = -self.ball_dx
            self.ball_x = self.opponent_x + self.paddle_width

        # Check for scoring
        reward = 0
        done = False
        if self.ball_x <= 0:
            # dissappear ball
            self.ball_x = -1
            reward = 1  # Player scores
            done = True
        elif self.ball_x >= self.width - self.ball_size:
            # dissappear ball
            self.ball_x = -1
            reward = -1  # Opponent scores
            done = True

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        obs[:, :, :] = self.bg_color
        obs[
            self.player_y : self.player_y + self.paddle_height,
            self.player_x : self.player_x + self.paddle_width,
        ] = self.player_color
        obs[
            self.opponent_y : self.opponent_y + self.paddle_height,
            self.opponent_x : self.opponent_x + self.paddle_width,
        ] = self.opponent_color
        if self.ball_x > 0:
            obs[
                self.ball_y : self.ball_y + self.ball_size,
                self.ball_x : self.ball_x + self.ball_size,
            ] = self.ball_color

        obs[: self.wall_width, :, :] = self.wall_color
        obs[-self.wall_width :, :, :] = self.wall_color
        return obs
