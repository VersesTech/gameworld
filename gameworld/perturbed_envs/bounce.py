import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw

from gameworld.envs.base import GameworldEnv


def sign(x):
    return x / np.abs(x)


class Bounce(GameworldEnv):
    """ Based on the Atari Pong game.
    
        Player moves a paddle up and down to bounce the ball back to the enemy.
    """

    def __init__(
        self,
        player_x=135,
        opponent_x=15,
        perturb=None,
        perturb_step=5000,
        **kwargs,
    ):
        super().__init__()
        assert perturb in (
            None,
            "None",
            "color",
            "shape",
        ), "perturb must be None, 'color', or 'shape'"
        self.perturb = None if perturb in (None, "None") else perturb
        self.perturb_step = perturb_step
        self.num_steps = 0

        # screen and wall
        self.width = 160
        self.height = 210
        self.wall_width = 15

        # original paddle & ball sizes
        self.orig_paddle_w = 10
        self.orig_paddle_h = 40
        self.orig_ball_size = 5

        # mutable sizes
        self.paddle_width = self.orig_paddle_w
        self.paddle_height = self.orig_paddle_h
        self.ball_size = self.orig_ball_size

        # paddle positions & speeds
        self.player_x = player_x
        self.player_y = self.height // 2 - self.paddle_height // 2
        self.opponent_x = opponent_x
        self.opponent_y = self.height // 2 - self.paddle_height // 2
        self.player_speed = 10
        # self.opponent_speed = 4

        # ball state

        # colors
        self.bg_color = (50, 50, 100)
        self.wall_color = (150, 150, 255)
        self.player_color = (255, 255, 0)
        self.opponent_color = (0, 255, 0)
        self.ball_color = (255, 0, 0)

        self.action_space = spaces.Discrete(3)  # stay, up, down
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.reset()

    def reset_ball(self):
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        self.ball_dx = np.random.choice([-3, 3])
        self.ball_dy = np.random.choice([-3, 3])

    def reset(self):
        self.reset_ball()
        self.opponent_y = self.height // 2 - self.paddle_height // 2
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
                else -3 if opponent_action == 2 else sign(self.ball_dy) * 6
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
        #     # perturb timing
        self.num_steps += 1
        if self.perturb and self.num_steps == self.perturb_step:
            self._apply_perturbation()
        return self._get_obs(), reward, done, False, {}

    def _apply_perturbation(self):
        print(f"Applying perturbation: {self.perturb}")
        if self.perturb == "color":
            self.bg_color = (32, 32, 32)
            self.player_color = (0, 128, 255)
            self.opponent_color = (255, 200, 0)
            self.ball_color = (0, 255, 255)
        elif self.perturb == "shape":
            scale = 1.2
            self.paddle_width = int(self.orig_paddle_w * scale)
            self.paddle_height = int(self.orig_paddle_h * scale)
            self.ball_size = int(self.orig_ball_size * scale)
            # clamp
            self.player_y = min(self.player_y, self.height - self.paddle_height)
            self.opponent_y = min(self.opponent_y, self.height - self.paddle_height)

    def _get_obs(self):
        img = Image.new("RGB", (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(img)

        # walls
        draw.rectangle([0, 0, self.width, self.wall_width], fill=self.wall_color)
        draw.rectangle(
            [0, self.height - self.wall_width, self.width, self.height],
            fill=self.wall_color,
        )

        # paddles
        # choose shape
        if self.perturb == "shape" and self.num_steps >= self.perturb_step:
            # player as triangle
            px0, py0 = self.player_x, self.player_y
            px1, py1 = px0, py0 + self.paddle_height
            px2 = px0 + self.paddle_width
            points = [(px0, py0), (px0, py1), (px2, py0 + self.paddle_height // 2)]
            draw.polygon(points, fill=self.player_color)
            # opponent as triangle
            ox0, oy0 = self.opponent_x + self.paddle_width, self.opponent_y
            ox1, oy1 = (
                self.opponent_x + self.paddle_width,
                self.opponent_y + self.paddle_height,
            )
            ox2 = self.opponent_x
            opp_pts = [(ox0, oy0), (ox0, oy1), (ox2, oy0 + self.paddle_height // 2)]
            draw.polygon(opp_pts, fill=self.opponent_color)
        else:
            # rectangles
            draw.rectangle(
                [
                    self.player_x,
                    self.player_y,
                    self.player_x + self.paddle_width,
                    self.player_y + self.paddle_height,
                ],
                fill=self.player_color,
            )
            draw.rectangle(
                [
                    self.opponent_x,
                    self.opponent_y,
                    self.opponent_x + self.paddle_width,
                    self.opponent_y + self.paddle_height,
                ],
                fill=self.opponent_color,
            )

        # ball
        bx0, by0 = self.ball_x, self.ball_y
        bx1, by1 = bx0 + self.ball_size, by0 + self.ball_size
        if self.perturb == "shape" and self.num_steps >= self.perturb_step:
            # ball as circle
            draw.ellipse([bx0, by0, bx1, by1], fill=self.ball_color)
        else:
            # original as square (rectangle)
            draw.rectangle([bx0, by0, bx1, by1], fill=self.ball_color)

        return np.array(img)