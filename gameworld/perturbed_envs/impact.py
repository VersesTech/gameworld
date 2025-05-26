import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw

from gameworld.envs.base import GameworldEnv


class Impact(GameworldEnv):
    """ Based on the Atari Breakout game.
    
        Player moves paddle left right to bounce a ball and destroy bricks.
    """

    def __init__(
        self,
        paddle_y: int = 190,
        perturb: str | None = None,
        perturb_step: int = 5000,
        **kwargs,
    ):
        super().__init__()
        assert perturb in (None, "None", "color", "shape"), (
            "perturb must be None, 'color', or 'shape'"
        )
        self.perturb = None if perturb in (None, "None") else perturb  # normalise
        self.perturb_step = perturb_step
        self.num_steps = 0  # global counter

        # -------- world geometry --------
        self.width, self.height = 160, 210
        self.paddle_y = paddle_y
        self.ball_start_y = 100

        # original sizes (needed for shape‑reset)
        self.orig_paddle_w, self.orig_paddle_h = 30, 8
        self.orig_ball_size = 4
        self.orig_brick_rows, self.orig_brick_cols = 5, 10
        self.orig_brick_h = 10
        self.orig_brick_w = self.width // self.orig_brick_cols

        # live, mutable sizes
        self.paddle_width, self.paddle_height = self.orig_paddle_w, self.orig_paddle_h
        self.ball_size = self.orig_ball_size
        self.brick_rows, self.brick_cols = self.orig_brick_rows, self.orig_brick_cols
        self.brick_height = self.orig_brick_h
        self.brick_width = self.orig_brick_w

        # speeds & physics
        self.paddle_speed = 12
        self.ball_speed = 4
        self.ball_dx = self.ball_speed
        self.ball_dy = -self.ball_speed

        # -------- colours --------
        self.paddle_color = (255, 255, 0)  # yellow
        self.ball_color = (255, 0, 0)  # red
        self.brick_color = (0, 255, 0)  # green
        self.bg_color = (50, 50, 100)  # dark blue

        # game state containers – reset() will (re)initialise
        self.bricks = np.ones((self.brick_rows, self.brick_cols), dtype=bool)
        self.lives = 3
        self.paddle_x = self.width // 2 - self.paddle_width // 2
        self.ball_x = self.width // 2
        self.ball_y = self.ball_start_y

        # spaces
        self.action_space = spaces.Discrete(3)  # stay, left, right
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )

        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self):
        self.lives = 3
        self.paddle_x = self.width // 2 - self.paddle_width // 2
        self.ball_x = self.width // 2
        self.ball_y = self.ball_start_y
        self.ball_dx = np.random.choice([self.ball_speed, -self.ball_speed])
        self.ball_dy = self.ball_speed
        self.bricks[:, :] = True
        self.num_steps = 0
        # no perturbation applied yet – sizes/colours already set in __init__
        return self._get_obs(), {}

    def step(self, action: int):
        # ------------------------------------------------------------ update game state
        # paddle movement
        if action == 1:
            self.paddle_x = max(0, self.paddle_x - self.paddle_speed)
        elif action == 2:
            self.paddle_x = min(
                self.width - self.paddle_width, self.paddle_x + self.paddle_speed
            )

        # ball movement
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        # wall collisions
        if self.ball_x <= 0 or self.ball_x + self.ball_size >= self.width:
            self.ball_dx *= -1
            self.ball_x = max(0, min(self.ball_x, self.width - self.ball_size))
        if self.ball_y <= 0:
            self.ball_dy *= -1
            self.ball_y = 0

        # paddle collision
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

        # brick collisions
        reward = 0
        for row in range(self.brick_rows):
            for col in range(self.brick_cols):
                if not self.bricks[row, col]:
                    continue
                bx = col * self.brick_width
                by = row * self.brick_height + 20
                if (
                    bx <= self.ball_x <= bx + self.brick_width
                    and by <= self.ball_y <= by + self.brick_height
                ):
                    self.bricks[row, col] = False
                    # basic overlap resolution
                    overlap_l = self.ball_x + self.ball_size - bx
                    overlap_r = bx + self.brick_width - self.ball_x
                    overlap_t = self.ball_y + self.ball_size - by
                    overlap_b = by + self.brick_height - self.ball_y
                    if min(overlap_l, overlap_r) < min(overlap_t, overlap_b):
                        self.ball_dx *= -1
                    else:
                        self.ball_dy *= -1
                    reward += 1

        # miss – bottom wall
        done = False
        if self.ball_y >= self.height:
            self.lives -= 1
            reward -= 1
            if self.lives > 0:
                # respawn ball
                self.ball_x = self.width // 2
                self.ball_y = self.ball_start_y
                self.ball_dx = np.random.choice([self.ball_speed, -self.ball_speed])
                self.ball_dy = self.ball_speed
            else:
                done = True

        # level clear
        if self.bricks.sum() == 0:
            done = True

        # ------------------------------------------------------------ perturbation check
        self.num_steps += 1
        if self.perturb and self.num_steps == self.perturb_step:
            self._apply_perturbation()

        return self._get_obs(), reward, done, False, {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_perturbation(self):
        # Only called once when num_steps == perturb_step.
        print(f"Applying perturbation: {self.perturb}")
        if self.perturb == "color":
            # swap palette – high contrast scheme
            self.paddle_color = (0, 128, 255)  # bright blue
            self.ball_color = (255, 64, 128)  # pink
            self.brick_color = (255, 200, 0)  # orange
            self.bg_color = (32, 32, 32)  # nearly black

        elif self.perturb == "shape":
            # enlarge primary entities
            scale = 1.5
            self.paddle_width = int(self.orig_paddle_w * scale)
            self.paddle_height = int(self.orig_paddle_h * scale)
            self.ball_size = int(self.orig_ball_size * 2)
            self.brick_height = int(self.orig_brick_h * scale)
            # keep number of columns constant; adjust width per brick
            self.brick_width = self.width // self.brick_cols

            # ensure paddle within bounds
            self.paddle_x = min(self.paddle_x, self.width - self.paddle_width)

    # -------------------------------------------------------------- rendering
    def _get_obs(self):
        img = Image.new("RGB", (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(img)

        # paddle
        px0, py0 = self.paddle_x, self.paddle_y
        px1, py1 = px0 + self.paddle_width, py0 + self.paddle_height
        if self.perturb == "shape" and self.num_steps >= self.perturb_step:
            draw.ellipse([px0, py0, px1, py1], fill=self.paddle_color)
        else:
            draw.rectangle([px0, py0, px1, py1], fill=self.paddle_color)

        # ball (only render if on‑screen)
        if self.ball_y < self.height:
            bx0, by0 = self.ball_x, self.ball_y
            bx1, by1 = bx0 + self.ball_size, by0 + self.ball_size
            if self.perturb == "shape" and self.num_steps >= self.perturb_step:
                draw.rectangle([bx0, by0, bx1, by1], fill=self.ball_color)  # ball is already a square, so keep as rect
            else:
                draw.rectangle([bx0, by0, bx1, by1], fill=self.ball_color)

        # bricks
        for row in range(self.brick_rows):
            for col in range(self.brick_cols):
                if not self.bricks[row, col]:
                    continue
                bx = col * self.brick_width
                by = row * self.brick_height + 20
                bx0, by0 = bx, by
                bx1, by1 = bx + self.brick_width, by + self.brick_height
                if self.perturb == "shape" and self.num_steps >= self.perturb_step:
                    draw.ellipse([bx0, by0, bx1, by1], fill=self.brick_color)
                else:
                    draw.rectangle([bx0, by0, bx1, by1], fill=self.brick_color)

        return np.array(img)

