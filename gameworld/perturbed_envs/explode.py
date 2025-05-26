import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw

from gameworld.envs.base import GameworldEnv
from gameworld.envs.utils import make_ball


class Explode(GameworldEnv):
    """ Based on the Atari Kaboom game.
    
        Player moves left right to catch bombs dropping from a spaceship.
    """
    def __init__(
        self,
        player_y=170,
        bomber_y=20,
        perturb=None,  # only: None, "color", or "shape"
        perturb_step=5000,
        **kwargs,
    ):
        super().__init__()
        assert perturb in (
            None,
            "None",
            "color",
            "shape",
        ), "perturb must be None, 'color' or 'shape'"
        self.perturb = perturb
        self.perturb_step = perturb_step
        self.num_steps = 0

        # world dims & movement
        self.width, self.height = 160, 210
        self.player_y, self.player_speed = player_y, 8
        self.bomber_y, self.bomber_dx = bomber_y, 2

        # --- geometry & colors ---
        self.orig_bucket_w, self.orig_bucket_h = 30, 12
        self.bucket_width, self.bucket_height = 30, 12
        self.bucket_color = (255, 255, 0)  # yellow

        self.orig_bomb_size = 5
        self.bomb_size = 5
        self.bomb_color = (255, 0, 0)  # red

        self.orig_bomber_w, self.orig_bomber_h = 20, 10
        self.bomber_width, self.bomber_height = 20, 10
        self.bomber_color = (0, 255, 0)  # green

        self.bg_color = (50, 50, 100)  # dark blue background

        self.bomb_img = make_ball(size=(10, 10))

        # action & observation spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )

        self.reset()

    def reset(self):
        self.player_x = self.width // 2 - self.bucket_width // 2
        self.bomber_x = 10
        self.bombs = []
        return self._get_obs(), {}

    def step(self, action):
        # player move
        if action == 1:
            self.player_x = max(0, self.player_x - self.player_speed)
        elif action == 2:
            self.player_x = min(
                self.width - self.bucket_width, self.player_x + self.player_speed
            )

        # bomber move
        self.bomber_x += self.bomber_dx
        if (
            self.bomber_x <= 10
            or self.bomber_x >= self.width - 10
            or np.random.rand() < 0.02
        ):
            self.bomber_dx *= -1

        # spawn bomb
        if len(self.bombs) < 1 and np.random.rand() < 0.05:
            self.bombs.append([int(self.bomber_x), int(self.bomber_y), 2])

        # update bombs
        for b in self.bombs:
            b[1] += b[2]
            b[2] += 0.5

        # rewards
        reward = 0
        for b in list(self.bombs):
            if (
                self.player_y - self.bomb_size
                <= b[1]
                <= self.player_y + self.bucket_height
                and self.player_x <= b[0] <= self.player_x + self.bucket_width
            ):
                reward += 1
                self.bombs.remove(b)
        for b in list(self.bombs):
            if b[1] >= self.height:
                reward -= 1
                self.bombs.remove(b)

        # perturb?
        self.num_steps += 1
        if self.perturb and self.num_steps == self.perturb_step:
            self.apply_perturbation()

        return self._get_obs(), reward, False, False, {}

    def apply_perturbation(self):
        print(f"Applying perturbation: {self.perturb}")

        if self.perturb == "color":
            # four fixed new colors
            self.bucket_color = (0, 128, 255)  # bright blue
            self.bomb_color = (255, 64, 128)  # pink
            self.bg_color = (32, 32, 32)  # almost black
            self.bomber_color = (255, 200, 0)  # orange

        elif self.perturb == "shape":
            # enlarge obs sizes
            self.bucket_width = int(self.orig_bucket_w * 1.5)
            self.bucket_height = int(self.orig_bucket_h * 1.5)
            self.bomb_size = self.orig_bomb_size * 2
            self.bomber_width = int(self.orig_bomber_w * 1.5)
            self.bomber_height = int(self.orig_bomber_h * 1.5)

        # None -> baseline retained exactly

    def _get_obs(self):
        img = Image.new("RGB", (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(img)

        # bucket
        bx0 = self.player_x
        by0 = self.player_y
        bx1 = bx0 + self.bucket_width
        by1 = by0 + self.bucket_height
        if self.perturb == "shape" and self.num_steps >= self.perturb_step:
            draw.ellipse([bx0, by0, bx1, by1], fill=self.bucket_color)
        else:
            draw.rectangle([bx0, by0, bx1, by1], fill=self.bucket_color)

        # bombs
        for x, y, _ in self.bombs:
            x0, y0 = x, y
            x1, y1 = x + self.bomb_size, y + self.bomb_size
            if self.perturb == "shape" and self.num_steps >= self.perturb_step:
                draw.ellipse([x0, y0, x1, y1], fill=self.bomb_color)
            else:
                draw.rectangle([x0, y0, x1, y1], fill=self.bomb_color)

        # bomber
        half = self.bomber_width // 2
        tx, ty = self.bomber_x, self.bomber_y
        if self.perturb == "shape" and self.num_steps >= self.perturb_step:
            pts = [
                (tx, ty),
                (tx - half, ty + self.bomber_height),
                (tx + half, ty + self.bomber_height),
            ]
            draw.polygon(pts, fill=self.bomber_color)
        else:
            draw.rectangle(
                [tx - half, ty, tx + half, ty + self.bomber_height],
                fill=self.bomber_color,
            )

        return np.array(img)
