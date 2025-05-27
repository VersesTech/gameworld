import numpy as np
from PIL import Image, ImageDraw

from gameworld.envs.base.explode import Explode as BaseExplode


class Explode(BaseExplode):
    """ Explode with exact baseline when perturb=None, and mid‐episode color/shape perturbations."""

    def __init__(self, player_y=170, bomber_y=20, perturb=None, perturb_step=5000, **kwargs):
        assert perturb in (None, "None", "color", "shape"), \
            "perturb must be None, 'color', or 'shape'"
        # normalize perturb
        self.perturb = None if perturb in (None, "None") else perturb
        self.perturb_step = perturb_step
        self.num_steps = 0

        # stash original geometry & colors
        # these match BaseExplode.__init__
        self.orig_bucket_w, self.orig_bucket_h = 30, 12
        self.orig_bomb_size = 5
        self.orig_bomber_w, self.orig_bomber_h = 20, 10

        super().__init__(player_y=player_y, bomber_y=bomber_y, **kwargs)

        # override mutable attrs on self
        self.bucket_width   = self.orig_bucket_w
        self.bucket_height  = self.orig_bucket_h
        self.bomb_size      = self.orig_bomb_size
        self.bomber_width   = self.orig_bomber_w
        self.bomber_height  = self.orig_bomber_h

        # override base’s hard‐coded colors
        self.bg_color       = (50,  50, 100)
        self.bucket_color   = (255,255,  0)
        self.bomb_color     = (255,  0,  0)
        self.bomber_color   = (  0,255,  0)

    def step(self, action):
        obs, reward, done, trunc, info = super().step(action)
        self.num_steps += 1
        if self.perturb and self.num_steps == self.perturb_step:
            self.apply_perturbation()
        return obs, reward, done, trunc, info

    def apply_perturbation(self):
        if self.perturb == "color":
            # high‐contrast swap
            self.bg_color      = (32,  32,  32)
            self.bucket_color  = (  0,128,255)
            self.bomb_color    = (255, 64,128)
            self.bomber_color  = (255,200,  0)
        elif self.perturb == "shape":
            # enlarge shapes
            self.bucket_width   = int(self.orig_bucket_w * 1.5)
            self.bucket_height  = int(self.orig_bucket_h * 1.5)
            self.bomb_size      = int(self.orig_bomb_size * 2)
            self.bomber_width   = int(self.orig_bomber_w * 1.5)
            self.bomber_height  = int(self.orig_bomber_h * 1.5)

    def _get_obs(self):
        # shape‐only post‐perturb
        if self.perturb == "shape" and self.num_steps >= self.perturb_step:
            return self._draw_shape_obs()
        # color‐only post‐perturb
        if self.perturb == "color" and self.num_steps >= self.perturb_step:
            return self._draw_color_obs()
        # baseline
        return super()._get_obs()

    def _draw_color_obs(self):
        # pure numpy draw using updated colors & sizes
        obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        obs[:, :, :] = np.array(self.bg_color, dtype=np.uint8)

        # bucket
        y0, y1 = self.player_y, self.player_y + self.bucket_height
        x0, x1 = self.player_x, self.player_x + self.bucket_width
        obs[y0:y1, x0:x1] = np.array(self.bucket_color, dtype=np.uint8)

        # bombs
        for x, y, _ in self.bombs:
            y0, y1 = int(y), int(y + self.bomb_size)
            x0, x1 = int(x), int(x + self.bomb_size)
            obs[y0:y1, x0:x1] = np.array(self.bomb_color, dtype=np.uint8)

        # bomber
        by0 = int(self.bomber_y)
        by1 = by0 + self.bomber_height
        bx0 = int(self.bomber_x - self.bomber_width/2)
        bx1 = bx0 + self.bomber_width
        obs[by0:by1, bx0:bx1] = np.array(self.bomber_color, dtype=np.uint8)

        return obs

    def _draw_shape_obs(self):
        # PIL draw for custom shapes & updated sizes/colors
        obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        obs[:, :, :] = np.array(self.bg_color, dtype=np.uint8)
        img = Image.fromarray(obs)
        draw = ImageDraw.Draw(img)

        # bucket → circle
        bx0 = self.player_x
        by0 = self.player_y
        bx1 = bx0 + self.bucket_width
        by1 = by0 + self.bucket_height
        draw.ellipse([bx0, by0, bx1, by1], fill=self.bucket_color)

        # bombs → circles
        for x, y, _ in self.bombs:
            x0, y0 = int(x), int(y)
            x1, y1 = x0 + self.bomb_size, y0 + self.bomb_size
            draw.ellipse([x0, y0, x1, y1], fill=self.bomb_color)

        # bomber → triangle
        half = self.bomber_width // 2
        tx, ty = self.bomber_x, self.bomber_y
        pts = [(tx, ty),
               (tx - half, ty + self.bomber_height),
               (tx + half, ty + self.bomber_height)]
        draw.polygon(pts, fill=self.bomber_color)

        return np.array(img)
