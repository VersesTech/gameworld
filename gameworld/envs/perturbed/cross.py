import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw

from gameworld.envs.base.cross import Cross as BaseCross


class Cross(BaseCross):
    """Freeway (‘Cross’) with exact baseline when perturb=None, and mid-episode color/shape perturbations."""

    def __init__(self, perturb=None, perturb_step=5000, **kwargs):
        assert perturb in (None, "None", "color", "shape"), \
            "perturb must be None, 'color', or 'shape'"
        # normalize perturb flag
        self.perturb = None if perturb in (None, "None") else perturb
        self.perturb_step = perturb_step
        self.num_steps = 0

        # init base env
        super().__init__(**kwargs)

        # stash “original” palette
        self.bg_color       = (50,  50, 100)
        self.divider_color  = (255,255,255)
        self.player_color   = (255,255,  0)
        # we keep the original per-lane car_colors on BaseCross.car_colors
        # and use a uniform color for color-perturb
        self.perturb_car_color = (255,200,  0)

    def step(self, action):
        obs, reward, done, trunc, info = super().step(action)
        self.num_steps += 1
        if self.perturb and self.num_steps == self.perturb_step:
            self._apply_perturbation()
        return obs, reward, done, trunc, info

    def _apply_perturbation(self):
        if self.perturb == "color":
            # switch to a high-contrast palette
            self.bg_color       = (32,  32,  32)
            self.divider_color  = (200,200,200)
            self.player_color   = (  0,128,255)
            self.perturb_car_color = (255,200,  0)
        # shape-only: no color state changes (handled in drawing)

    def _get_obs(self):
        # before shape-perturbation, and for default/color-only, delegate to either base or our color-aware draw
        if not (self.perturb == "shape" and self.num_steps >= self.perturb_step):
            # if color-perturbed, we need to re-draw with our new palette
            if self.perturb == "color" and self.num_steps >= self.perturb_step:
                return self._draw_obs(color_only=True)
            return super()._get_obs()

        # shape-perturb: switch to custom shapes
        return self._draw_obs(shape_only=True)


    def _draw_obs(self, *, color_only=False, shape_only=False):
        # start with flat background + lane dividers
        obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        obs[:, :, :] = np.array(self.bg_color, dtype=np.uint8)
        for i in range(self.lane_count + 1):
            y = self.top_margin + i * self.lane_height
            obs[y : y + 3, :, :] = np.array(self.divider_color, dtype=np.uint8)

        img = Image.fromarray(obs)
        draw = ImageDraw.Draw(img)

        # draw player
        px, py, sz = self.player_x, self.player_y, self.player_size
        if shape_only:
            # circle player
            cx, cy, r = px + sz / 2, py + sz / 2, sz / 2
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=self.player_color)
        else:
            # square player
            draw.rectangle([px, py, px + sz, py + sz], fill=self.player_color)

        # draw cars
        for x, y, speed, orig_color in self.cars:
            # choose fill color
            if color_only and not shape_only:
                fill = self.perturb_car_color
            else:
                fill = tuple(orig_color)

            if shape_only:
                # triangle pointing in direction of motion
                if speed > 0:
                    pts = [(x, y), (x, y + self.car_size), (x + self.car_size, y + self.car_size // 2)]
                else:
                    pts = [(x + self.car_size, y), (x + self.car_size, y + self.car_size), (x, y + self.car_size // 2)]
                draw.polygon(pts, fill=fill)
            else:
                # square car
                draw.rectangle([x, y, x + self.car_size, y + self.car_size], fill=fill)

        return np.array(img)
