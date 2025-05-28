import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw

from gameworld.envs.base.aviate import Aviate as BaseAviate


class Aviate(BaseAviate):
    """Aviate with exact baseline when perturb=None, and mid-episode color/shape perturbations."""

    def __init__(
        self,
        perturb=None,
        perturb_step=5000,
        **kwargs,
    ):
        assert perturb in (None, "None", "color", "shape"), \
            "perturb must be None, 'color', or 'shape'"
        self.perturb = None if perturb in (None, "None") else perturb
        self.perturb_step = perturb_step
        self.num_steps = 0

        super().__init__(**kwargs)
       
    def step(self, action):
        obs, reward, done, trunc, info = super().step(action)
        self.num_steps += 1
        if self.perturb and self.num_steps == self.perturb_step:
            self._apply_perturbation()
        return obs, reward, done, trunc, info

    def _apply_perturbation(self):
        if self.perturb == "color":
            self.bg_color = (32, 32, 32)
            self.bird_color = (255, 64, 128)
            self.pipe_color_upper = (255, 200, 0)
            self.pipe_color_lower = (0, 255, 255)
        # shape-only: handled in custom drawing

    def _get_obs(self):
        # before shape-perturbation, delegate to base for pixel-perfect match
        if not (self.perturb == "shape" and self.num_steps >= self.perturb_step):
            return super()._get_obs()

        # custom shape: bird circle + triangular pipes
        obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        obs[:, :, :] = np.array(self.bg_color, dtype=np.uint8)
        img = Image.fromarray(obs)
        draw = ImageDraw.Draw(img)
        # bird as circle
        x0 = self.bird_x - self.bird_radius
        y0 = int(self.bird_y - self.bird_radius)
        x1 = self.bird_x + self.bird_radius
        y1 = int(self.bird_y + self.bird_radius)
        draw.ellipse([x0, y0, x1, y1], fill=self.bird_color)
        # pipes as triangles
        for pipe in self.pipes:
            x = pipe["x"]
            gap_y = pipe["gap_y"]
            # upper triangle
            pts_u = [
                (x, 0),
                (x + self.pipe_width // 2, gap_y),
                (x + self.pipe_width, 0),
            ]
            draw.polygon(pts_u, fill=self.pipe_color_upper)
            # lower triangle
            pts_l = [
                (x, self.height),
                (x + self.pipe_width // 2, gap_y + self.pipe_gap),
                (x + self.pipe_width, self.height),
            ]
            draw.polygon(pts_l, fill=self.pipe_color_lower)
        return np.array(img)