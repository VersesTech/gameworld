import numpy as np
from PIL import Image, ImageDraw

from gameworld.envs.base.jump import Jump as BaseJump


class Jump(BaseJump):
    """Jump with exact baseline when perturb=None, and mid‐episode color/shape perturbations."""

    def __init__(self, perturb=None, perturb_step=5000, **kwargs):
        assert perturb in (
            None,
            "None",
            "color",
            "shape",
        ), "perturb must be None, 'color', or 'shape'"
        # perturb config
        self.perturb = None if perturb in (None, "None") else perturb
        self.perturb_step = perturb_step
        self.num_steps = 0

        # initialize base (sets up dims, runner, obstacles)
        super().__init__(**kwargs)

        # stash baseline palette
        self.bg_color = (50, 50, 100)
        self.ground_color = (150, 150, 255)
        self.ceiling_color = (150, 150, 255)
        self.runner_color = (255, 255, 0)
        self.obstacle_color = (255, 0, 0)

    def step(self, action):
        obs, reward, done, trunc, info = super().step(action)
        self.num_steps += 1
        if self.perturb and self.num_steps == self.perturb_step:
            self._apply_perturbation()
        return obs, reward, done, trunc, info

    def _apply_perturbation(self):
        if self.perturb == "color":
            # switch to high‐contrast palette
            self.bg_color = (32, 32, 32)
            self.ground_color = (100, 100, 100)
            self.ceiling_color = (100, 100, 100)
            self.runner_color = (0, 128, 255)
            self.obstacle_color = (255, 64, 128)
        # shape-only: handled in drawing

    def _get_obs(self):
        # shape‐only override
        if self.perturb == "shape" and self.num_steps >= self.perturb_step:
            return self._draw_shape_obs()
        # color‐only override
        if self.perturb == "color" and self.num_steps >= self.perturb_step:
            return self._draw_color_obs()
        # baseline
        return super()._get_obs()

    def _draw_color_obs(self):
        # numpy draw with updated colors
        obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        obs[:, :, :] = np.array(self.bg_color, dtype=np.uint8)

        # ground
        obs[self.ground_y :, :] = np.array(self.ground_color, dtype=np.uint8)
        # ceiling
        obs[: self.top_margin, :] = np.array(self.ceiling_color, dtype=np.uint8)

        # runner
        ry, rx = int(self.runner_y), int(self.runner_x)
        h, w = self.runner_height, self.runner_width
        obs[ry : ry + h, rx : rx + w] = np.array(self.runner_color, dtype=np.uint8)

        # obstacles
        for ob in self.obstacles:
            ox, oy = int(ob["x"]), int(ob["y"])
            oh, ow = ob["height"], ob["width"]
            y0, y1 = oy, oy + oh
            x0, x1 = ox, ox + ow
            obs[y0:y1, x0:x1] = np.array(self.obstacle_color, dtype=np.uint8)

        return obs

    def _draw_shape_obs(self):
        # PIL draw for new shapes
        img = Image.new("RGB", (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(img)

        # ground
        draw.rectangle(
            [0, self.ground_y, self.width, self.height], fill=self.ground_color
        )
        # ceiling
        draw.rectangle([0, 0, self.width, self.top_margin], fill=self.ceiling_color)

        # runner → circle
        ry, rx = self.runner_y, self.runner_x
        h, w = self.runner_height, self.runner_width
        draw.ellipse([rx, ry, rx + w, ry + h], fill=self.runner_color)

        # obstacles → triangles pointing down
        for ob in self.obstacles:
            ox, oy = ob["x"], ob["y"]
            ow, oh = ob["width"], ob["height"]
            pts = [(ox, oy), (ox + ow, oy), (ox + ow / 2, oy + oh)]
            draw.polygon(pts, fill=self.obstacle_color)

        return np.array(img)
