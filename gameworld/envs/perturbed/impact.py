import numpy as np
from PIL import Image, ImageDraw

from gameworld.envs.base.impact import Impact as BaseImpact


class Impact(BaseImpact):
    """Impact with exact baseline when perturb=None, and mid-episode color/shape perturbations."""

    def __init__(self, paddle_y=190, perturb=None, perturb_step=5000, **kwargs):
        assert perturb in (None, "None", "color", "shape"), \
            "perturb must be None, 'color', or 'shape'"
        # normalize perturb flag
        self.perturb = None if perturb in (None, "None") else perturb
        self.perturb_step = perturb_step
        self.num_steps = 0

        # stash original geometry & palette
        self.orig_paddle_w   = 30
        self.orig_paddle_h   = 8
        self.orig_ball_size  = 4
        self.orig_brick_w    =  self.width // self.brick_cols if hasattr(self, "brick_cols") else None
        self.orig_brick_h    = 10
        self.bg_color        = (50,  50, 100)
        self.paddle_color    = (255,255,  0)
        self.ball_color      = (255,  0,  0)
        self.brick_color     = (  0,255,  0)

        # initialize base
        super().__init__(paddle_y=paddle_y, **kwargs)

    def step(self, action):
        obs, reward, done, trunc, info = super().step(action)
        self.num_steps += 1
        if self.perturb and self.num_steps == self.perturb_step:
            self._apply_perturbation()
        return obs, reward, done, trunc, info

    def _apply_perturbation(self):
        if self.perturb == "color":
            # swap to high-contrast palette
            self.bg_color     = (32,  32,  32)
            self.paddle_color = (  0,128,255)
            self.ball_color   = (255, 64,128)
            self.brick_color  = (255,200,  0)
        # shape-only: no attribute change—handled in drawing

    def _get_obs(self):
        # shape-only after perturb
        if self.perturb == "shape" and self.num_steps >= self.perturb_step:
            return self._draw_shape_obs()
        # color-only after perturb
        if self.perturb == "color" and self.num_steps >= self.perturb_step:
            return self._draw_color_obs()
        # otherwise baseline
        return super()._get_obs()

    def _draw_color_obs(self):
        # numpy-based full redraw with new palette
        obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        obs[:, :, :] = np.array(self.bg_color, dtype=np.uint8)

        # paddle
        y0 = self.paddle_y
        y1 = y0 + self.paddle_height
        x0 = self.paddle_x
        x1 = x0 + self.paddle_width
        obs[y0:y1, x0:x1] = np.array(self.paddle_color, dtype=np.uint8)

        # ball
        if self.ball_y < self.height:
            by0 = int(self.ball_y)
            by1 = by0 + self.ball_size
            bx0 = int(self.ball_x)
            bx1 = bx0 + self.ball_size
            obs[by0:by1, bx0:bx1] = np.array(self.ball_color, dtype=np.uint8)

        # bricks
        for row in range(self.brick_rows):
            for col in range(self.brick_cols):
                if self.bricks[row, col]:
                    bx = col * self.brick_width
                    by = row * self.brick_height + 20
                    obs[by:by+self.brick_height, bx:bx+self.brick_width] = np.array(self.brick_color, dtype=np.uint8)

        return obs

    def _draw_shape_obs(self):
        # PIL-based redraw with new shapes
        img = Image.new("RGB", (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(img)

        # paddle → ellipse
        px0 = self.paddle_x
        py0 = self.paddle_y
        px1 = px0 + self.paddle_width
        py1 = py0 + self.paddle_height
        draw.ellipse([px0, py0, px1, py1], fill=self.paddle_color)

        # ball → circle (bigger for visibility)
        bx0 = int(self.ball_x)
        by0 = int(self.ball_y)
        bx1 = bx0 + self.ball_size
        by1 = by0 + self.ball_size
        draw.ellipse([bx0, by0, bx1, by1], fill=self.ball_color)

        # bricks → triangles
        for row in range(self.brick_rows):
            for col in range(self.brick_cols):
                if self.bricks[row, col]:
                    bx = col * self.brick_width
                    by = row * self.brick_height + 20
                    # downward-pointing triangle
                    pts = [
                        (bx,                by + self.brick_height),
                        (bx + self.brick_width/2, by),
                        (bx + self.brick_width,   by + self.brick_height),
                    ]
                    draw.polygon(pts, fill=self.brick_color)

        return np.array(img)
