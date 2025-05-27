import numpy as np
from PIL import Image, ImageDraw

from gameworld.envs.base.fruits import Fruits as BaseFruits


class Fruits(BaseFruits):
    """Fruits with exact baseline when perturb=None, and mid-episode color/shape perturbations."""

    def __init__(self, perturb=None, perturb_step=5000, **kwargs):
        assert perturb in (None, "None", "color", "shape"), \
            "perturb must be None, 'color', or 'shape'"
        # normalize
        self.perturb = None if perturb in (None, "None") else perturb
        self.perturb_step = perturb_step
        self.num_steps = 0

        # init base env (sets up widths, colors, etc.)
        super().__init__(**kwargs)

        # stash original palette & geometry
        self.bg_color        = ( 50,  50, 100)
        self.player_color    = (255, 255,   0)
        self.basket_color    = (255, 255,   0)
        self.ground_color    = (150, 150, 255)
        self.orig_fruit_colors = [tuple(c) for c in self.fruit_colors]
        self.orig_rock_color   = tuple(self.rock_color)

        # convert to tuples for easy re-assignment
        self.fruit_colors = [tuple(c) for c in self.fruit_colors]
        self.rock_color   = tuple(self.rock_color)

    def step(self, action):
        obs, reward, done, trunc, info = super().step(action)
        self.num_steps += 1
        if self.perturb and self.num_steps == self.perturb_step:
            self._apply_perturbation()
        return obs, reward, done, trunc, info

    def _apply_perturbation(self):
        if self.perturb == "color":
            # high-contrast palette
            self.bg_color        = (32,  32,  32)
            self.player_color    = (  0,128,255)
            self.basket_color    = (  0,128,255)
            # cycle three vivid fruit colors
            self.fruit_colors    = [
                (255,200,  0),
                (  0,255,255),
                (255, 64,128),
            ]
            self.rock_color      = (255,200,  0)
            self.ground_color    = (100,100,100)
        # shape-only: no attribute changes—handled in custom drawing

    def _get_obs(self):
        # shape perturb takes priority
        if self.perturb == "shape" and self.num_steps >= self.perturb_step:
            return self._draw_shape_obs()
        # color-only after perturb
        if self.perturb == "color" and self.num_steps >= self.perturb_step:
            return self._draw_color_obs()
        # otherwise baseline
        return super()._get_obs()

    def _draw_color_obs(self):
        # pure numpy draw with updated colors
        obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        obs[:, :, :] = np.array(self.bg_color, dtype=np.uint8)

        # player
        obs[
            self.player_y : self.player_y + self.player_height,
            self.player_x : self.player_x + self.player_width
        ] = np.array(self.player_color, dtype=np.uint8)

        # basket
        bx = self.player_x - self.basket_width//2 + self.player_width//2
        by = self.player_y - self.basket_height
        obs[by : self.player_y, bx : bx + self.basket_width] = np.array(self.basket_color, dtype=np.uint8)

        # falling objects
        for x, y, is_rock, color_idx, _ in self.falling_objects:
            c = self.rock_color if is_rock else self.fruit_colors[color_idx]
            x0, y0 = int(x), int(y)
            x1, y1 = x0 + self.fruit_size, y0 + self.fruit_size
            obs[y0:y1, x0:x1] = np.array(c, dtype=np.uint8)

        # ground
        obs[self.height - self.ground_height :, :, :] = np.array(self.ground_color, dtype=np.uint8)

        return obs

    def _draw_shape_obs(self):
        # PIL draw for circles & triangles
        obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        obs[:, :, :] = np.array(self.bg_color, dtype=np.uint8)
        # draw ground
        obs[self.height - self.ground_height :, :, :] = np.array(self.ground_color, dtype=np.uint8)

        img = Image.fromarray(obs)
        draw = ImageDraw.Draw(img)

        # basket as upward-pointing triangle
        bx = self.player_x - self.basket_width//2 + self.player_width//2
        by = self.player_y - self.basket_height
        pts_b = [
            (bx,                 by + self.basket_height),
            (bx + self.basket_width, by + self.basket_height),
            (bx + self.basket_width/2, by)
        ]
        draw.polygon(pts_b, fill=self.basket_color)

        # player as rectangle (unchanged)
        draw.rectangle([
            (self.player_x, self.player_y),
            (self.player_x + self.player_width, self.player_y + self.player_height)
        ], fill=self.player_color)

        # falling objects: fruits → circles, rocks → downward-pointing triangles
        for x, y, is_rock, color_idx, _ in self.falling_objects:
            c = self.rock_color if is_rock else self.fruit_colors[color_idx]
            x0, y0 = x, y
            x1, y1 = x0 + self.fruit_size, y0 + self.fruit_size
            if is_rock:
                # triangle pointing down
                pts_r = [
                    (x0,    y0),
                    (x1,    y0),
                    (x0 + self.fruit_size/2, y1)
                ]
                draw.polygon(pts_r, fill=c)
            else:
                # circle fruit
                draw.ellipse([x0, y0, x1, y1], fill=c)

        return np.array(img)
