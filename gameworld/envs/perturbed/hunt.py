import numpy as np
from PIL import Image, ImageDraw

from gameworld.envs.base.hunt import Hunt as BaseHunt


class Hunt(BaseHunt):
    """Hunt with exact baseline when perturb=None, and mid‐episode color/shape perturbations."""

    def __init__(self, max_objects=3, perturb=None, perturb_step=5000, **kwargs):
        assert perturb in (None, "None", "color", "shape"), \
            "perturb must be None, 'color', or 'shape'"
        # normalize
        self.perturb = None if perturb in (None, "None") else perturb
        self.perturb_step = perturb_step
        self.num_steps = 0

        # initialize base
        super().__init__(max_objects=max_objects, **kwargs)

        # stash original sizes
        self.orig_player_w = self.player_width
        self.orig_player_h = self.player_height
        self.orig_item_size = self.item_size

        # override base colors for easy mutation
        self.bg_color        = (50,  50, 100)
        self.lane_color      = (255,255,255)
        self.player_color    = (255,255,  0)
        self.item_color      = (  0,255,  0)
        self.obstacle_color  = (255,  0,  0)

    def step(self, action):
        obs, reward, done, trunc, info = super().step(action)
        self.num_steps += 1
        if self.perturb and self.num_steps == self.perturb_step:
            self._apply_perturbation()
        return obs, reward, done, trunc, info

    def _apply_perturbation(self):
        if self.perturb == "color":
            # switch to high-contrast palette
            self.bg_color        = (32,  32,  32)
            self.lane_color      = (200,200,200)
            self.player_color    = (  0,128,255)
            self.item_color      = (255, 64,128)
            self.obstacle_color  = (255,200,  0)
        elif self.perturb == "shape":
            # enlarge player & items
            scale = 1.5
            self.player_width  = int(self.orig_player_w * scale)
            self.player_height = int(self.orig_player_h * scale)
            self.item_size     = int(self.orig_item_size * 2)
            # clamp position
            self.player_x = min(self.player_x, self.width - self.player_width)
            self.player_y = min(self.player_y, self.height - self.bottom_margin - self.player_height)

    def _get_obs(self):
        # shape perturb takes priority
        if self.perturb == "shape" and self.num_steps >= self.perturb_step:
            return self._draw_shape_obs()
        # color-only after perturb
        if self.perturb == "color" and self.num_steps >= self.perturb_step:
            return self._draw_color_obs()
        # otherwise exact baseline
        return super()._get_obs()

    def _draw_color_obs(self):
        """Re-draw using numpy + updated colors, same shapes as baseline."""
        obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        obs[:, :, :] = np.array(self.bg_color, dtype=np.uint8)

        # lanes
        for i in range(self.lane_count + 1):
            y = self.top_margin + i * self.lane_height - 1
            obs[y:y+3, :, :] = np.array(self.lane_color, dtype=np.uint8)

        # player (rectangle)
        y0, y1 = self.player_y, self.player_y + self.player_height
        x0, x1 = self.player_x, self.player_x + self.player_width
        obs[y0:y1, x0:x1] = np.array(self.player_color, dtype=np.uint8)

        # items (rectangles)
        for x, y, dx in self.items:
            x0, y0 = int(x), int(y)
            obs[y0:y0+self.item_size, x0:x0+self.item_size] = np.array(self.item_color, dtype=np.uint8)

        # obstacles (rectangles)
        for x, y, dx in self.obstacles:
            x0, y0 = int(x), int(y)
            obs[y0:y0+self.item_size, x0:x0+self.item_size] = np.array(self.obstacle_color, dtype=np.uint8)

        return obs

    def _draw_shape_obs(self):
        """Re-draw with new shapes: player→circle, items→circles, obstacles→triangles."""
        img = Image.new("RGB", (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(img)

        # lanes
        for i in range(self.lane_count + 1):
            y = self.top_margin + i * self.lane_height - 1
            draw.rectangle([0, y, self.width, y+2], fill=self.lane_color)

        # player → circle
        px0 = self.player_x
        py0 = self.player_y
        px1 = px0 + self.player_width
        py1 = py0 + self.player_height
        draw.ellipse([px0, py0, px1, py1], fill=self.player_color)

        # items → circles
        for x, y, dx in self.items:
            x0, y0 = int(x), int(y)
            x1, y1 = x0 + self.item_size, y0 + self.item_size
            draw.ellipse([x0, y0, x1, y1], fill=self.item_color)

        # obstacles → triangles (pointing in direction of motion)
        for x, y, dx in self.obstacles:
            x0, y0 = int(x), int(y)
            s = self.item_size
            if dx > 0:
                pts = [(x0, y0), (x0, y0+s), (x0+s, y0+s//2)]
            else:
                pts = [(x0+s, y0), (x0+s, y0+s), (x0, y0+s//2)]
            draw.polygon(pts, fill=self.obstacle_color)

        return np.array(img)
