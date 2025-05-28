import numpy as np
from PIL import Image, ImageDraw

from gameworld.envs.base.gold import Gold as BaseGold


class Gold(BaseGold):
    """Gold with exact baseline when perturb=None, and mid-episode color/shape perturbations."""

    def __init__(
        self, max_coins=3, max_obstacles=3, perturb=None, perturb_step=5000, **kwargs
    ):
        assert perturb in (None, "None", "color", "shape"), (
            "perturb must be None, 'color', or 'shape'"
        )
        self.perturb = None if perturb in (None, "None") else perturb
        self.perturb_step = perturb_step
        self.num_steps = 0

        # initialize base game with coin/obstacle limits
        super().__init__(max_coins=max_coins, max_obstacles=max_obstacles, **kwargs)

        # stash original palette
        self.bg_color = (50, 50, 100)
        self.player_color = (255, 255, 0)
        self.item_color = (0, 255, 0)
        self.obstacle_color = (255, 0, 0)

    def step(self, action):
        obs, reward, done, trunc, info = super().step(action)
        self.num_steps += 1
        if self.perturb and self.num_steps == self.perturb_step:
            self._apply_perturbation()
        return obs, reward, done, trunc, info

    def _apply_perturbation(self):
        if self.perturb == "color":
            # switch to high-contrast colors
            self.bg_color = (32, 32, 32)
            self.player_color = (0, 128, 255)
            self.item_color = (255, 200, 0)
            self.obstacle_color = (255, 64, 128)
        # shape-only handled in drawing

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
        # numpy-based draw with updated palette
        obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        obs[:, :, :] = np.array(self.bg_color, dtype=np.uint8)

        # player
        obs[
            self.player_y : self.player_y + self.player_height,
            self.player_x : self.player_x + self.player_width,
        ] = np.array(self.player_color, dtype=np.uint8)

        # items (coins)
        for x, y, is_obs in self.items + self.obstacles:
            # items list holds only fruits; obstacles in separate list
            pass  # not used here

        for x, y, is_obs in self.items:
            x0, y0 = int(x), int(y)
            x1, y1 = x0 + self.item_size, y0 + self.item_size
            obs[y0:y1, x0:x1] = np.array(self.item_color, dtype=np.uint8)

        # obstacles
        for x, y, _ in self.obstacles:
            x0, y0 = int(x), int(y)
            x1, y1 = x0 + self.item_size, y0 + self.item_size
            obs[y0:y1, x0:x1] = np.array(self.obstacle_color, dtype=np.uint8)

        return obs

    def _draw_shape_obs(self):
        # PIL-based draw with new shapes
        obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        obs[:, :, :] = np.array(self.bg_color, dtype=np.uint8)
        img = Image.fromarray(obs)
        draw = ImageDraw.Draw(img)

        # player → circle
        px, py = self.player_x, self.player_y
        pw, ph = self.player_width, self.player_height
        draw.ellipse([px, py, px + pw, py + ph], fill=self.player_color)

        # items → circles (coins)
        for x, y, _ in self.items:
            x0, y0 = int(x), int(y)
            x1, y1 = x0 + self.item_size, y0 + self.item_size
            draw.ellipse([x0, y0, x1, y1], fill=self.item_color)

        # obstacles → triangles pointing in movement direction
        for x, y, speed in self.obstacles:
            x0, y0 = int(x), int(y)
            size = self.item_size
            if speed > 0:
                pts = [(x0, y0), (x0, y0 + size), (x0 + size, y0 + size // 2)]
            else:
                pts = [(x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size // 2)]
            draw.polygon(pts, fill=self.obstacle_color)

        return np.array(img)
