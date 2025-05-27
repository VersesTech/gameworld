import numpy as np
from PIL import Image, ImageDraw
from gameworld.envs.base.drive import Drive as BaseDriveEnv


class Drive(BaseDriveEnv):
    """Drive with mid‐episode color or shape perturbations."""

    def __init__(self, perturb=None, perturb_step=5000, **kwargs):
        assert perturb in (
            None,
            "None",
            "color",
            "shape",
        ), "perturb must be None, 'color', or 'shape'"
        # normalize
        self.perturb = None if perturb in (None, "None") else perturb
        self.perturb_step = perturb_step
        # track steps
        self.num_steps = 0

        # call base and then stash the “original” palette
        super().__init__(**kwargs)
        self._orig_bg_color = (150, 150, 255)
        self._orig_road_color = (50, 50, 100)
        self._orig_player_color = (255, 255, 0)
        # for opponents we’ll just recolor them all to one solid color on perturb
        self._orig_opponent_color = None  # not used by base draw

        # set current palette to original
        self.bg_color = self._orig_bg_color
        self.road_color = self._orig_road_color
        self.player_color = self._orig_player_color
        self.opponent_color = (255, 0, 0)

    def step(self, action):
        obs, reward, done, trunc, info = super().step(action)
        self.num_steps += 1
        if self.perturb and self.num_steps == self.perturb_step:
            self._apply_perturbation()
        return obs, reward, done, trunc, info

    def _apply_perturbation(self):
        if self.perturb == "color":
            # choose a completely new colour scheme
            self.bg_color = (32, 32, 32)  # darker “sky”
            self.road_color = (100, 100, 100)  # grey road
            self.player_color = (0, 255, 128)  # mint‐green player car
            self.opponent_color = (255, 200, 0)  # golden opponent cars
        # if shape perturb, we do nothing here—shape logic lives in _get_obs()

    def _get_obs(self):
        # shape‐only override
        if self.perturb == "shape" and self.num_steps >= self.perturb_step:
            # blank canvas & fill sky
            obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            obs[:, :, :] = np.array(self.bg_color, dtype=np.uint8)
            # draw road
            road_left = self.width // 2 - self.road_width // 2
            road_right = self.width // 2 + self.road_width // 2
            obs[:, road_left:road_right, :] = np.array(self.road_color, dtype=np.uint8)

            img = Image.fromarray(obs)
            draw = ImageDraw.Draw(img)

            # draw each opponent car as a circle
            for opp in self.opponents:
                x, y = int(opp["x"]), int(opp["y"])
                r = max(self.car_width, self.car_height) // 2
                cx, cy = x + self.car_width // 2, y + self.car_height // 2
                bbox = [cx - r, cy - r, cx + r, cy + r]
                draw.ellipse(bbox, fill=tuple(opp["color"]))

            # draw the player car as a triangle
            px, py = int(self.player_x), int(self.player_y)
            p_w, p_h = self.car_width, self.car_height
            # triangle points: top-center, bottom-left, bottom-right
            p0 = (px + p_w / 2, py)
            p1 = (px, py + p_h)
            p2 = (px + p_w, py + p_h)
            draw.polygon([p0, p1, p2], fill=self.player_color)

            return np.array(img)

        # color‐only override
        if self.perturb == "color" and self.num_steps >= self.perturb_step:
            # blank canvas & fill sky
            obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            obs[:, :, :] = np.array(self.bg_color, dtype=np.uint8)
            # draw road
            road_left = self.width // 2 - self.road_width // 2
            road_right = self.width // 2 + self.road_width // 2
            obs[:, road_left:road_right, :] = np.array(self.road_color, dtype=np.uint8)

            # draw opponent cars as rectangles in the new opponent_color
            for opp in self.opponents:
                x, y = int(opp["x"]), int(opp["y"])
                if 0 <= y < self.height:
                    obs[y : y + self.car_height, x : x + self.car_width] = np.array(
                        self.opponent_color, dtype=np.uint8
                    )

            # draw the player car as a rectangle in the new player_color
            if 0 <= self.player_y < self.height:
                obs[
                    self.player_y : self.player_y + self.car_height,
                    int(self.player_x) : int(self.player_x) + self.car_width,
                ] = np.array(self.player_color, dtype=np.uint8)

            return obs

        # otherwise default rendering
        return super()._get_obs()
