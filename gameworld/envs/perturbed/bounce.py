import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw

from gameworld.envs.base.bounce import Bounce as BaseBounce


class Bounce(BaseBounce):
    """Bounce with exact baseline when perturb=None, and mid-episode color/shape perturbations."""

    def __init__(
        self,
        player_x=135,
        opponent_x=15,
        perturb=None,
        perturb_step=5000,
        **kwargs,
    ):
        assert perturb in (None, "None", "color", "shape"), \
            "perturb must be None, 'color', or 'shape'"
        self.perturb = None if perturb in (None, "None") else perturb
        self.perturb_step = perturb_step
        self.num_steps = 0
        super().__init__(player_x=player_x, opponent_x=opponent_x, **kwargs)
    

    def step(self, action):
        obs, reward, done, trunc, info = super().step(action)
        self.num_steps += 1
        if self.perturb and self.num_steps == self.perturb_step:
            self._apply_perturbation()
        return obs, reward, done, trunc, info

    def _apply_perturbation(self):
        if self.perturb == "color":
            self.bg_color = (32, 32, 32)
            self.player_color = (0, 128, 255)
            self.opponent_color = (255, 200, 0)
            self.ball_color = (0, 255, 255)
        # shape-only: handled in _get_obs drawing

    def _get_obs(self):
        # before shape-perturbation, delegate to base for pixel-perfect match
        if not (self.perturb == "shape" and self.num_steps >= self.perturb_step):
            return super()._get_obs()

        # custom shape: triangle paddles + circle ball
        # start from uniform background + walls
        obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        obs[:, :, :] = np.array(self.bg_color, dtype=np.uint8)
        obs[: self.wall_width, :, :] = self.wall_color
        obs[-self.wall_width :, :, :] = self.wall_color

        img = Image.fromarray(obs)
        draw = ImageDraw.Draw(img)
        # player triangle
        p0 = (self.player_x, self.player_y)
        p1 = (self.player_x, self.player_y + self.paddle_height)
        p2 = (self.player_x + self.paddle_width, self.player_y + self.paddle_height // 2)
        draw.polygon([p0, p1, p2], fill=self.player_color)
        # opponent triangle
        o0 = (self.opponent_x + self.paddle_width, self.opponent_y)
        o1 = (self.opponent_x + self.paddle_width, self.opponent_y + self.paddle_height)
        o2 = (self.opponent_x, self.opponent_y + self.paddle_height // 2)
        draw.polygon([o0, o1, o2], fill=self.opponent_color)
        # ball circle
        bx0, by0 = self.ball_x, self.ball_y
        bx1, by1 = bx0 + self.ball_size, by0 + self.ball_size
        draw.ellipse([bx0, by0, bx1, by1], fill=self.ball_color)
        return np.array(img)