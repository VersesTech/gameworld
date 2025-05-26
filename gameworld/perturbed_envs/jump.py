import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw

from gameworld.envs.base import GameworldEnv


class Jump(GameworldEnv):
    """ Player needs to jump or dodge incoming obstacles.
    """

    def __init__(
        self,
        perturb=None,
        perturb_step=5000,
        **kwargs,
    ):
        super().__init__()
        assert perturb in (None, "None", "color", "shape"), (
            "perturb must be None, 'color', or 'shape'"
        )
        self.perturb = None if perturb in (None, "None") else perturb
        self.perturb_step = perturb_step
        self.num_steps = 0

        # screen
        self.width = 160
        self.height = 210
        self.ground_y = 180

        # runner
        self.orig_runner_w = 10
        self.orig_runner_h = 20
        self.runner_width = self.orig_runner_w
        self.runner_height = self.orig_runner_h
        self.runner_x = 20
        self.runner_y = self.ground_y - self.runner_height
        self.gravity = 3
        self.jump_speed = -20
        self.runner_vel_y = 0
        self.is_jumping = False

        # obstacles
        self.obstacles = []    # list of dict {x,y,width,height}
        self.obstacle_spawn_prob = 0.05

        # colors
        self.bg_color = (50, 50, 100)
        self.ground_color = (150, 150, 255)
        self.ceiling_color = (150, 150, 255)
        self.runner_color = (255, 255, 0)
        self.obstacle_color = (255, 0, 0)

        # spaces
        self.action_space = spaces.Discrete(2)  # 0: nothing, 1: jump
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )

        self.reset()

    def reset(self):
        self.runner_y = self.ground_y - self.runner_height
        self.runner_vel_y = 0
        self.is_jumping = False
        self.obstacles = []
        return self._get_obs(), {}

    def step(self, action):
        reward = 0
        done = False

        # jump
        if action == 1 and not self.is_jumping:
            self.runner_vel_y = self.jump_speed
            self.is_jumping = True

        # physics
        self.runner_y += self.runner_vel_y
        self.runner_vel_y += self.gravity

        # ground collision
        if self.runner_y >= self.ground_y - self.runner_height:
            self.runner_y = self.ground_y - self.runner_height
            self.runner_vel_y = 0
            self.is_jumping = False

        # spawn obstacle
        if np.random.rand() < self.obstacle_spawn_prob and (
            not self.obstacles or self.obstacles[-1]["x"] < self.width - 50
        ):
            h = 20
            w = 15
            floating = np.random.rand() < 0.3
            y = self.ground_y - h if not floating else self.ground_y - h - np.random.randint(40, 70)
            self.obstacles.append({"x": self.width, "y": y, "width": w, "height": h})

        # move and cull
        for ob in self.obstacles:
            ob["x"] -= 3
        self.obstacles = [o for o in self.obstacles if o["x"] + o["width"] > 0]

        # collision detection
        for ob in self.obstacles:
            if (
                self.runner_x < ob["x"] + ob["width"] and
                self.runner_x + self.runner_width > ob["x"] and
                self.runner_y < ob["y"] + ob["height"] and
                self.runner_y + self.runner_height > ob["y"]
            ):
                reward = -1
                self.runner_y = self.height + 1
                done = True
                break

        # perturb timing
        self.num_steps += 1
        if self.perturb and self.num_steps == self.perturb_step:
            self._apply_perturbation()

        return self._get_obs(), reward, done, False, {}

    def _apply_perturbation(self):
        print(f"Applying perturbation: {self.perturb}")
        if self.perturb == "color":
            # swap to high-contrast palette
            self.bg_color = (32, 32, 32)
            self.ground_color = (64, 64, 64)
            self.ceiling_color = (64, 64, 64)
            self.runner_color = (0, 128, 255)
            self.obstacle_color = (255, 200, 0)
        elif self.perturb == "shape":
            # scale shapes slightly
            scale = 1.2
            self.runner_width = int(self.orig_runner_w * scale)
            self.runner_height = int(self.orig_runner_h * scale)
            # obstacles become circles

    def _get_obs(self):
        img = Image.new("RGB", (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(img)

        # ground and ceiling
        draw.rectangle([0, self.ground_y, self.width, self.height], fill=self.ground_color)
        draw.rectangle([0, 0, self.width, self.ground_y - 160], fill=self.ceiling_color)

        # draw runner
        rx0 = self.runner_x
        ry0 = int(self.runner_y)
        rx1 = rx0 + self.runner_width
        ry1 = ry0 + self.runner_height
        if self.perturb == "shape" and self.num_steps >= self.perturb_step:
            # runner as triangle
            pts = [(rx0, ry1), (rx0 + self.runner_width/2, ry0), (rx1, ry1)]
            draw.polygon(pts, fill=self.runner_color)
        else:
            draw.rectangle([rx0, ry0, rx1, ry1], fill=self.runner_color)

        # draw obstacles
        for ob in self.obstacles:
            x, y, w, h = ob["x"], ob["y"], ob["width"], ob["height"]
            if self.perturb == "shape" and self.num_steps >= self.perturb_step:
                # circle obstacle
                draw.ellipse([x, y, x+w, y+h], fill=self.obstacle_color)
            else:
                draw.rectangle([x, y, x+w, y+h], fill=self.obstacle_color)

        return np.array(img)


