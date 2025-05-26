import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw

from gameworld.envs.base import GameworldEnv


class Aviate(GameworldEnv):
    """ Based on the FlappyBird game
    
        Player needs to flap the birds wings to navigate through the pipes.
    """

    def __init__(
        self,
        perturb=None,
        perturb_step=5000,
        **kwargs,
    ):
        super().__init__()
        assert perturb in (
            None,
            "None",
            "color",
            "shape",
        ), "perturb must be None, 'color', or 'shape'"
        self.perturb = None if perturb in (None, "None") else perturb
        self.perturb_step = perturb_step
        self.num_steps = 0

        # screen
        self.width = 160
        self.height = 210

        # bird
        self.bird_x = 30
        self.bird_y = 100
        self.orig_bird_radius = 5
        self.bird_radius = self.orig_bird_radius
        self.bird_vel_y = 0
        self.gravity = 1
        self.jump_speed = -10

        # pipes
        self.pipe_gap = 100
        self.pipe_width = 20
        self.pipe_speed = 2
        self.pipe_spawn_prob = 0.03
        self.pipes = []  # list of dict {x, gap_y}

        # default colors
        self.bg_color = (50, 50, 100)
        self.bird_color = (255, 255, 0)
        self.pipe_color_upper = (30, 255, 0)
        self.pipe_color_lower = (0, 255, 30)

        # spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )

        # init
        self.reset()

    def reset(self):
        self.bird_y = 160
        self.bird_vel_y = 0
        self.pipes = []
        gap_y = np.random.randint(40, self.height - 40 - self.pipe_gap)
        self.pipes.append({"x": self.width, "gap_y": gap_y})
        return self._get_obs(), {}

    def step(self, action):
        # physics
        if action == 1 and self.bird_vel_y > 2:
            self.bird_vel_y = self.jump_speed
        self.bird_y += self.bird_vel_y
        self.bird_vel_y += self.gravity

        # check bounds
        reward = 0
        done = False
        if self.bird_y < 0 or self.bird_y > self.height:
            reward = -1
            done = True

        # spawn
        if np.random.rand() < self.pipe_spawn_prob and (
            not self.pipes or self.pipes[-1]["x"] < self.width - 80
        ):
            gap_y = np.random.randint(40, self.height - 40 - self.pipe_gap)
            self.pipes.append({"x": self.width, "gap_y": gap_y})

        # move pipes
        for pipe in self.pipes:
            pipe["x"] -= self.pipe_speed
        self.pipes = [p for p in self.pipes if p["x"] + self.pipe_width > 0]

        # collision
        for pipe in self.pipes:
            x, gap_y = pipe["x"], pipe["gap_y"]
            if (
                self.bird_x + self.bird_radius > x
                and self.bird_x - self.bird_radius < x + self.pipe_width
            ):
                if not (gap_y < self.bird_y < gap_y + self.pipe_gap):
                    reward = -1
                    done = True
                    self.bird_y = self.height + 1
                    break

        # perturbation timing
        self.num_steps += 1
        if self.perturb and self.num_steps == self.perturb_step:
            self._apply_perturbation()

        return self._get_obs(), reward, done, False, {}

    def _apply_perturbation(self):
        print(f"Applying perturbation: {self.perturb}")
        if self.perturb == "color":
            # swap to high contrast palette
            self.bg_color = (32, 32, 32)
            self.bird_color = (255, 64, 128)
            self.pipe_color_upper = (255, 200, 0)
            self.pipe_color_lower = (0, 255, 255)
        elif self.perturb == "shape":
            # change shapes: bird circle->square, pipes rectangles->triangles
            self.bird_radius = int(self.orig_bird_radius * 1.2)

    def _get_obs(self):
        img = Image.new("RGB", (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(img)

        # draw bird
        x0, y0 = self.bird_x - self.bird_radius, int(self.bird_y - self.bird_radius)
        x1, y1 = self.bird_x + self.bird_radius, int(self.bird_y + self.bird_radius)
        if self.perturb == "shape" and self.num_steps >= self.perturb_step:
            # square bird
            draw.rectangle([x0, y0, x1, y1], fill=self.bird_color)
        else:
            # circle bird
            draw.ellipse([x0, y0, x1, y1], fill=self.bird_color)

        # draw pipes
        for pipe in self.pipes:
            x = pipe["x"]
            gap_y = pipe["gap_y"]
            # upper
            if self.perturb == "shape" and self.num_steps >= self.perturb_step:
                # upper pipe as triangle
                pts = [
                    (x, 0),
                    (x + self.pipe_width // 2, gap_y),
                    (x + self.pipe_width, 0),
                ]
                draw.polygon(pts, fill=self.pipe_color_upper)
            else:
                draw.rectangle(
                    [x, 0, x + self.pipe_width, gap_y], fill=self.pipe_color_upper
                )
            # lower
            if self.perturb == "shape" and self.num_steps >= self.perturb_step:
                # lower pipe triangle
                pts = [
                    (x, self.height),
                    (x + self.pipe_width // 2, gap_y + self.pipe_gap),
                    (x + self.pipe_width, self.height),
                ]
                draw.polygon(pts, fill=self.pipe_color_lower)
            else:
                draw.rectangle(
                    [x, gap_y + self.pipe_gap, self.width, self.height],
                    fill=self.pipe_color_lower,
                )

        return np.array(img)
