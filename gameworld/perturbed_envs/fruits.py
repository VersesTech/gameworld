import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw

from gameworld.envs.base import GameworldEnv


class Fruits(GameworldEnv):
    """ Player needs to catch fruits falling from the sky, why avoiding rocks.
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

        # Screen dimensions
        self.width = 160
        self.height = 210

        # Player properties
        self.orig_player_w = 16
        self.orig_player_h = 30
        self.player_width = self.orig_player_w
        self.player_height = self.orig_player_h
        self.basket_width = 24
        self.basket_height = 16
        self.ground_height = 16
        self.player_speed = 8
        self.player_x = self.width // 2 - self.player_width // 2
        self.player_y = self.height - self.ground_height - self.basket_height

        # Falling object properties
        self.orig_fruit_size = 12
        self.orig_rock_size = 10
        self.fruit_size = self.orig_fruit_size
        self.rock_size = self.orig_rock_size
        self.fruit_colors = [
            (255, 0, 0),   # apple
            (128, 0, 128), # grape
            (0, 255, 0),   # pear
        ]
        self.rock_color = (180, 180, 180)
        self.rock_probability = 0.25
        self.max_objects = 6
        self.falling_objects = []  # [x, y, is_rock, color_idx, speed]

        # Colors
        self.bg_color = (50, 50, 100)
        self.player_color = (255, 255, 0)
        self.basket_color = (255, 255, 0)

        # Action and observation spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )

        self.reset()

    def reset(self):
        self.player_x = self.width // 2 - self.player_width // 2
        self.falling_objects = []
        return self._get_obs(), {}

    def step(self, action):
        # Player movement
        if action == 0:
            self.player_x = max(0, self.player_x - self.player_speed)
        elif action == 2:
            self.player_x = min(
                self.width - self.player_width, self.player_x + self.player_speed
            )

        # Spawn new object
        if len(self.falling_objects) < self.max_objects and np.random.rand() < 0.05:
            self._spawn_object()

        # Move objects
        for obj in self.falling_objects:
            obj[1] += obj[4]

        # Remove off-screen
        self.falling_objects = [o for o in self.falling_objects if o[1] < self.height]

        # Catch logic
        basket_x = self.player_x - self.basket_width//2 + self.player_width//2
        basket_y = self.player_y - self.basket_height
        reward, done = 0, False
        to_remove = []
        for i, o in enumerate(self.falling_objects):
            x, y, is_rock, color_idx, _ = o
            size = self.rock_size if is_rock else self.fruit_size
            if y + size >= basket_y and y <= basket_y + self.basket_height:
                if basket_x <= x <= basket_x + self.basket_width or \
                   basket_x <= x + size <= basket_x + self.basket_width:
                    to_remove.append(i)
                    if is_rock:
                        reward = -1
                        done = True
                    else:
                        reward = 1
        for i in sorted(to_remove, reverse=True):
            self.falling_objects.pop(i)

        # Perturbation check
        self.num_steps += 1
        if self.perturb and self.num_steps == self.perturb_step:
            self._apply_perturbation()

        return self._get_obs(), reward, done, False, {}

    def _spawn_object(self):
        x = np.random.randint(0, self.width - self.fruit_size)
        is_rock = np.random.rand() < self.rock_probability
        color_idx = np.random.randint(len(self.fruit_colors))
        speed = np.random.randint(2, 6)
        for obj in self.falling_objects:
            if abs(obj[0] - x) < self.fruit_size:
                return
        self.falling_objects.append([x, 0, is_rock, color_idx, speed])

    def _apply_perturbation(self):
        print(f"Applying perturbation: {self.perturb}")
        if self.perturb == "color":
            self.bg_color = (32, 32, 32)
            self.player_color = (0, 128, 255)
            self.basket_color = (0, 128, 255)
            self.fruit_colors = [
                (255, 64, 128),
                (255, 200, 0),
                (0, 255, 255),
            ]
            self.rock_color = (255, 200, 0)
        # shape perturb only affects rendering

    def _get_obs(self):
        img = Image.new("RGB", (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(img)

        # Draw player (triangle on shape perturb)
        px0, py0 = self.player_x, self.player_y
        px1, py1 = px0 + self.player_width, py0 + self.player_height
        if self.perturb == "shape" and self.num_steps >= self.perturb_step:
            # triangle pointing up
            points = [
                (px0, py1),
                (px0 + self.player_width/2, py0),
                (px1, py1),
            ]
            draw.polygon(points, fill=self.player_color)
        else:
            draw.rectangle([px0, py0, px1, py1], fill=self.player_color)

        # Draw basket
        bx0 = self.player_x - self.basket_width//2 + self.player_width//2
        by0 = self.player_y - self.basket_height
        bx1, by1 = bx0 + self.basket_width, self.player_y
        draw.rectangle([bx0, by0, bx1, by1], fill=self.basket_color)

        # Draw falling objects
        for x, y, is_rock, color_idx, _ in self.falling_objects:
            size = self.rock_size if is_rock else self.fruit_size
            color = self.rock_color if is_rock else self.fruit_colors[color_idx]
            if self.perturb == "shape" and self.num_steps >= self.perturb_step:
                if is_rock:
                    draw.ellipse([x, y, x+size, y+size], fill=color)
                else:
                    half = size/2
                    pts = [(x+half, y), (x, y+size), (x+size, y+size)]
                    draw.polygon(pts, fill=color)
            else:
                draw.rectangle([x, y, x+size, y+size], fill=color)

        # Draw ground
        draw.rectangle([0, self.height-self.ground_height, self.width, self.height], fill=(150,150,255))

        return np.array(img)
