import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw
from gameworld.envs.base import GameworldEnv


class Cross(GameworldEnv):
    """ Based on the Atari Freeway game.
    
        Player needs cross the highway with cars moving left and right.
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

        # Screen dimensions & margins
        self.width = 160
        self.height = 210
        self.top_margin = 20
        self.bottom_margin = 20

        # Lane and divider
        self.lane_count = 8
        self.lane_height = (
            self.height - self.top_margin - self.bottom_margin
        ) // self.lane_count
        self.divider_thickness = 3
        self.lane_color = (255, 255, 255)

        # Player properties
        self.orig_player_size = 10
        self.player_size = self.orig_player_size
        self.player_speed = 6
        self.player_x = self.width // 2 - self.player_size // 2
        self.player_y = self.height - self.bottom_margin - self.player_size
        self.player_color = (255, 255, 0)

        # Car properties
        self.orig_car_size = 14
        self.car_size = self.orig_car_size
        self.car_speeds = [-1, -2, -1, -3, 3, 1, 2, 1]
        self.car_colors = [
            (255, 20, 20),
            (20, 255, 20),
            (20, 20, 255),
            (255, 40, 255),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 0, 255),
        ]
        # Background
        self.bg_color = (50, 50, 100)

        # Action & observation spaces
        self.action_space = spaces.Discrete(3)  # 0: stay, 1: up, 2: down
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 3),
            dtype=np.uint8,
        )

        self.reset()

    def reset(self):
        self.player_y = self.height - self.bottom_margin - self.player_size
        # Initialize cars per lane
        self.cars = [
            [
                (10 if self.car_speeds[i] > 0 else self.width - self.car_size - 10),
                self.top_margin + i * self.lane_height + 5,
                self.car_speeds[i],
                self.car_colors[i],
            ]
            for i in range(self.lane_count)
        ]
        return self._get_obs(), {}

    def step(self, action):
        # Move player
        if action == 1:
            self.player_y = max(self.player_y - self.player_speed, 0)
        elif action == 2:
            self.player_y = min(
                self.player_y + self.player_speed,
                self.height - self.player_size,
            )

        # Move cars
        for car in self.cars:
            car[0] += car[2]
            if car[2] > 0 and car[0] >= self.width:
                car[0] = 0
            elif car[2] < 0 and car[0] <= 0:
                car[0] = self.width

        # Collision and reward
        reward = 0
        done = False
        for car in self.cars:
            if (
                self.player_y < car[1] + self.car_size
                and self.player_y + self.player_size > car[1]
                and self.player_x < car[0] + self.car_size
                and self.player_x + self.player_size > car[0]
            ):
                self.player_y = self.height - self.bottom_margin - self.player_size
                reward = -1
                done = True
                break

        # Check crossing
        if self.player_y <= 0:
            reward = 1
            self.player_y = self.height - self.bottom_margin - self.player_size

        # Perturbation
        self.num_steps += 1
        if self.perturb and self.num_steps == self.perturb_step:
            self._apply_perturbation()

        return self._get_obs(), reward, done, False, {}

    def _apply_perturbation(self):
        print(f"Applying perturbation: {self.perturb}")
        if self.perturb == "color":
            # Change palettes
            self.bg_color = (32, 32, 32)
            self.lane_color = (200, 200, 200)
            self.player_color = (0, 128, 255)
            self.car_colors = [
                (255, 64, 128),
                (255, 200, 0),
                (0, 255, 255),
                (128, 128, 128),
                (0, 255, 128),
                (128, 0, 255),
                (255, 128, 0),
                (0, 128, 128),
            ]
            # Update existing cars' colors
            for i, car in enumerate(self.cars):
                car[3] = self.car_colors[i % len(self.car_colors)]
        elif self.perturb == "shape":
            pass

    def _get_obs(self):
        # Use PIL to draw shapes for shape perturbations
        img = Image.new("RGB", (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(img)

        # Lane dividers
        for i in range(self.lane_count + 1):
            y = self.top_margin + i * self.lane_height
            draw.rectangle(
                [0, y, self.width, y + self.divider_thickness],
                fill=self.lane_color,
            )

        # Player
        px0, py0 = self.player_x, self.player_y
        px1, py1 = px0 + self.player_size, py0 + self.player_size
        if self.perturb == "shape" and self.num_steps >= self.perturb_step:
            draw.ellipse([px0, py0, px1, py1], fill=self.player_color)
        else:
            draw.rectangle([px0, py0, px1, py1], fill=self.player_color)

        # Cars
        for car in self.cars:
            cx, cy, _, color = car
            x0, y0 = cx, cy
            x1, y1 = cx + self.car_size, cy + self.car_size
            if self.perturb == "shape" and self.num_steps >= self.perturb_step:
                draw.ellipse([x0, y0, x1, y1], fill=color)
            else:
                draw.rectangle([x0, y0, x1, y1], fill=color)

        return np.array(img)
