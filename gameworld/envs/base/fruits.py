import numpy as np
from gymnasium import spaces
from gameworld.envs.base.base_env import GameworldEnv


class Fruits(GameworldEnv):
    """ Player needs to catch fruits falling from the sky, why avoiding rocks.
    """

    def __init__(self, **kwargs):
        super().__init__()

        # Screen dimensions
        self.width = 160
        self.height = 210

        # Player properties
        self.player_width = 16
        self.player_height = 30
        self.basket_width = 24
        self.basket_height = 16
        self.ground_height = 16
        self.player_x = self.width // 2 - self.player_width // 2
        self.player_y = self.height - self.ground_height - self.basket_height
        self.player_speed = 8

        # Falling object properties
        self.fruit_size = 12  # Increased from 8 to 12
        self.rock_size = 10  # Increased slightly
        self.fruit_colors = [
            [255, 0, 0],  # Red (apple)
            [128, 0, 128],  # Purple (grape)
            [0, 255, 0],  # Green (pear)
        ]
        self.rock_color = [180, 180, 180]  # Darker/black rock

        # Falling objects list
        # Each object is [x, y, is_rock, color_idx, speed]
        self.falling_objects = []
        self.rock_probability = 0.25
        self.max_objects = 6  # Maximum objects on screen

        # Action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: left, 1: stay, 2: right
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )

    def reset(self):
        # Reset player position
        self.player_x = self.width // 2 - self.player_width // 2

        # Clear falling objects
        self.falling_objects = []

        return self._get_obs(), {}

    def step(self, action):
        if action == 0:  # Move left
            self.player_x = max(0, self.player_x - self.player_speed)
        elif action == 2:  # Move right
            self.player_x = min(
                self.width - self.player_width, self.player_x + self.player_speed
            )

        if len(self.falling_objects) < self.max_objects and np.random.rand() < 0.05:
            self._spawn_object()

        # Move existing objects
        for obj in self.falling_objects:
            obj[1] += obj[4]  # Move by speed

        # Check for objects that are off-screen
        self.falling_objects = [
            obj for obj in self.falling_objects if obj[1] < self.height
        ]

        # Calculate basket position (top of player's head)
        basket_x = self.player_x - self.basket_width // 2 + self.player_width // 2
        basket_y = self.player_y - self.basket_height

        # Check for collisions with basket
        reward = 0
        done = False
        objects_to_remove = []

        for i, obj in enumerate(self.falling_objects):
            # Check if object has reached basket height
            if (
                obj[1] + self.fruit_size >= basket_y
                and obj[1] <= basket_y + self.basket_height
            ):
                # Check if object is horizontally aligned with basket
                if (
                    basket_x <= obj[0] <= basket_x + self.basket_width
                    or basket_x
                    <= obj[0] + self.fruit_size
                    <= basket_x + self.basket_width
                ):
                    # Object is caught
                    objects_to_remove.append(i)

                    if obj[2]:  # It's a rock
                        reward = -1
                        done = True  # Game over when rock is caught (single-life)
                    else:  # It's a fruit
                        reward = 1

        # Remove caught objects
        for i in sorted(objects_to_remove, reverse=True):
            if i < len(self.falling_objects):
                self.falling_objects.pop(i)

        return self._get_obs(), reward, done, False, {}

    def _spawn_object(self):
        """Spawn a new falling object"""

        # Determine x position (random but within screen bounds)
        x = np.random.randint(0, self.width - self.fruit_size)

        # Determine if it's a rock (increasing chance over time)
        is_rock = np.random.random() < self.rock_probability

        # Determine color (if fruit)
        color_idx = np.random.randint(0, len(self.fruit_colors))

        # Determine speed (random between 2-5)
        min_speed = 2
        max_speed = 6
        speed = np.random.randint(min_speed, max_speed)

        # Avoid overlapping objects
        for obj in self.falling_objects:
            if abs(obj[0] - x) < self.fruit_size:
                return

        # Add object to list
        self.falling_objects.append([x, 0, is_rock, color_idx, speed])

    def _get_obs(self):
        obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        obs[:, :, 0] = 50  # Background color (dark blue)
        obs[:, :, 1] = 50
        obs[:, :, 2] = 100

        # player
        obs[
            self.player_y : self.player_y + self.player_height,
            self.player_x : self.player_x + self.player_width,
        ] = [
            255,
            255,
            0,
        ]

        # basket
        basket_x = max(
            0, self.player_x - self.basket_width // 2 + self.player_width // 2
        )
        basket_y = self.player_y - self.basket_height
        obs[
            basket_y : self.player_y,
            basket_x : basket_x + self.basket_width,
        ] = [
            255,
            255,
            0,
        ]  # Yellow basket

        # Draw falling objects
        for obj in self.falling_objects:
            x, y, is_rock, color_idx, _ = obj

            if is_rock:
                # Draw rock
                obs[
                    int(y) : int(y) + self.fruit_size,
                    int(x) : int(x) + self.fruit_size,
                ] = self.rock_color
            else:
                # Draw fruit
                obs[
                    int(y) : int(y) + self.fruit_size,
                    int(x) : int(x) + self.fruit_size,
                ] = self.fruit_colors[color_idx]

        # Draw ground
        obs[self.height - self.ground_height :, :, 0] = 150  # R
        obs[self.height - self.ground_height :, :, 1] = 150  # G
        obs[self.height - self.ground_height :, :, 2] = 255  # B

        return obs
