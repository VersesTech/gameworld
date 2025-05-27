import numpy as np
from gymnasium import spaces
from gameworld.envs.base.base_env import GameworldEnv


class Cross(GameworldEnv):
    """ Based on the Atari Freeway game.
    
        Player needs cross the highway with cars moving left and right.
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.width = 160
        self.height = 210
        self.top_margin = 20
        self.bottom_margin = 20
        self.lane_count = 8
        self.lane_height = (
            self.height - self.top_margin - self.bottom_margin
        ) // self.lane_count
        self.player_x = self.width // 2 - 5
        self.player_y = self.height - self.bottom_margin - 10
        self.player_size = 10
        self.player_speed = 6

        self.cars = []
        self.car_speeds = self.car_speeds = [
            -1,
            -2,
            -1,
            -3,
            3,
            1,
            2,
            1,
        ]  # Different speeds for cars
        self.car_colors = [
            (255, 20, 20),
            (20, 255, 20),
            (20, 20, 255),
            (255, 40, 255),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 0, 255),
        ]  # Different colors for cars
        self.car_size = 14
        self.reset()

        self.action_space = spaces.Discrete(3)  # 0: stay, 1: up, 2: down
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(210, 160, 3), dtype=np.uint8
        )

    def reset(self):
        self.player_y = self.height - 10
        self.cars = [
            [
                (
                    10
                    if self.car_speeds[i % len(self.car_speeds)] > 0
                    else self.width - self.car_size - 10
                ),
                self.top_margin + i * self.lane_height + 5,
                self.car_speeds[i % len(self.car_speeds)],
                self.car_colors[i % len(self.car_colors)],
            ]
            for i in range(self.lane_count)
        ]
        return self._get_obs(), {}

    def step(self, action):
        if action == 1:
            self.player_y = max(self.player_y - self.player_speed, 0)
        elif action == 2:
            self.player_y = min(self.player_y + self.player_speed, self.height - 10)

        for car in self.cars:
            car[0] += car[2]  # Move left/right
            if car[2] > 0 and car[0] >= self.width:
                car[0] = 0  # Respawn on left
            elif car[2] < 0 and car[0] <= 0:
                car[0] = self.width  # Respawn on right

        reward = 0
        done = False

        for car in self.cars:
            # Check for collision
            if (
                self.player_y - self.car_size
                < car[1]
                < self.player_y + self.player_size
                and self.player_x - self.car_size
                < car[0]
                < self.player_x + self.player_size
            ):
                self.player_y = self.height - 10  # Reset player position
                reward = -1
                done = True  # Reset game after collision
                break

        if self.player_y == 0:
            reward = 1  # Reward for reaching the top
            # done = True  # Reset game after successful crossing
            # Reset player position
            self.player_y = self.height - 10

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        obs[:, :, 0] = 50  # Background color (dark blue)
        obs[:, :, 1] = 50
        obs[:, :, 2] = 100

        for i in range(self.lane_count + 1):
            y = self.top_margin + i * self.lane_height
            obs[y : y + 3, :, :] = [255, 255, 255]  # White lane dividers

        if self.player_y > 0:
            obs[
                self.player_y : self.player_y + self.player_size,
                self.player_x : self.player_x + self.player_size,
            ] = [
                255,
                255,
                0,
            ]  # Yellow player
        for car in self.cars:
            obs[car[1] : car[1] + self.car_size, car[0] : car[0] + self.car_size] = car[
                3
            ]  # Car with assigned color

        return obs
