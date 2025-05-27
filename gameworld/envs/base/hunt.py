import numpy as np
from gymnasium import spaces
from gameworld.envs.base.base_env import GameworldEnv


class Hunt(GameworldEnv):
    """ Based on the Atari Asterix game.
    
        Player needs to catch rewards and avoid obstacles.
        The player can move left, right, up, or down.
        The game is played in a 2D grid with a fixed number of lanes.
    """

    def __init__(self, max_objects=3, **kwargs):
        super().__init__()

        self.width = 160
        self.height = 210
        self.top_margin = 20
        self.bottom_margin = 20
        self.player_width = 20
        self.player_speed = 8
        self.lane_count = 8
        self.lane_height = (
            self.height - self.top_margin - self.bottom_margin
        ) // self.lane_count
        self.player_lane = self.lane_count // 2  # Start in the middle lane
        self.player_height = self.lane_height
        self.item_size = self.lane_height // 2
        self.player_x = self.width // 2 - self.player_width // 2
        self.player_y = self.top_margin + self.player_lane * self.lane_height
        self.items = []
        self.obstacles = []
        self.max_items = max_objects
        self.max_obstacles = max_objects

        self.action_space = spaces.Discrete(
            5
        )  # 0: stay, 1: left, 2: right, 3: up, 4: down  # 0: stay, 1: left, 2: right
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(210, 160, 3), dtype=np.uint8
        )

    def reset(self):
        self.player_x = self.width // 2 - self.player_width // 2
        self.items = []
        self.obstacles = []
        return self._get_obs(), {}

    def _is_empty(self, lane):
        for item in self.items:
            if (
                item[1]
                == self.top_margin + lane * self.lane_height + self.lane_height // 4
            ):
                return False
        for obstacle in self.obstacles:
            if (
                obstacle[1]
                == self.top_margin + lane * self.lane_height + self.lane_height // 4
            ):
                return False
        return True

    def step(self, action):
        if action == 0:
            self.player_x = max(self.player_x - self.player_speed, 0)
        elif action == 1:
            self.player_x = min(
                self.player_x + self.player_speed, self.width - self.player_width
            )
        elif action == 2:
            self.player_y = max(self.player_y - self.player_speed, self.top_margin)
        elif action == 3:
            self.player_y = min(
                self.player_y + self.player_speed, self.height - self.bottom_margin
            )

        for item in self.items:
            item[0] += item[2]  # Move left or right

        for obstacle in self.obstacles:
            obstacle[0] += obstacle[2]  # Move left or right

        if len(self.items) < self.max_items and np.random.rand() < 0.05:
            left = np.random.choice([0, 1])
            lane = np.random.choice(range(self.lane_count))
            if self._is_empty(lane):
                self.items.append(
                    [
                        0 if left else self.width,
                        self.top_margin
                        + lane * self.lane_height
                        + self.lane_height // 4,
                        2 if left else -2,
                    ]
                )  # Moves left or right  # (x, y, speed)

        if len(self.obstacles) < self.max_obstacles and np.random.rand() < 0.05:
            left = np.random.choice([0, 1])
            lane = np.random.choice(range(self.lane_count))
            if self._is_empty(lane):
                self.obstacles.append(
                    [
                        0 if left else self.width,
                        self.top_margin
                        + lane * self.lane_height
                        + self.lane_height // 4,
                        2 if left else -2,
                    ]
                )  # Moves left or right  # (x, y, speed)

        reward = 0
        done = False

        for item in self.items:
            if (
                self.player_y - self.item_size
                <= item[1]
                <= self.player_y + self.player_width
                and self.player_x - self.item_size
                <= item[0]
                <= self.player_x + self.player_width
            ):
                reward += 1
                self.items.remove(item)
            elif item[0] < 0 or item[0] > self.width:
                self.items.remove(item)

        for obstacle in self.obstacles:
            if (
                self.player_y - self.item_size
                <= obstacle[1]
                <= self.player_y + self.player_width
                and self.player_x - self.item_size
                <= obstacle[0]
                <= self.player_x + self.player_width
            ):
                reward -= 1
                self.obstacles.remove(obstacle)
            elif obstacle[0] < 0 or obstacle[0] > self.width:
                self.obstacles.remove(obstacle)

        self.items = [item for item in self.items if item[1] < self.height]
        self.obstacles = [
            obstacle for obstacle in self.obstacles if obstacle[1] < self.height
        ]

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        obs[:, :, 0] = 50  # Background color (dark blue)
        obs[:, :, 1] = 50
        obs[:, :, 2] = 100

        for i in range(self.lane_count + 1):
            y = self.top_margin + i * self.lane_height - 1
            obs[y : y + 3, :, :] = [255, 255, 255]  # White lane dividers

        obs[
            self.player_y : self.player_y + self.lane_height,
            self.player_x : self.player_x + self.player_width,
        ] = [
            255,
            255,
            0,
        ]  # Yellow player
        for item in self.items:
            obs[
                item[1] : item[1] + self.item_size, item[0] : item[0] + self.item_size
            ] = [
                0,
                255,
                0,
            ]  # Green item

        for obstacle in self.obstacles:
            obs[
                obstacle[1] : obstacle[1] + self.item_size,
                obstacle[0] : obstacle[0] + self.item_size,
            ] = [
                255,
                0,
                0,
            ]  # Red obstacle

        return obs
