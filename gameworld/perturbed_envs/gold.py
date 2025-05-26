import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw

from gameworld.envs.base import GameworldEnv


class Gold(GameworldEnv):
    """ Basic coin collector game.
    
        Player moves left, right, up, or down to collect coins and avoid obstacles.
    """

    def __init__(
        self,
        max_coins=3,
        max_obstacles=3,
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

        # world dimensions & lanes
        self.width = 160
        self.height = 210
        self.top_margin = 20
        self.bottom_margin = 20
        self.lane_count = 8
        self.lane_height = (
            self.height - self.top_margin - self.bottom_margin
        ) // self.lane_count

        # sizes
        self.player_width = 20
        self.player_height = self.lane_height
        self.item_size = self.lane_height // 2

        # player start & speed
        self.player_speed = 8
        self.player_x = self.width // 2 - self.player_width // 2
        self.player_y = self.top_margin + (self.lane_count // 2) * self.lane_height

        # items & obstacles
        self.items = []  # coins
        self.obstacles = []  # moving dangers
        self.max_coins = max_coins
        self.max_obstacles = max_obstacles

        # colors
        self.bg_color = (50, 50, 100)       # dark blue
        self.player_color = (255, 255, 0)   # yellow
        self.coin_color = (0, 255, 0)       # green
        self.obstacle_color = (255, 0, 0)   # red

        # spaces
        self.action_space = spaces.Discrete(5)  # stay, left, right, up, down
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )

        self.reset()

    def reset(self):
        self.player_x = self.width // 2 - self.player_width // 2
        self.player_y = self.top_margin + (self.lane_count // 2) * self.lane_height
        self.items = []
        self.obstacles = []
        return self._get_obs(), {}

    def step(self, action):
        # player movement
        if action == 1:
            self.player_x = max(0, self.player_x - self.player_speed)
        elif action == 2:
            self.player_x = min(
                self.width - self.player_width, self.player_x + self.player_speed
            )
        elif action == 3:
            self.player_y = max(self.top_margin, self.player_y - self.player_speed)
        elif action == 4:
            self.player_y = min(
                self.height - self.bottom_margin - self.player_height,
                self.player_y + self.player_speed,
            )

        # update obstacles
        for obs in self.obstacles:
            obs[0] += obs[2]

        # spawn coins (stationary)
        if len(self.items) < self.max_coins and np.random.rand() < 0.05:
            lane = np.random.randint(self.lane_count)
            y = self.top_margin + lane * self.lane_height + self.lane_height // 4
            x = np.random.randint(10, self.width - 10)
            self.items.append([x, y, 0])

        # spawn obstacles (move)
        if len(self.obstacles) < self.max_obstacles and np.random.rand() < 0.05:
            left = bool(np.random.randint(2))
            lane = np.random.randint(self.lane_count)
            y = self.top_margin + lane * self.lane_height + self.lane_height // 4
            x = 0 if left else self.width
            dx = 3 if left else -3
            self.obstacles.append([x, y, dx])

        # collisions & cleanup
        reward = 0
        done = False
        for coin in list(self.items):
            if self._overlap(coin):
                reward = 1
                self.items.remove(coin)

        for obs in list(self.obstacles):
            if self._overlap(obs):
                reward = -1
                done = True
                self.obstacles.remove(obs)
            elif obs[0] < -self.item_size or obs[0] > self.width + self.item_size:
                self.obstacles.remove(obs)

        # perturbation check
        self.num_steps += 1
        if self.perturb and self.num_steps == self.perturb_step:
            self._apply_perturbation()

        return self._get_obs(), reward, done, False, {}

    def _overlap(self, obj):
        x, y, _ = obj
        return (
            self.player_y - self.item_size <= y <= self.player_y + self.player_height
            and self.player_x - self.item_size <= x <= self.player_x + self.player_width
        )

    def _apply_perturbation(self):
        print(f"Applying perturbation: {self.perturb}")
        if self.perturb == "color":
            self.bg_color = (32, 32, 32)
            self.player_color = (0, 128, 255)
            self.coin_color = (255, 64, 128)
            self.obstacle_color = (255, 200, 0)
        # shape perturb only affects rendering, no size changes

    def _get_obs(self):
        img = Image.new("RGB", (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(img)

        # draw player
        px0, py0 = self.player_x, self.player_y
        px1, py1 = px0 + self.player_width, py0 + self.player_height
        draw.rectangle([px0, py0, px1, py1], fill=self.player_color)

        # draw coins
        for x, y, _ in self.items:
            ix0, iy0 = x, y
            ix1, iy1 = ix0 + self.item_size, iy0 + self.item_size
            if self.perturb == "shape" and self.num_steps >= self.perturb_step:
                # triangular coin
                half = self.item_size // 2
                points = [(ix0 + half, iy0), (ix0, iy1), (ix1, iy1)]
                draw.polygon(points, fill=self.coin_color)
            else:
                draw.rectangle([ix0, iy0, ix1, iy1], fill=self.coin_color)

        # draw obstacles
        for x, y, _ in self.obstacles:
            ox0, oy0 = x, y
            ox1, oy1 = ox0 + self.item_size, oy0 + self.item_size
            if self.perturb == "shape" and self.num_steps >= self.perturb_step:
                # circular obstacle
                draw.ellipse([ox0, oy0, ox1, oy1], fill=self.obstacle_color)
            else:
                draw.rectangle([ox0, oy0, ox1, oy1], fill=self.obstacle_color)

        return np.array(img)

