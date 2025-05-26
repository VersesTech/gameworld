import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw

from gameworld.envs.base import GameworldEnv


class Hunt(GameworldEnv):
    """ Based on the Atari Asterix game.
    
        Player needs to catch rewards and avoid obstacles.
        The player can move left, right, up, or down.
        The game is played in a 2D grid with a fixed number of lanes.
    """

    def __init__(
        self,
        max_objects=3,
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

        # original sizes for shape perturb
        self.orig_player_w = 20
        self.orig_player_h = self.lane_height
        self.orig_item_size = self.lane_height // 2

        # mutable sizes
        self.player_width = self.orig_player_w
        self.player_height = self.orig_player_h
        self.item_size = self.orig_item_size

        # player starting position & speed
        self.player_speed = 8
        self.player_x = self.width // 2 - self.player_width // 2
        self.player_y = self.top_margin + (self.lane_count // 2) * self.lane_height

        # containers for items and obstacles
        self.items = []
        self.obstacles = []
        self.max_items = max_objects
        self.max_obstacles = max_objects

        # colours
        self.bg_color = (50, 50, 100)  # dark blue
        self.lane_color = (255, 255, 255)  # white
        self.player_color = (255, 255, 0)  # yellow
        self.item_color = (0, 255, 0)  # green
        self.obstacle_color = (255, 0, 0)  # red

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

        # spawn items/obstacles
        if len(self.items) < self.max_items and np.random.rand() < 0.05:
            self._spawn_entity(self.items)
        if len(self.obstacles) < self.max_obstacles and np.random.rand() < 0.05:
            self._spawn_entity(self.obstacles)

        # update positions
        for obj in self.items + self.obstacles:
            obj[0] += obj[2]

        # collisions and cleanup
        reward = 0
        for obj in list(self.items):
            if self._overlap(obj):
                reward += 1
                self.items.remove(obj)
            elif obj[0] < -self.item_size or obj[0] > self.width + self.item_size:
                self.items.remove(obj)
        for obj in list(self.obstacles):
            if self._overlap(obj):
                reward -= 1
                self.obstacles.remove(obj)
            elif obj[0] < -self.item_size or obj[0] > self.width + self.item_size:
                self.obstacles.remove(obj)

        # perturbation check
        self.num_steps += 1
        if self.perturb and self.num_steps == self.perturb_step:
            self._apply_perturbation()

        return self._get_obs(), reward, False, False, {}

    def _spawn_entity(self, container):
        # try up to 4 lanes to find empty
        for _ in range(4):
            lane = np.random.randint(self.lane_count)
            y = self.top_margin + lane * self.lane_height + self.lane_height // 4
            if all(ent[1] != y for ent in self.items + self.obstacles):
                left = bool(np.random.randint(2))
                x = 0 if left else self.width
                dx = 2 if left else -2
                container.append([x, y, dx])
                break

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
            self.lane_color = (200, 200, 200)
            self.player_color = (0, 128, 255)
            self.item_color = (255, 64, 128)
            self.obstacle_color = (255, 200, 0)
        elif self.perturb == "shape":
            scale = 1.5
            self.player_width = int(self.orig_player_w * scale)
            self.player_height = int(self.orig_player_h * scale)
            self.item_size = int(self.orig_item_size * 2)
            # clamp
            self.player_x = min(self.player_x, self.width - self.player_width)
            self.player_y = min(
                self.player_y, self.height - self.bottom_margin - self.player_height
            )

    def _get_obs(self):
        img = Image.new("RGB", (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(img)

        # draw lanes
        for i in range(self.lane_count + 1):
            y = self.top_margin + i * self.lane_height - 1
            draw.rectangle([0, y, self.width, y + 2], fill=self.lane_color)

        # draw player
        px0, py0 = self.player_x, self.player_y
        px1, py1 = px0 + self.player_width, py0 + self.player_height
        if self.perturb == "shape" and self.num_steps >= self.perturb_step:
            draw.ellipse([px0, py0, px1, py1], fill=self.player_color)
        else:
            draw.rectangle([px0, py0, px1, py1], fill=self.player_color)

        # draw items
        for x, y, _ in self.items:
            ix0, iy0 = x, y
            ix1, iy1 = ix0 + self.item_size, iy0 + self.item_size
            if self.perturb == "shape" and self.num_steps >= self.perturb_step:
                draw.ellipse([ix0, iy0, ix1, iy1], fill=self.item_color)
            else:
                draw.rectangle([ix0, iy0, ix1, iy1], fill=self.item_color)

        # draw obstacles
        for x, y, _ in self.obstacles:
            ox0, oy0 = x, y
            ox1, oy1 = ox0 + self.item_size, oy0 + self.item_size
            if self.perturb == "shape" and self.num_steps >= self.perturb_step:
                draw.ellipse([ox0, oy0, ox1, oy1], fill=self.obstacle_color)
            else:
                draw.rectangle([ox0, oy0, ox1, oy1], fill=self.obstacle_color)

        return np.array(img)

