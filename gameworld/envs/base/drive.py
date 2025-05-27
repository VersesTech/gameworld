import numpy as np
from gymnasium import spaces
from gameworld.envs.base.base_env import GameworldEnv


class Drive(GameworldEnv):
    """ Player needs to drive a car on a highway and avoid other cars.
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.width = 160
        self.height = 210
        self.car_width = 14
        self.car_height = 24
        self.road_width = 100
        self.lane_count = 4
        self.max_cars_per_lane = 4

        self.player_y = self.height - 30

        self.lane_positions = [
            int(
                self.width // 2
                - self.road_width // 2
                + (i + 0.5) * (self.road_width // self.lane_count)
                - self.car_width // 2
            )
            for i in range(self.lane_count)
        ]

        self.opponents = []
        self.spawn_probability = 0.05

        self.action_space = spaces.Discrete(3)  # 0: stay, 1: left, 2: right
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(210, 160, 3), dtype=np.uint8
        )
        self.colors = np.array(
            [
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 255],
            ]
        )

        self.reset()

    def reset(self):
        self.player_y = self.height - 30
        self.player_x = self.width // 2 - self.car_width // 2
        self.opponents = []
        return self._get_obs(), {}

    def step(self, action):
        if action == 1 and self.player_x > self.width // 2 - self.road_width // 2:
            self.player_x -= 2
        elif (
            action == 2
            and self.player_x
            < self.width // 2 - self.road_width // 2 + self.road_width - self.car_width
        ):
            self.player_x += 2

        reward = 0
        done = False

        lane_counts = {lane: 0 for lane in range(self.lane_count)}
        for opp in self.opponents:
            lane_counts[opp["lane"]] += 1

        # Occasionally spawn new opponent cars with no overlap in same lane and under max cars per lane
        if np.random.rand() < self.spawn_probability and len(self.opponents) < 3:
            lane = np.random.randint(self.lane_count)
            if lane_counts[lane] < self.max_cars_per_lane:
                x = self.lane_positions[lane]
                y = -self.car_height
                speed = (
                    np.random.randint(1, 3) if lane >= 2 else np.random.randint(3, 5)
                )
                new_car = {
                    "x": x,
                    "y": y,
                    "speed": speed,
                    "color": self.colors[np.random.randint(0, self.colors.shape[0])],
                    "lane": lane,
                }
                # Avoid overlap with cars already in lane
                same_lane = [opp for opp in self.opponents if opp["lane"] == lane]
                if all(
                    abs(new_car["y"] - opp["y"]) > self.car_height for opp in same_lane
                ):
                    self.opponents.append(new_car)

        # Move opponents and slow down if they would collide
        for i, opp in enumerate(self.opponents):
            same_lane = [
                o
                for j, o in enumerate(self.opponents)
                if j != i and o["lane"] == opp["lane"]
            ]
            for other in same_lane:
                if 0 < (other["y"] - (opp["y"] + self.car_height)) < opp["speed"] + 5:
                    opp["speed"] = other["speed"]

            opp["y"] += opp["speed"]

        # Remove off-screen cars
        self.opponents = [opp for opp in self.opponents if opp["y"] <= self.height]

        # Collision check
        for opp in self.opponents:
            if (
                self.player_x < opp["x"] + self.car_width
                and self.player_x + self.car_width > opp["x"]
                and self.player_y < opp["y"] + self.car_height
                and self.player_y + self.car_height > opp["y"]
            ):
                reward = -1
                # dissappear the player and opponent car it collided with
                self.player_y = self.height + 1
                opp["y"] = self.height + 1
                done = True

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        obs[:, :, 0] = 150
        obs[:, :, 1] = 150
        obs[:, :, 2] = 255

        # Road
        road_left = self.width // 2 - self.road_width // 2
        road_right = self.width // 2 + self.road_width // 2
        obs[:, road_left:road_right] = [50, 50, 100]

        # Player car
        if self.player_y < self.height:
            player_x = int(self.player_x)
            obs[
                self.player_y : self.player_y + self.car_height,
                player_x : player_x + self.car_width,
            ] = [255, 255, 0]

        # Opponent cars
        for opp in self.opponents:
            opp_x = int(opp["x"])
            opp_y = int(opp["y"])
            if 0 <= opp_y + self.car_height and opp_y < self.height:
                obs[
                    max(0, opp_y) : min(self.height, opp_y + self.car_height),
                    opp_x : opp_x + self.car_width,
                ] = opp["color"]

        return obs
