import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw

from gameworld.envs.base import GameworldEnv


class Drive(GameworldEnv):
    """ Player needs to drive a car on a highway and avoid other cars.
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
        self.num_steps = 0  # global step counter, persists across resets

        # screen dims and road
        self.width = 160
        self.height = 210
        self.road_width = 100
        self.lane_count = 4
        self.max_cars_per_lane = 4
        self.car_width = 14
        self.car_height = 24
        self.spawn_probability = 0.05

        # precompute lane x positions
        center = self.width//2
        half_road = self.road_width//2
        lane_w = self.road_width/self.lane_count
        self.lane_positions = [
            int(center - half_road + (i+0.5)*lane_w - self.car_width/2)
            for i in range(self.lane_count)
        ]

        # player initial
        self.player_x = center - self.car_width//2
        self.player_y = self.height - 30

        # opponents
        self.opponents = []  # list of dict {x,y,speed,color,lane}
        # static palette for opponents
        self.colors = [
            (255,0,0), (0,255,0), (0,0,255), (255,0,255)
        ]

        # default colors
        self.bg_color = (150,150,255)
        self.road_color = (50,50,100)
        self.player_color = (255,255,0)
        self.obstacle_color = (255,0,0)

        # spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )

        # initialize
        self.reset()

    def reset(self):
        # reposition player and clear opponents
        self.player_x = self.width//2 - self.car_width//2
        self.player_y = self.height - 30
        self.opponents = []
        # note: self.num_steps NOT reset here
        return self._get_obs(), {}

    def step(self, action):
        # lateral control
        left_bound = self.width//2 - self.road_width//2
        right_bound = left_bound + self.road_width - self.car_width
        if action == 1 and self.player_x > left_bound:
            self.player_x -= 2
        elif action == 2 and self.player_x < right_bound:
            self.player_x += 2

        # spawn logic
        reward, done = 0, False
        lane_counts = {i:0 for i in range(self.lane_count)}
        for opp in self.opponents:
            lane_counts[opp['lane']] += 1
        if np.random.rand() < self.spawn_probability and len(self.opponents)<3:
            lane = np.random.randint(self.lane_count)
            if lane_counts[lane] < self.max_cars_per_lane:
                x = self.lane_positions[lane]
                y = -self.car_height
                speed = np.random.randint(1,3) if lane>=2 else np.random.randint(3,5)
                color = self.colors[np.random.randint(len(self.colors))]
                # avoid overlap
                same = [o for o in self.opponents if o['lane']==lane]
                if all(abs(y-o['y'])>self.car_height for o in same):
                    self.opponents.append({'x':x,'y':y,'speed':speed,'color':color,'lane':lane})

        # move opponents and adjust speeds
        for i,opp in enumerate(self.opponents):
            same_lane = [o for j,o in enumerate(self.opponents) if j!=i and o['lane']==opp['lane']]
            for other in same_lane:
                if 0 < other['y']-(opp['y']+self.car_height) < opp['speed']+5:
                    opp['speed'] = other['speed']
            opp['y'] += opp['speed']
        # cull off-screen
        self.opponents = [o for o in self.opponents if o['y']<=self.height]

        # collision check
        for opp in self.opponents:
            if (self.player_x < opp['x']+self.car_width and
                self.player_x+self.car_width > opp['x'] and
                self.player_y < opp['y']+self.car_height and
                self.player_y+self.car_height > opp['y']):
                reward, done = -1, True
                # disappear
                self.player_y = self.height+1
                opp['y'] = self.height+1
                break

        # perturbation timing
        self.num_steps += 1
        if self.perturb and self.num_steps==self.perturb_step:
            self._apply_perturbation()

        return self._get_obs(), reward, done, False, {}

    def _apply_perturbation(self):
        print(f"Applying perturbation: {self.perturb}")
        if self.perturb=='color':
            # swap to high contrast
            self.bg_color = (32,32,32)
            self.road_color = (100,100,100)
            self.player_color = (0,128,255)
            self.obstacle_color = (255,200,0)
        elif self.perturb=='shape':
            # shape: player as triangle, opponents as circles
            self.scale = 1.2
            self.player_w_tri = self.car_width*self.scale
            self.player_h_tri = self.car_height*self.scale
            # opponents same size circles

    def _get_obs(self):
        # canvas
        img = Image.new('RGB',(self.width,self.height),self.bg_color)
        draw = ImageDraw.Draw(img)
        # road
        left = self.width//2 - self.road_width//2
        right = left + self.road_width
        draw.rectangle([left,0,right,self.height],fill=self.road_color)
        # player
        px,py = self.player_x,self.player_y
        if self.perturb=='shape' and self.num_steps>=self.perturb_step:
            # triangle pointing up
            pts = [(px,py+self.car_height),(px+self.car_width/2,py),(px+self.car_width,py+self.car_height)]
            draw.polygon(pts,fill=self.player_color)
        else:
            draw.rectangle([px,py,px+self.car_width,py+self.car_height],fill=self.player_color)
        # opponents
        for opp in self.opponents:
            ox,oy = opp['x'],opp['y']
            if self.perturb=='shape' and self.num_steps>=self.perturb_step:
                draw.ellipse([ox,oy,ox+self.car_width,oy+self.car_height],fill=opp['color'])
            else:
                draw.rectangle([ox,oy,ox+self.car_width,oy+self.car_height],fill=opp['color'])
        return np.array(img)