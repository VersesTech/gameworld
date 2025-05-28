# Gameworld 10k

We introduce a set of environments using the `gymnasium.Gym` api. 

|Aviate |Bounce |Cross |Drive |Explode |Fruits |Gold |Hunt |Impact |Jump |
| -- |  -- |  -- |  -- |  -- |  -- |  -- |  -- |  -- |  -- | 
|<img src=.github/videos/Aviate.gif width=50/> |<img src=.github/videos/Bounce.gif width=50/> |<img src=.github/videos/Cross.gif width=50/> |<img src=.github/videos/Drive.gif width=50/> |<img src=.github/videos/Explode.gif width=50/> |<img src=.github/videos/Fruits.gif width=50/> |<img src=.github/videos/Gold.gif width=50/> |<img src=.github/videos/Hunt.gif width=50/> |<img src=.github/videos/Impact.gif width=50/> |<img src=.github/videos/Jump.gif width=50/> |
## Installation

The  


## Useage

To run your own algorithm against our environments, create an environment instance as:

```python
from gameworld.envs import create_gameworld_env

env = create_gameworld_env(game="Explode")

obs, info = env.reset()

for t in range(10_000):
    # random actions as example
    action = env.action_space.sample()

    # step env
    obs, reward, done, truncated, info = env.step(action)

    # reset when done
    if done:
        obs, info = env.reset()
```
