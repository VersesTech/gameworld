## Gameworld 10k

We introduce a set of environments using the `gymnasium.Gym` api. To run your own algorithm against our environments, create an environment instance as:

```
from gameworld.envs import create_gameworld_env

env = create_gameworld_env(game="Explode")

obs, info = env.reset()

for t in range(num_steps):
    # random actions as example
    action = env.action_space.sample()

    # step env
    obs, reward, done, truncated, info = env.step(action)

    # reset when done
    if done:
        obs, info = env.reset()

```
