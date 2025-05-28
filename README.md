# Gameworld 10k

The Gameworld environments are designed to develop sample-efficient learning algorithms, which is enforced in the challenge by limiting the interactions to 10K steps. These environments could be solved by humans within minutes, ensuring that learning does not hinge on brittle exploration or complex credit assignment.

Gameworld also supports controlled perturbations such as changes in object color or shape, testing an agent's ability to generalize across superficial domain shifts. 

## Environments

The suite includes 10 diverse games generated with the aid of a large language model, drawing inspiration from ALE and classic video games, while maintaining a lightweight and structured design.

<table>
<tr><th>Aviate</th><th>Bounce</th><th>Cross</th><th>Drive</th><th>Explode</th></tr>
<tr><td><img src=.github/videos/Aviate.gif width=50/></td><td><img src=.github/videos/Bounce.gif width=50/></td><td><img src=.github/videos/Cross.gif width=50/></td><td><img src=.github/videos/Drive.gif width=50/></td><td><img src=.github/videos/Explode.gif width=50/></td></tr>
<tr><th>Fruits</th><th>Gold</th><th>Hunt</th><th>Impact</th><th>Jump</th></tr>
<tr><td><img src=.github/videos/Fruits.gif width=50/></td><td><img src=.github/videos/Gold.gif width=50/></td><td><img src=.github/videos/Hunt.gif width=50/></td><td><img src=.github/videos/Impact.gif width=50/></td><td><img src=.github/videos/Jump.gif width=50/></td></tr>
</table>

## Installation

To install the `gameworld` library, either use the pypi registry `pip install gameworld`, or install from source by cloning the repository and running `pip install -e .`. 

This library has been developed and tested on`python3.11` for both Linux and macOS. 

## API

The Gameworld environments use the gymnasium.Gym api. To run your own algorithm against our environments, create an environment instance as:

```python
import gameworld.envs # Triggers registering the environments in Gymnasium
import gymnasium

game = "Aviate"
env = gymnasium.make(f"Gameworld-{game}-v0")

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

## Perturbations

The Gameworld environments support controlled perturbations in order. For each environment, you can choose no (`None`), shape (`shape`), or a color (`color`) perturbation.

For example, a shape perturbation after 5000 steps in the `Explode` environment can be created using the following snippet: 

```python
env = gymnasium.make(
    f"Gameworld-Explode-v0", perturb='shape', perturb_step=5000
)
```

Below we show an example of a shape and color perturbation on Explode and Fruits: 

<table>
<tr><th>Explode Color</th><th>Explode Shape</th><th>Fruits Color</th><th>Fruits Shape</th>
<tr>
    <td>
        <img src='.github/videos/Explode_color.gif' width=50/> 
    </td>
    <td>
        <img src='.github/videos/Explode_shape.gif' width=50/> 
    </td>
    <td>
        <img src='.github/videos/Fruits_color.gif' width=50/> 
    </td>
    <td>
        <img src='.github/videos/Fruits_shape.gif' width=50/> 
    </td>
</tr>

</table>