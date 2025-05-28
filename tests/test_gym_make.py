import pytest
import gymnasium
import gameworld.envs  # this import triggers your register() calls

from gymnasium import Env
from gymnasium.spaces import Space

# list out the names you registered
GAME_NAMES = [
    "Aviate",
    "Bounce",
    "Cross",
    "Drive",
    "Explode",
    "Fruits",
    "Gold",
    "Hunt",
    "Impact",
    "Jump",
]

@pytest.mark.parametrize("game", GAME_NAMES)
def test_gym_make_and_basic_api(game):
    env_id = f"Gameworld-{game}-v0"

    # should register without error
    env = gymnasium.make(env_id)
    assert isinstance(env, Env), f"{env_id} did not return a gymnasium.Env"

    # check the spec is correct
    assert env.spec is not None
    assert env.spec.id == env_id

    # basic reset API
    obs, info = env.reset()
    # obs can be anything; info should be a dict
    assert isinstance(info, dict)

    # check we have a well-formed action_space
    assert hasattr(env, "action_space"), "no action_space"
    assert isinstance(env.action_space, Space)

    # take one random step
    action = env.action_space.sample()
    step_out = env.step(action)
    # should be a tuple of (obs, reward, terminated, truncated, info)
    assert len(step_out) == 5
    new_obs, reward, terminated, truncated, info = step_out

    # check types
    assert isinstance(info, dict)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(reward, (int, float))

    # cleanup
    env.close()
