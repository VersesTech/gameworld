import numpy as np
import pytest
import matplotlib.pyplot as plt

from gameworld.envs.base import create_gameworld_env as create_env
from gameworld.envs.perturbed import create_gameworld_env as create_perturbed_env
import gymnasium
import gameworld.envs

# all of your game names
ENV_NAMES = [
    "Aviate", "Bounce", "Cross", "Drive", "Explode",
    "Fruits", "Gold", "Hunt", "Impact", "Jump"
]

@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_reset_equivalence_gym_make(env_name):
    # --------------------------------------------------
    # seed BEFORE reset so both envs see the same randomness
    # --------------------------------------------------
    np.random.seed(12345)
    # env       = create_env(env_name)
    env_id = f"Gameworld-{env_name}-v0"
    env = gymnasium.make(env_id)

    obs1, info1 = env.reset()

    np.random.seed(12345)
    # env_p     = create_perturbed_env(env_name, perturb=None)
    env_p = gymnasium.make(env_id, perturb='shape', perturb_step=5000)
    obs2, info2 = env_p.reset()

    # compare raw observations and info
    assert obs1.dtype == obs2.dtype
    assert obs1.shape == obs2.shape
    assert np.allclose(obs1, obs2)
    assert info1 == info2


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_reset_equivalence_gameworld_create(env_name):
    # --------------------------------------------------
    # seed BEFORE reset so both envs see the same randomness
    # --------------------------------------------------
    np.random.seed(12345)
    env       = create_env(env_name)
    obs1, info1 = env.reset()

    np.random.seed(12345)
    env_p     = create_perturbed_env(env_name, perturb=None)
    obs2, info2 = env_p.reset()

    # compare raw observations and info
    assert obs1.dtype == obs2.dtype
    assert obs1.shape == obs2.shape
    assert np.allclose(obs1, obs2)
    assert info1 == info2

@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_step_equivalence_gym_make(env_name):
    # --------------------------------------------------
    # 1) seed BEFORE reset
    # 2) reset both envs with the same seed
    # 3) step through with identical actions,
    #    but wrap each pair of env.step() calls
    #    with get_state / set_state so that
    #    any np.random calls inside step()
    #    are drawn identically.
    # --------------------------------------------------
    # a) seed & reset "plain" env
    np.random.seed(54321)
    # env   = create_env(env_name)
    env_id = f"Gameworld-{env_name}-v0"
    env = gymnasium.make(env_id)
    obs1, info1 = env.reset()

    # b) same seed & reset "perturbed" env
    np.random.seed(54321)
    # env_p = create_perturbed_env(env_name, perturb=None)
    env_p = gymnasium.make(env_id, perturb='shape', perturb_step=5000)
    obs2, info2 = env_p.reset()

    # sanity check initial state
    assert np.array_equal(obs1, obs2)
    assert info1 == info2

    # we'll draw actions from a separate RNG so it doesn't clash
    action_rng = np.random.RandomState(2025)

    for _ in range(100):
        action = int(action_rng.randint(env.action_space.n))

        # save the global RNG state
        state_before = np.random.get_state()
        # step the plain env
        out1 = env.step(action)
        # capture the state AFTER stepping plain env
        state_after = np.random.get_state()

        # restore RNG so perturbed sees the same draws
        np.random.set_state(state_before)
        out2 = env_p.step(action)
        # now restore RNG to after‐step state so the next iteration
        # continues the original sequence
        np.random.set_state(state_after)

        # unpack and compare everything
        obs1, rew1, done1, trunc1, info1 = out1
        obs2, rew2, done2, trunc2, info2 = out2

        assert obs1.dtype == obs2.dtype
        assert obs1.shape == obs2.shape
        assert np.array_equal(obs1, obs2)

        assert rew1    == rew2
        assert done1   == done2
        assert trunc1  == trunc2
        assert info1   == info2

        if done1:
            break

@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_step_equivalence_gameworld_create(env_name):
    # --------------------------------------------------
    # 1) seed BEFORE reset
    # 2) reset both envs with the same seed
    # 3) step through with identical actions,
    #    but wrap each pair of env.step() calls
    #    with get_state / set_state so that
    #    any np.random calls inside step()
    #    are drawn identically.
    # --------------------------------------------------
    # a) seed & reset "plain" env
    np.random.seed(54321)
    env   = create_env(env_name)
    obs1, info1 = env.reset()

    # b) same seed & reset "perturbed" env
    np.random.seed(54321)
    env_p = create_perturbed_env(env_name, perturb=None)
    obs2, info2 = env_p.reset()

    # sanity check initial state
    assert np.array_equal(obs1, obs2)
    assert info1 == info2

    # we'll draw actions from a separate RNG so it doesn't clash
    action_rng = np.random.RandomState(2025)

    for _ in range(100):
        action = int(action_rng.randint(env.action_space.n))

        # save the global RNG state
        state_before = np.random.get_state()
        # step the plain env
        out1 = env.step(action)
        # capture the state AFTER stepping plain env
        state_after = np.random.get_state()

        # restore RNG so perturbed sees the same draws
        np.random.set_state(state_before)
        out2 = env_p.step(action)
        # now restore RNG to after‐step state so the next iteration
        # continues the original sequence
        np.random.set_state(state_after)

        # unpack and compare everything
        obs1, rew1, done1, trunc1, info1 = out1
        obs2, rew2, done2, trunc2, info2 = out2

        assert obs1.dtype == obs2.dtype
        assert obs1.shape == obs2.shape
        assert np.array_equal(obs1, obs2)

        assert rew1    == rew2
        assert done1   == done2
        assert trunc1  == trunc2
        assert info1   == info2

        if done1:
            break

