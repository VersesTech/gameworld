import importlib
from gymnasium.envs.registration import register

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

def create_gameworld_env(game, *, perturb=None, perturb_step=5000, **kwargs):
    """
    A single factory for all of our GameWorld-<Game>-v0 envs.
    It always gets called with:
      - game:     one of the strings in GAME_NAMES
      - perturb:  None or "color"/"shape"
      - perturb_step: int step at which to trigger
      - **kwargs → any other args you want to forward to the env ctor
    """

    # dynamically import the base class
    module_base = importlib.import_module(f"gameworld.envs.base.{game.lower()}")
    BaseCls    = getattr(module_base, game)

    # if no perturb requested, just return the base
    if perturb in (None, "None"):
        return BaseCls(**kwargs)

    # otherwise import & return the perturbed subclass
    module_pert = importlib.import_module(f"gameworld.envs.perturbed.{game.lower()}")
    PertCls     = getattr(module_pert, game)
    return PertCls(perturb=perturb, perturb_step=perturb_step, **kwargs)


# register every Gameworld-<Game>-v0 to factory,
# passing `game=<Game>` by default.  Any user‐supplied
# kwargs (like perturb="color") will override these.
for game in GAME_NAMES:
    register(
        id=f"Gameworld-{game}-v0",
        entry_point="gameworld.envs:create_gameworld_env",
        kwargs={"game": game},
    )
