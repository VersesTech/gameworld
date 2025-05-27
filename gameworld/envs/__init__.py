from gymnasium.envs.registration import register

from gameworld.envs.aviate import Aviate
from gameworld.envs.bounce import Bounce
from gameworld.envs.cross import Cross
from gameworld.envs.drive import Drive
from gameworld.envs.explode import Explode
from gameworld.envs.fruits import Fruits
from gameworld.envs.gold import Gold
from gameworld.envs.hunt import Hunt
from gameworld.envs.impact import Impact
from gameworld.envs.jump import Jump


# Registering the environments
game_names = [
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
for game in game_names:
    register(
        id=f"GameWorld-{game}-v0",
        entry_point=f"gameworld.envs:{game}"
        )

def create_gameworld_env(game, **kwargs):
    if game == "Aviate":
        return Aviate(**kwargs)
    elif game == "Bounce":
        return Bounce(**kwargs)
    elif game == "Cross":
        return Cross(**kwargs)
    elif game == "Drive":
        return Drive(**kwargs)
    elif game == "Explode":
        return Explode(**kwargs)
    elif game == "Fruits":
        return Fruits(**kwargs)
    elif game == "Gold":
        return Gold(**kwargs)
    elif game == "Hunt":
        return Hunt(**kwargs)
    elif game == "Impact":
        return Impact(**kwargs)
    elif game == "Jump":
        return Jump(**kwargs)
    else:
        raise Exception(f"Unsupported game in the gameworld set: {game}")
