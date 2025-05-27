from gameworld.envs.base.aviate import Aviate
from gameworld.envs.base.bounce import Bounce
from gameworld.envs.base.cross import Cross
from gameworld.envs.base.drive import Drive
from gameworld.envs.base.explode import Explode
from gameworld.envs.base.fruits import Fruits
from gameworld.envs.base.gold import Gold
from gameworld.envs.base.hunt import Hunt
from gameworld.envs.base.impact import Impact
from gameworld.envs.base.jump import Jump


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
