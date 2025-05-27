from gameworld.envs.perturbed.aviate import Aviate
from gameworld.envs.perturbed.bounce import Bounce
from gameworld.envs.perturbed.cross import Cross
from gameworld.envs.perturbed.drive import Drive
from gameworld.envs.perturbed.explode import Explode
from gameworld.envs.perturbed.fruits import Fruits
from gameworld.envs.perturbed.gold import Gold
from gameworld.envs.perturbed.hunt import Hunt
from gameworld.envs.perturbed.impact import Impact
from gameworld.envs.perturbed.jump import Jump


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
