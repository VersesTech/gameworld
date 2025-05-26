from gameworld.perturbed_envs.aviate import Aviate
from gameworld.perturbed_envs.bounce import Bounce
from gameworld.perturbed_envs.cross import Cross
from gameworld.perturbed_envs.drive import Drive
from gameworld.perturbed_envs.explode import Explode
from gameworld.perturbed_envs.fruits import Fruits
from gameworld.perturbed_envs.gold import Gold
from gameworld.perturbed_envs.hunt import Hunt
from gameworld.perturbed_envs.impact import Impact
from gameworld.perturbed_envs.jump import Jump


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
