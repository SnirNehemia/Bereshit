from creature import Creature


def plot_traits_scatter(ax,
                        creatures: dict[int, Creature],
                        trait_x: str, trait_y: str,
                        trait_x_min: float = 0, trait_x_max: float = 2,
                        trait_y_min: float = 0, trait_y_max: float = 0.2,
                        marker_size: float = 50):
    # Update the mass and height of each creature (example dynamic changes)
    traits_x = [getattr(creature, trait_x) for creature in creatures.values()]
    traits_y = [getattr(creature, trait_y) for creature in creatures.values()]
    colors = [creature.color for creature in creatures.values()]

    # Clear the previous scatter plot and create a new one
    ax.clear()
    ax.set_xlim(trait_x_min, trait_x_max)
    ax.set_ylim(trait_y_min, trait_y_max)
    ax.set_xlabel(trait_x)
    ax.set_ylabel(trait_y)
    traits_scat = ax.scatter(traits_x, traits_y, c=colors, s=marker_size)
    return traits_scat
