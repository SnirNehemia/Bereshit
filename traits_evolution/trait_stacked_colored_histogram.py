import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.colors import rgb_to_hsv

from creature import Creature


def trait_stacked_colored_histogram(ax,
                                    creatures: dict[int, Creature],
                                    trait_name: str,
                                    num_bins: int, min_value: float = 0, max_value: float = 0.2):
    """
    Plot stacked colored histogram
    """

    # get trait and color
    traits = [getattr(creature, trait_name) for creature in creatures.values()]
    colors = [creature.color for creature in creatures.values()]

    # Digitize mass values into bins
    bins = np.linspace(min_value, max_value, num_bins + 1)
    bin_indices = np.digitize(traits, bins) - 1

    # Prepare bin storage
    bin_heights = [0] * num_bins
    bin_creatures = [[] for _ in range(num_bins)]  # store (mass, rgb, hue) per bin

    # Group creatures by bin and attach hue for sorting
    num_creatures = len(creatures)
    for i in range(num_creatures):
        bin_idx = bin_indices[i]
        if 0 <= bin_idx < num_bins:
            hue = rgb_to_hsv(np.array([colors[i]]))[0][0]  # hue for sorting
            bin_creatures[bin_idx].append((traits[i], colors[i], hue))

    # Sort creatures by hue in each bin and draw bars
    ax.clear()
    for i in range(num_bins):
        bin_data = sorted(bin_creatures[i], key=lambda x: x[2])  # sort by hue
        for _, rgb_color, _ in bin_data:
            ax.bar(
                bins[i],
                1,
                bottom=bin_heights[i],
                width=bins[1] - bins[0],
                color=rgb_color,
                edgecolor='none',
                align='edge'
            )
            bin_heights[i] += 1

    # Final styling
    ax.set_title("Stacked Colored Histogram (RGB sorted by hue)")
    ax.set_xlabel(trait_name)
    ax.set_ylabel(f"Creature Count (total={num_creatures})")
    ax.set_xlim(min_value, max_value)


if __name__ == '__main__':

    num_creatures = 900
    num_bins = 20
    min_value = 0
    max_value = 30


    class Creature:
        def __init__(self, id, mass, color):
            self.id = id
            self.mass = mass
            self.color = color

    # create creatures
    creatures = dict()
    for creature_id in range(num_creatures):
        creatures[creature_id] = Creature(id=creature_id,
                                          mass=np.clip(np.random.normal(loc=15, scale=5, size=1), 0, None),
                                          color=(random.random(), random.random(), random.random()))

    # plot
    fig, ax = plt.subplots(figsize=(10, 5))

    trait_stacked_colored_histogram(ax=ax,
                                    creatures=creatures,
                                    trait_name='mass',
                                    num_bins=num_bins, min_value=min_value, max_value=max_value)

    plt.tight_layout()
    plt.show()
