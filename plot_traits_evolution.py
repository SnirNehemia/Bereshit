import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
from collections import defaultdict


# Example Creature class definition (with multiple traits)
class Creature:
    def __init__(self, gen, parent, birth_frame, **traits):
        self.gen = gen
        self.parent = parent
        self.birth_frame = birth_frame
        self.traits = traits  # Dictionary of trait names & values

    def mutate(self):
        # Apply small random mutations to each trait
        for trait in self.traits:
            self.traits[trait] *= random.uniform(0.95, 1.05)


# Example creature data (more traits added)
creatures = {
    "A": Creature(gen=0, parent=None, birth_frame=0, size=10, strength=5, speed=2, intelligence=3, endurance=4),
    "B": Creature(gen=1, parent="A", birth_frame=10, size=12, strength=6, speed=3, intelligence=3.5, endurance=4.2),
    "C": Creature(gen=1, parent="A", birth_frame=12, size=11, strength=5.5, speed=2.5, intelligence=3.2, endurance=4.1),
    "D": Creature(gen=2, parent="B", birth_frame=20, size=13, strength=7, speed=3.5, intelligence=3.8, endurance=4.5),
    "E": Creature(gen=2, parent="B", birth_frame=22, size=12.5, strength=6.5, speed=3, intelligence=3.6, endurance=4.3),
    "F": Creature(gen=3, parent="D", birth_frame=30, size=14, strength=8, speed=4, intelligence=4, endurance=4.8),
}

# Mutate creatures for a few generations
for creature in creatures.values():
    creature.mutate()
    creature.mutate()

# Collect traits by generation
generations = defaultdict(list)
for creature_id, creature in creatures.items():
    generations[creature.gen].append(creature)

# Prepare data for the animation (average of attributes per generation)
gen_trait_averages = []

# This will hold data for each generation's average traits
all_trait_names = set()  # Collect all possible trait names
for gen in sorted(generations.keys()):
    trait_sums = defaultdict(float)
    count = len(generations[gen])

    # Sum trait values for all creatures in this generation
    for creature in generations[gen]:
        for trait, value in creature.traits.items():
            trait_sums[trait] += value
            all_trait_names.add(trait)  # Track all trait names

    # Compute averages
    trait_averages = {trait: (trait_sums[trait] / count) for trait in all_trait_names}
    gen_trait_averages.append(trait_averages)

# Convert data into a format suitable for animation
sorted_generations = sorted(generations.keys())
num_generations = len(sorted_generations)

# Set up the figure for the animated bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.6  # Width of bars


def update(frame):
    ax.clear()  # Clear the axis for each frame

    if frame >= len(gen_trait_averages):
        return  # Stop if frame exceeds available data

    trait_values = gen_trait_averages[frame]

    # Sort trait names to maintain consistent order
    sorted_traits = sorted(trait_values.keys())
    sorted_values = [trait_values[trait] for trait in sorted_traits]

    ax.bar(sorted_traits, sorted_values, color='lightblue', width=bar_width)

    # Labels and title
    ax.set_xlabel("Traits")
    ax.set_ylabel("Average Value")
    ax.set_title(f"Trait Evolution - Generation {sorted_generations[frame]}")
    ax.set_ylim(0, max(max(values) for values in gen_trait_averages) * 1.2)  # Dynamic y-axis
    ax.set_xticklabels(sorted_traits, rotation=30, ha='right')

    plt.tight_layout()


# Create the animation
ani = animation.FuncAnimation(fig, update, frames=num_generations, repeat=False)

# Save the animation (try both MP4 and GIF)
try:
    ani.save('trait_evolution.mp4', writer='ffmpeg', fps=5)
    print("Animation saved as 'trait_evolution.mp4'")
except Exception as e:
    print(f"Error saving as .mp4: {e}")
    try:
        ani.save('trait_evolution.gif', writer='imagemagick', fps=5)
        print("Animation saved as 'trait_evolution.gif'")
    except Exception as e:
        print(f"Error saving as .gif: {e}")

# Show the animation
plt.show()
