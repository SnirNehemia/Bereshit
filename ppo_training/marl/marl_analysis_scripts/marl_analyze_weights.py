"""
SYNAPTIC ATTENTION PROBE
------------------------
Analyzes the absolute weight magnitudes of the first layer in the Actor network.
This reveals the 'Sensory Priority' the species has evolved.

What is GOOD to see:
- Herbivores: High weights for 'Agent Seen', 'A-Dist', and 'A-Type'. This shows they
  are paying attention to predators.
- Carnivores: High weights for 'A-Mass' and 'Health'. This suggests they are
  evaluating if a target is worth the risk of recoil damage.
- Both: Declining weights for 'Constant' inputs or irrelevant data.

What is NOT GOOD to see:
- 'Flat' lines: If all 13 inputs have the same weight, the brain is 'Blind'
  and acting randomly.
- Zero weight for 'Energy': If the agent isn't looking at its own energy,
  it won't learn homeostatic regulation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from ppo_training.ppo_brain import PPOBrain

RESULTS_PATH = rf'C:\Users\saar.nehemia\PycharmProjects\Bereshit\ppo_training\marl\results'

def analyze_synaptic_attention(results_folder: str, species: str,
                               to_show: bool = True, to_save: bool = True):
    """

    :param results_folder:
    :param species: 'herb' / 'carn'
    :return:
    """
    input_dim = 13
    action_dim = 3

    input_labels = [
        'Energy', 'Health', 'Mass', 'Strength', 'Vel',
        'Food Seen', 'F-Dist', 'F-Angle',
        'Agent Seen', 'A-Dist', 'A-Angle', 'A-Type', 'A-Mass'
    ]

    updates = list(range(0, 510, 10))
    weight_history = {label: [] for label in input_labels}
    valid_updates = []

    model = PPOBrain(input_dim, action_dim)

    results_full_folder = rf"{RESULTS_PATH}\{results_folder}"
    for update in updates:
        filepath = fr"{results_full_folder}\marl_checkpoints\{species}_brain_{update:03d}.pth"
        if not os.path.exists(filepath):
            continue

        model.load_state_dict(torch.load(filepath))
        valid_updates.append(update)

        # Access the first linear layer of the Actor
        first_layer_weights = model.actor[0].weight.detach().numpy()

        for i, label in enumerate(input_labels):
            mean_weight = np.mean(np.abs(first_layer_weights[:, i]))
            weight_history[label].append(mean_weight)

    plt.figure(figsize=(12, 7))

    # Plot with distinct styles to separate 'Self', 'Food', and 'Agent' vision
    for i, label in enumerate(input_labels):
        if i < 5:
            fmt = '--'  # Self stats
        elif i < 8:
            fmt = '-'  # Food Vision
        else:
            fmt = '-.'  # Agent Vision

        plt.plot(valid_updates, weight_history[label], linestyle=fmt, label=label, linewidth=2)

    plt.title(f"Synaptic Attention ({species.capitalize()} Brain) - 1st Layer Weights")
    plt.xlabel("Training Update")
    plt.ylabel("Mean Absolute Synaptic Weight")

    # Put legend outside the plot since there are 13 lines
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if to_save:
        filename = f"{results_full_folder}\synaptic_attention_{species}.png"
        plt.savefig(filename, dpi=150)
        print(f"Saved plot to '{filename}'")

    if to_show:
        plt.show()


if __name__ == "__main__":
    species = 'carn'
    results_folder = "marl_results_500_ent001"
    analyze_synaptic_attention(results_folder=results_folder, species=species)
