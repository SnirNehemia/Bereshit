"""
EXPLORATION VS. EXPLOITATION (LOG-STD) PROBE
-------------------------------------------
Analyzes the standard deviation (sigma) of the action distributions.

What is GOOD to see:
- A smooth, downward slope: This indicates the agent is becoming more
  confident in its decisions.
- Accel vs. Turn vs. Attack differentiation: It's normal for the brain to
  master 'Turning' before it masters 'Attacking'. You should see the
  uncertainty for turning drop first, followed later by the attack signal.
- Non-zero plateau: Biological agents should never reach 0.0 uncertainty.
  A small amount of remaining noise allows for adaptation if the
  environment changes.

What is NOT GOOD to see:
- 'Flat' high line: The agent isn't learning anything; it's staying random.
- 'Instant' drop to zero: The agent has collapsed into a single behavior
  (like spinning in circles) and has stopped learning entirely (Pre-convergence).
- Herbivore 'Attack' uncertainty staying high: If a herbivore is still
  'curious' about attacking after 100 updates, your reward signal isn't
  punishing useless energy expenditure enough.
"""

import torch
import matplotlib.pyplot as plt
import os

from ppo_training.ppo_brain import PPOBrain

RESULTS_PATH = rf'C:\Users\saar.nehemia\PycharmProjects\Bereshit\ppo_training\marl\results'


def analyze_exploration_decay(results_folder: str, species: str,
                              to_show: bool = True, to_save: bool = True):
    """

    :param results_folder:
    :param species: 'herb' / 'carn'
    :return:
    """
    input_dim = 13
    action_dim = 3
    updates = list(range(0, 510, 10))

    std_accel_history = []
    std_turn_history = []
    std_attack_history = []
    valid_updates = []

    model = PPOBrain(input_dim, action_dim)

    results_full_folder = rf"{RESULTS_PATH}\{results_folder}"
    for update in updates:
        filepath = fr"{results_full_folder}\marl_checkpoints\{species}_brain_{update:03d}.pth"
        if not os.path.exists(filepath):
            continue

        model.load_state_dict(torch.load(filepath))
        valid_updates.append(update)

        current_std = torch.exp(model.log_std).detach().numpy()[0]

        std_accel_history.append(current_std[0])
        std_turn_history.append(current_std[1])
        std_attack_history.append(current_std[2])  # The new attack dimension

    plt.figure(figsize=(8, 5))
    plt.plot(valid_updates, std_accel_history, label='Accel/Brake ($\sigma$)', color='blue', linewidth=2)
    plt.plot(valid_updates, std_turn_history, label='Turning ($\sigma$)', color='red', linewidth=2)
    plt.plot(valid_updates, std_attack_history, label='Attack Choice ($\sigma$)', color='green', linewidth=2)

    plt.title(f"Exploration vs. Exploitation ({species.capitalize()} Brain)")
    plt.xlabel("Training Update")
    plt.ylabel("Action Uncertainty (Standard Deviation)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    if to_save:
        filename = f"{results_full_folder}\exploration_decay_{species}.png"
        plt.savefig(filename, dpi=150)
        print(f"Saved plot to '{filename}'")

    if to_show:
        plt.show()


if __name__ == "__main__":
    species = 'carn'
    results_folder = "marl_results_500_ent001"
    analyze_exploration_decay(results_folder=results_folder, species=species,
                              to_show=True, to_save=False)
