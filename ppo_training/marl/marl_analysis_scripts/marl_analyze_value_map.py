"""
CRITIC SPATIAL VALUE MAP
------------------------
Sweeps a 'Ghost Food' or 'Ghost Agent' across the FOV to see how much
future reward the Critic network expects.

What is GOOD to see:
- A clear 'Peak' at Angle 0 and Distance -1.0 (Close). This shows the agent
  understands that food/prey is most valuable when it's right in front of the nose.
- Symmetrical gradients. The value should drop off as the target moves to
  the periphery (±1.0 Angle).

What is NOT GOOD to see:
- 'Value Islands': Random bright spots in corners. This indicates 'Overfitting'
  to specific coordinates rather than learning a general spatial rule.
- Flat Heatmap: If the entire map is one color, the Critic hasn't learned
  the difference between a full stomach and starvation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from ppo_training.ppo_brain import PPOBrain


def plot_critic_value_map_food(results_folder: str, species: str,
                               to_show: bool = True, to_save: bool = True):
    """

    :param results_folder:
    :param species: 'herb' / 'carn'
    :return:
    """
    input_dim = 13
    action_dim = 3
    model = PPOBrain(input_dim, action_dim)

    # Find the latest checkpoint dynamically
    results_full_folder = rf"C:\Users\saar.nehemia\PycharmProjects\Bereshit\ppo_training\marl\{results_folder}"
    latest_update = max(
        [int(f.split('_')[-1].split('.')[0]) for f in
         os.listdir(rf'{results_full_folder}\marl_checkpoints') if
         f.startswith(species)],
        default=0)
    filepath = fr"{results_full_folder}\marl_checkpoints/{species}_brain_{latest_update:03d}.pth"

    try:
        model.load_state_dict(torch.load(filepath))
        print(f"Loaded {filepath}")
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}")
        return

    grid_res = 50
    dist_sweep = np.linspace(-1.0, 1.0, grid_res)  # Close to Far
    angle_sweep = np.linspace(-1.0, 1.0, grid_res)  # Left to Right

    X, Y = np.meshgrid(angle_sweep, dist_sweep)
    Z = np.zeros_like(X)

    # Constant 'Neutral' States (Normalized)
    norm_energy = 0.0  # Medium Energy
    norm_health = 1.0  # Full Health
    norm_mass = 0.0  # Average Mass
    norm_strength = 0.0  # Average Strength
    norm_vel = -1.0  # Stationary

    # Constant Agent Vision (No agents seen)
    agent_seen, a_dist, a_angle, a_type, a_mass = 0.0, 1.0, 0.0, 0.0, 0.0

    for i in range(grid_res):
        for j in range(grid_res):
            state = torch.FloatTensor([[
                norm_energy, norm_health, norm_mass, norm_strength, norm_vel,  # Self
                1.0, Y[i, j], X[i, j],  # Food
                agent_seen, a_dist, a_angle, a_type, a_mass  # Other Agents
            ]])

            with torch.no_grad():
                value = model.critic(state).item()
            Z[i, j] = value

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Z, levels=30, cmap='viridis')
    plt.colorbar(contour, label='Critic Value (Expected Reward)')

    plt.title(f"Critic Spatial Value Map - Food ({species.capitalize()} Brain, Update {latest_update})")
    plt.xlabel("Food Angle (Left to Right)")
    plt.ylabel("Food Distance (Close to Far)")

    plt.tight_layout()

    if to_save:
        filename = f"{results_full_folder}\critic_value_map_food_{species}.png"
        plt.savefig(filename, dpi=150)
        print(f"Saved plot to '{filename}'")

    if to_show:
        plt.show()


def plot_critic_value_map_creature(results_folder: str, species: str,
                                   to_show: bool = True, to_save: bool = True):
    """
    :param results_folder:
    :param species: 'herb' / 'carn'
    :param to_show:
    :param to_save:
    :return:
    """
    input_dim = 13
    action_dim = 3
    model = PPOBrain(input_dim, action_dim)

    # Find the latest checkpoint dynamically
    results_full_folder = rf"C:\Users\saar.nehemia\PycharmProjects\Bereshit\ppo_training\marl\{results_folder}"
    latest_update = max(
        [int(f.split('_')[-1].split('.')[0]) for f in
         os.listdir(rf'{results_full_folder}\marl_checkpoints') if
         f.startswith(species)],
        default=0)
    filepath = fr"{results_full_folder}\marl_checkpoints/{species}_brain_{latest_update:03d}.pth"

    try:
        model.load_state_dict(torch.load(filepath))
        print(f"Loaded {filepath}")
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}")
        return

    grid_res = 50
    dist_sweep = np.linspace(-1.0, 1.0, grid_res)  # Close to Far
    angle_sweep = np.linspace(-1.0, 1.0, grid_res)  # Left to Right

    X, Y = np.meshgrid(angle_sweep, dist_sweep)
    Z = np.zeros_like(X)

    # Constant 'Neutral' States (Normalized)
    norm_energy = 0.0  # Medium Energy
    norm_health = 1.0  # Full Health
    norm_mass = 0.0  # Average Mass
    norm_strength = 0.0  # Average Strength
    norm_vel = -1.0  # Stationary

    # Constant Food vision (no food seen)
    food_seen, food_dist, food_angle = 0.0, 1.0, 0.0

    fig, ax = plt.subplots(figsize=(8, 6), nrows=2, ncols=2)
    ax = ax.flatten()
    i_ax = 0
    for a_type in [-1.0, 1.0]:
        for a_mass in [-1.0, 1.0]:
            for i in range(grid_res):
                for j in range(grid_res):
                    state = torch.FloatTensor([[
                        norm_energy, norm_health, norm_mass, norm_strength, norm_vel,  # Self
                        food_seen, food_dist, food_angle,  # Food
                        1.0, Y[i, j], X[i, j], a_type, a_mass  # Other Agents
                    ]])

                    with torch.no_grad():
                        value = model.critic(state).item()
                    Z[i, j] = value

            contour = ax[i_ax].contourf(X, Y, Z, levels=30, cmap='viridis')
            cbar = fig.colorbar(contour, ax=ax[i_ax], orientation='vertical', pad=0.05)
            cbar.set_label('Critic Value (Expected Reward)')
            ax[i_ax].set_title(f"{'Tiny' if a_mass == -1.0 else 'Huge'} "
                               f"{'Herb' if a_type == -1.0 else 'Carn'}")
            ax[i_ax].set_xlabel("Creature Angle (Left to Right)")
            ax[i_ax].set_ylabel("Creature Distance (Close to Far)")
            i_ax += 1

    fig.suptitle(f"Critic Spatial Value Map - Creature ({species.capitalize()} Brain, Update {latest_update})")
    plt.tight_layout()

    if to_save:
        filename = f"{results_full_folder}\critic_value_map_creature_{species}.png"
        plt.savefig(filename, dpi=150)
        print(f"Saved plot to '{filename}'")

    if to_show:
        plt.show()


if __name__ == "__main__":
    species = 'carn'
    results_folder = "marl_results_500_ent005_ex_eat_dist_reward"
    plot_critic_value_map_creature(results_folder=results_folder, species=species,
                                   to_show=True, to_save=False)
