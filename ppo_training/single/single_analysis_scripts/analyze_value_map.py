import torch
import numpy as np
import matplotlib.pyplot as plt

from ppo_training.ppo_brain import PPOBrain

RESULTS_PATH = rf'C:\Users\saar.nehemia\PycharmProjects\Bereshit\ppo_training\single\results'

def plot_critic_value_map(results_folder: str, update_milestone: int,
                          to_show: bool = True, to_save: bool = True):
    input_dim = 5
    action_dim = 2
    model = PPOBrain(input_dim, action_dim)

    # Load the final trained model
    results_full_folder = f"{RESULTS_PATH}\{results_folder}"
    filepath = rf"{results_full_folder}\checkpoints\brain_update_{update_milestone:03d}.pth"
    try:
        model.load_state_dict(torch.load(filepath))
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}")
        return

    # Create a grid of artificial inputs
    grid_res = 50
    # Sweep distance from Close (-1.0) to Far (1.0)
    dist_sweep = np.linspace(-1.0, 1.0, grid_res)
    # Sweep angle from Left (-1.0) to Right (1.0)
    angle_sweep = np.linspace(-1.0, 1.0, grid_res)

    X, Y = np.meshgrid(angle_sweep, dist_sweep)
    Z = np.zeros_like(X)

    # Constant states for the sweep
    food_seen = 1.0
    norm_vel = 0.0  # Stationary
    norm_energy = 0.5  # Healthy energy

    for i in range(grid_res):
        for j in range(grid_res):
            state = torch.FloatTensor([[
                food_seen,
                Y[i, j],  # Distance
                X[i, j],  # Angle
                norm_vel,
                norm_energy
            ]])

            with torch.no_grad():
                value = model.critic(state).item()
            Z[i, j] = value

    # Plotting a heatmap contour
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Z, levels=30, cmap='viridis')
    plt.colorbar(contour, label='Critic Value (Expected Reward)')

    plt.title(f"Critic Spatial Value Map (Brain, Update {update_milestone})")
    plt.xlabel("Target Angle (Left to Right)")
    plt.ylabel("Target Distance (Close to Far)")

    plt.tight_layout()

    if to_save:
        filename = f'critic_value_map_update{update_milestone:03d}.png'
        plt.savefig(fr"{results_full_folder}\{filename}", dpi=150)
        print(f"Saved plot to {filename}")

    if to_show:
        plt.show()


if __name__ == "__main__":
    results_folder = 'single_results_eating_reward'
    update_milestone = 20
    plot_critic_value_map(results_folder=results_folder, update_milestone=update_milestone,
                          to_show=False, to_save=True)
