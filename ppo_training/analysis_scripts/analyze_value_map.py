import torch
import numpy as np
import matplotlib.pyplot as plt

from ppo_training.ppo_brain import PPOBrain


def plot_critic_value_map():
    input_dim = 5
    action_dim = 2
    model = PPOBrain(input_dim, action_dim)

    # Load the final trained model
    filepath = r"C:\Users\saar.nehemia\PycharmProjects\Bereshit\ppo_training\eating_reward\checkpoints\brain_update_300.pth"
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

    plt.title("Critic Spatial Value Map (Trained Brain)")
    plt.xlabel("Target Angle (Left to Right)")
    plt.ylabel("Target Distance (Close to Far)")

    plt.tight_layout()
    plt.savefig("critic_value_map.png", dpi=150)
    print("Saved plot to 'critic_value_map.png'")
    plt.show()


if __name__ == "__main__":
    plot_critic_value_map()
