import torch
import numpy as np
import matplotlib.pyplot as plt

from ppo_training.ppo_brain import PPOBrain

if __name__ == '__main__':
    # Load model
    plot_layer = 2
    update_milestone = 300
    model_path = fr"C:\Users\saar.nehemia\PycharmProjects\Bereshit\ppo_training\single\results\single_results_eating_reward\checkpoints\brain_update_{update_milestone:03d}.pth"

    # 1. Create the Grid for Distance and Angle
    res = 40
    dist_range = np.linspace(-1, 1, res)
    angle_range = np.linspace(-1, 1, res)
    D, A = np.meshgrid(dist_range, angle_range)

    # Prepare the 5D input state: [food_flag, dist, angle, velocity, energy]
    # Shape of grid_2d is (1600, 2)
    grid_2d = np.vstack([D.ravel(), A.ravel()]).T

    # Create a container for 5 inputs
    full_input_np = np.zeros((grid_2d.shape[0], 5))

    # Fill the slots based on obs = [food_seen, norm_dist, norm_angle, norm_vel, norm_energy]:
    full_input_np[:, 0] = 1.0  # food_seen_flag (Assume target is visible)
    full_input_np[:, 1] = grid_2d[:, 0]  # target_dist (Varying)
    full_input_np[:, 2] = grid_2d[:, 1]  # target_angle (Varying)
    full_input_np[:, 3] = 0.0  # velocity (Assume standing still)
    full_input_np[:, 4] = 1.0  # energy (Assume full health)

    full_input = torch.tensor(full_input_np, dtype=torch.float32)

    # 2. Extract Activations from the desired layer (e.g., Layer 2)
    model = PPOBrain(5, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        h1_act = model.actor[1](model.actor[0](full_input))
        h2_act = model.actor[3](model.actor[2](h1_act))
        if plot_layer == 1:
            activations = h2_act.numpy()  # Shape: (res*res, 64)
        elif plot_layer == 2:
            activations = h2_act.numpy()  # Shape: (res*res, 64)

    # 3. Plotting the 8x8 Grid
    fig, axes = plt.subplots(8, 8, figsize=(20, 20))
    fig.suptitle(f"Layer {plot_layer} - Neurons Activity Heatmaps, Update {update_milestone:03d} (X=Dist, Y=Angle)", fontsize=24)

    for i in range(64):
        ax = axes[i // 8, i % 8]

        # Reshape the flat activations for neuron 'i' back into the grid shape
        neuron_grid = activations[:, i].reshape(res, res)

        # Plotting
        im = ax.imshow(neuron_grid, extent=[-1, 1, -1, 1], origin='lower',
                       cmap='magma', aspect='auto')

        ax.set_title(f"Neuron {i}", fontsize=10)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
