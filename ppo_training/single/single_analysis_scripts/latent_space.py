import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from ppo_training.ppo_brain import PPOBrain

if __name__ == '__main__':
    # Load model
    plot_layer = 2
    color_by = ['Distance', 'Angle'][1]
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

    # Fill the slots based on obs=[food_seen, norm_dist, norm_angle, norm_vel, norm_energy]:
    full_input_np[:, 0] = 1.0  # food_seen_flag (Assume target is visible)
    full_input_np[:, 1] = grid_2d[:, 0]  # target_dist (Varying)
    full_input_np[:, 2] = grid_2d[:, 1]  # target_angle (Varying)
    full_input_np[:, 3] = 0.0  # velocity (Assume standing still)
    full_input_np[:, 4] = 1.0  # energy (Assume full health)

    full_input = torch.tensor(full_input_np, dtype=torch.float32)

    # 2. Load Model and Extract Activations
    model = PPOBrain(5, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        # Pass through first block: Linear -> Tanh
        h1 = model.actor[0](full_input)
        h1_act = model.actor[1](h1)

        # Pass through second block: Linear -> Tanh (This is often the richest representation)
        h2 = model.actor[2](h1_act)
        h2_act = model.actor[3](h2)

        # Final Output: Linear (Speed/Direction)
        actions = model.actor[4](h2_act)

    # Convert to numpy for analysis
    if plot_layer == 1:
        representations = h1_act.numpy()
    elif plot_layer == 2:
        representations = h2_act.numpy()
    outputs = actions.numpy()

    # 3. PCA and Visualization
    pca = PCA(n_components=2)
    components = pca.fit_transform(representations)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: The Input Map (How the brain "sees" the target angle)
    color_by_idx = 0 if color_by == 'Distance' else 1
    im1 = ax[0].scatter(components[:, 0], components[:, 1],
                        c=grid_2d[:, color_by_idx], cmap='hsv', s=10)
    ax[0].set_title(
        f"Hidden Layer {plot_layer} Manifold\nColored by Target {color_by}, Update {update_milestone:03d} ({pca.explained_variance_ratio_.sum():.2%} Var)")
    plt.colorbar(im1, ax=ax[0], label="Angle")

    # Plot 2: Action Mapping (Responsibility of neurons)
    # Arrows show how the hidden state translates to the physical action
    skip = 5
    factor = 3
    im2 = ax[1].scatter(components[:, 0], components[:, 1], c='lightgrey', s=5, alpha=0.3)
    ax[1].quiver(components[::skip, 0], components[::skip, 1],
                 outputs[::skip, 0] / factor, outputs[::skip, 1] / factor,
                 color='red', alpha=0.7, scale=10)
    ax[1].set_title("Decision Vectors Overlay\n(Where the brain 'pushes' the agent)")

    plt.tight_layout()
    plt.show()
