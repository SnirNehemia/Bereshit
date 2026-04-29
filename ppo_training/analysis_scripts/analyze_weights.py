import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from ppo_training.ppo_brain import PPOBrain


def analyze_synaptic_attention():
    # Model dimensions based on the Bounded Radar environment
    input_dim = 5
    action_dim = 2

    input_labels = ['Food Seen (Flag)', 'Norm Dist', 'Norm Angle', 'Norm Vel', 'Norm Energy']
    updates = list(range(0, 300, 10))  # Assuming 150 total updates, saved every 10

    # Store the mean absolute weight for each input across all updates
    weight_history = {label: [] for label in input_labels}
    valid_updates = []

    model = PPOBrain(input_dim, action_dim)

    for update in updates:
        filepath = fr"C:\Users\saar.nehemia\PycharmProjects\Bereshit\ppo_training\eating_reward\checkpoints/brain_update_{update:03d}.pth"
        if not os.path.exists(filepath):
            continue

        model.load_state_dict(torch.load(filepath))
        valid_updates.append(update)

        # Access the first linear layer of the Actor: nn.Sequential(nn.Linear, nn.Tanh, ...)
        # model.actor[0].weight has shape [64, 5]
        first_layer_weights = model.actor[0].weight.detach().numpy()

        for i, label in enumerate(input_labels):
            # Calculate the mean absolute magnitude of connections from input 'i' to the hidden layer
            mean_weight = np.mean(np.abs(first_layer_weights[:, i]))
            weight_history[label].append(mean_weight)

    # Plotting
    plt.figure(figsize=(10, 6))
    for label in input_labels:
        plt.plot(valid_updates, weight_history[label], label=label, linewidth=2, marker='o')

    plt.title("Synaptic Attention: 1st Layer Weight Magnitudes Over Time")
    plt.xlabel("Training Update")
    plt.ylabel("Mean Absolute Synaptic Weight")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("synaptic_attention.png", dpi=150)
    print("Saved plot to 'synaptic_attention.png'")
    plt.show()


if __name__ == "__main__":
    analyze_synaptic_attention()
