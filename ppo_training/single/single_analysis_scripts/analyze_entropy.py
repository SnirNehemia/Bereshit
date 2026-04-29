import torch
import matplotlib.pyplot as plt
import os

from ppo_training.ppo_brain import PPOBrain

RESULTS_PATH = rf'C:\Users\saar.nehemia\PycharmProjects\Bereshit\ppo_training\single\results'


def analyze_exploration_decay(results_folder: str,
                              to_show: bool = True, to_save: bool = True):
    input_dim = 5
    action_dim = 2
    updates = list(range(0, 300, 10))

    std_accel_history = []
    std_turn_history = []
    valid_updates = []

    model = PPOBrain(input_dim, action_dim)

    for update in updates:
        filepath = rf"{RESULTS_PATH}\{results_folder}\checkpoints\brain_update_{update:03d}.pth"
        if not os.path.exists(filepath):
            continue

        model.load_state_dict(torch.load(filepath))
        valid_updates.append(update)

        # log_std is a trainable parameter of shape [1, 2]
        # We take the exponent to convert log(std) back to actual standard deviation
        current_std = torch.exp(model.log_std).detach().numpy()[0]

        std_accel_history.append(current_std[0])
        std_turn_history.append(current_std[1])

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(valid_updates, std_accel_history, label='Accel/Brake Uncertainty ($\sigma$)', color='blue', linewidth=2)
    plt.plot(valid_updates, std_turn_history, label='Turning Uncertainty ($\sigma$)', color='red', linewidth=2)

    plt.title("Exploration vs. Exploitation (Action Standard Deviation)")
    plt.xlabel("Training Update")
    plt.ylabel("Action Uncertainty (Standard Deviation)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    if to_save:
        plt.savefig("exploration_decay.png", dpi=150)
        print("Saved plot to 'exploration_decay.png'")

    if to_show:
        plt.show()


if __name__ == "__main__":
    results_folder = 'single_results_eating_reward'
    analyze_exploration_decay()
