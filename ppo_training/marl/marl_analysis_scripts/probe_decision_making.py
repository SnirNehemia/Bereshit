"""
DECISION BOUNDARY PROBE
-----------------------
Specifically tests the 'Fight or Flight' thresholds by simulating
predator/prey encounters.

What is GOOD to see:
- The 'Panic Threshold': In Herbivores, acceleration should stay at 0 until
  the predator is close, then spike to 1.0.
- 'Risk Assessment': In Carnivores, the attack signal should only be positive
  when Prey Mass is LOW and Distance is CLOSE.

What is NOT GOOD to see:
- 'Suicidal Predators': Carnivores attacking 'Huge Mass' targets at far distances.
- 'Frozen Prey': Herbivores that don't increase acceleration even when
  a predator is touching them (The 'Statue' problem).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from ppo_training.ppo_brain import PPOBrain

RESULTS_PATH = rf'C:\Users\saar.nehemia\PycharmProjects\Bereshit\ppo_training\marl\results'


def get_action(brain, obs_array):
    """Helper to pass a synthetic observation to the brain and return deterministic actions."""
    obs_tensor = torch.FloatTensor([obs_array])
    with torch.no_grad():
        mu, _, _ = brain(obs_tensor)
    return mu[0].numpy()


def test_herbivore_flight(herb_brain, results_full_folder,
                          to_show: bool = True, to_save: bool = True):
    print("Running Herbivore Flight Test...")
    distances = np.linspace(1.0, -1.0, 50)  # Sweep from Far (1.0) to Touching (-1.0)

    # Store actions for different predator masses
    accel_huge, turn_huge = [], []
    accel_tiny, turn_tiny = [], []

    for d in distances:
        # Base Herbivore State: Healthy, Medium Mass, Stopped, No Food Seen
        base_state = [
            0.5, 1.0, 0.0, 0.0, -1.0,  # Self: Energy, Health, Mass, Strength, Vel
            0.0, 1.0, 0.0,  # Food: None seen
            1.0, d, 0.0, 1.0, 0.0  # Agent: Seen, Dist=d, Angle=0 (Front), Type=Carnivore, Mass=Placeholder
        ]

        # Test 1: Massive Predator (Mass = 1.0)
        state_huge = base_state.copy()
        state_huge[-1] = 1.0
        act_h = get_action(herb_brain, state_huge)
        accel_huge.append(act_h[0])
        turn_huge.append(act_h[1])

        # Test 2: Tiny Predator (Mass = -1.0)
        state_tiny = base_state.copy()
        state_tiny[-1] = -1.0
        act_t = get_action(herb_brain, state_tiny)
        accel_tiny.append(act_t[0])
        turn_tiny.append(act_t[1])

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Herbivore Decision Making: Approaching Predator (Angle = 0)", fontsize=14)

    ax1.plot(distances, accel_huge, 'r-', label='Huge Predator', linewidth=2)
    ax1.plot(distances, accel_tiny, 'r--', label='Tiny Predator', linewidth=2)
    ax1.set_title("Acceleration Response")
    ax1.set_xlabel("Predator Distance (1 = Far, -1 = Close)")
    ax1.set_ylabel("Motor Output: Acceleration")
    ax1.invert_xaxis()  # Read left to right as predator gets closer
    ax1.legend()
    ax1.grid(True, alpha=0.5)

    ax2.plot(distances, turn_huge, 'b-', label='Huge Predator', linewidth=2)
    ax2.plot(distances, turn_tiny, 'b--', label='Tiny Predator', linewidth=2)
    ax2.set_title("Turning (Evasion) Response")
    ax2.set_xlabel("Predator Distance (1 = Far, -1 = Close)")
    ax2.set_ylabel("Motor Output: Turn Velocity")
    ax2.invert_xaxis()
    ax2.legend()
    ax2.grid(True, alpha=0.5)

    plt.tight_layout()
    if to_save:
        plt.savefig(f"{results_full_folder}\probe_herbivore.png", dpi=150)
    if to_show:
        plt.show()


def test_carnivore_hunting(carn_brain, results_full_folder,
                           to_show: bool = True, to_save: bool = True):
    print("Running Carnivore Hunting Test...")
    distances = np.linspace(-1.0, 1.0, 50)  # Close to Far
    masses = np.linspace(-1.0, 1.0, 50)  # Tiny to Huge Prey

    # We will create a 2D map of the Attack Signal
    attack_map = np.zeros((len(masses), len(distances)))

    for i, m in enumerate(masses):
        for j, d in enumerate(distances):
            # Base Carnivore State: Healthy, High Mass (0.5), Stopped, No Food
            state = [
                0.5, 1.0, 0.5, 0.5, -1.0,  # Self
                0.0, 1.0, 0.0,  # Food: None
                1.0, d, 0.0, -1.0, m  # Agent: Seen, Dist=d, Angle=0, Type=Herbivore, Mass=m
            ]

            act = get_action(carn_brain, state)
            attack_signal = act[2]  # 3rd output is attack
            attack_map[i, j] = attack_signal

    # Plotting
    plt.figure(figsize=(8, 6))

    # Use imshow/contour to map the decision boundary
    X, Y = np.meshgrid(distances, masses)
    contour = plt.contourf(X, Y, attack_map, levels=20, cmap='RdYlGn')
    plt.colorbar(contour, label='Attack Signal (>0 triggers bite)')

    # Draw a bold line at 0.0 to show the exact threshold of attack
    plt.contour(X, Y, attack_map, levels=[0.0], colors='black', linewidths=2, linestyles='dashed')

    plt.title("Carnivore Decision Making: Attack Threshold")
    plt.xlabel("Prey Distance (-1 = Close, 1 = Far)")
    plt.ylabel("Prey Mass (-1 = Tiny, 1 = Huge)")

    plt.tight_layout()
    if to_save:
        plt.savefig(f"{results_full_folder}\probe_carnivore.png", dpi=150)
    if to_show:
        plt.show()


def run_probes(results_folder: str, update_milestone: int,
               to_show: bool = True, to_save: bool = True):
    herb_brain = PPOBrain(13, 3)
    carn_brain = PPOBrain(13, 3)

    results_full_folder = fr"{RESULTS_PATH}\{results_folder}"
    herb_checkpoint = fr"{results_full_folder}\marl_checkpoints\herb_brain_{update_milestone:03d}.pth"
    carn_checkpoint = fr"{results_full_folder}\marl_checkpoints\carn_brain_{update_milestone:03d}.pth"
    if os.path.exists(herb_checkpoint) and os.path.exists(carn_checkpoint):
        herb_brain.load_state_dict(torch.load(herb_checkpoint))
        carn_brain.load_state_dict(torch.load(carn_checkpoint))
        herb_brain.eval()
        carn_brain.eval()

        test_herbivore_flight(herb_brain, results_full_folder,
                              to_show=to_show, to_save=to_save)
        test_carnivore_hunting(carn_brain, results_full_folder,
                               to_show=to_show, to_save=to_save)
    else:
        print("Checkpoints not found. Train the model first!")


if __name__ == "__main__":
    results_folder = "marl_results_500_ent005_exeatdist_reward"
    update_milestone = 500
    run_probes(results_folder=results_folder, update_milestone=update_milestone,
               to_show=True, to_save=False)
