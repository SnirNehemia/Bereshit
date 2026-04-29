import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from ppo_training.ppo_brain import PPOBrain

RESULTS_PATH = rf'C:\Users\saar.nehemia\PycharmProjects\Bereshit\ppo_training\marl\results'


def load_brain(results_folder, species, update_milestone):
    brain = PPOBrain(13, 3)

    results_full_folder = fr"{RESULTS_PATH}\{results_folder}"
    brain_checkpoint = fr"{results_full_folder}\marl_checkpoints\{species}_brain_{update_milestone:03d}.pth"
    if os.path.exists(brain_checkpoint):
        brain.load_state_dict(torch.load(brain_checkpoint))
        brain.eval()

    return brain


def probe_behavior_vectors(results_folder: str, species: str, update_milestone: int,
                           to_show: bool = True, to_save: bool = True):
    """
    Probes the directional intent of the brain using vector alignment.
    Alignment = Dot(Target_Direction_Vector, Agent_Movement_Vector)
    """

    # Load brain
    brain = load_brain(results_folder, species, update_milestone)

    res = 30
    angles = np.linspace(-1, 1, res)  # Normalized -1 to 1 (Left to Right)
    dists = np.linspace(1, -1, res)  # Normalized 1 to -1 (Far to Close)

    alignment_map = np.zeros((res, res))

    # Brain Input Dimensions: [Energy, Health, Mass, Strength, Vel, F_Seen, F_D, F_A, A_Seen, A_D, A_A, A_Type, A_Mass]
    # We assume a 'Predator' for Herbs and 'Prey' for Carnivores
    a_type = 1.0 if species == 'herb' else -1.0

    for i, d in enumerate(dists):
        for j, a in enumerate(angles):
            # Construct synthetic observation (Medium stats, stationary)
            obs = [0.0, 1.0, 1.0, 0.0, -1.0,    # Energy, health, mass, strength, velocity
                   0.0, 1.0, 0.0,               # food_seen, f_dist, f_angle
                   1.0, d, a, a_type, -1.0]      # agent_seen, a_dist, a_angle, a_type, a_mass

            obs_t = torch.FloatTensor([obs])
            with torch.no_grad():
                mu, _, _ = brain(obs_t)

            accel, turn, _ = mu[0].numpy()

            # --- Vector Math ---
            # 1. Target Vector (Where the stimulus is)
            # In our normalized space, 'a' is the angle.
            # We treat the agent's nose as 0 degrees (pointing 'up' on the y-axis)
            target_angle_rad = a * (np.pi / 2)  # Mapping -1..1 to -90..90 degrees
            v_target = np.array([np.sin(target_angle_rad), np.cos(target_angle_rad)])

            # 2. Response Vector (Where the agent is going)
            # We assume the agent turns first, then accelerates
            response_angle_rad = turn * 0.2  # Scaling based on our env.py turn logic
            v_response = np.array([
                accel * np.sin(response_angle_rad),
                accel * np.cos(response_angle_rad)
            ])

            # 3. The Alignment (Dot Product)
            # A negative value means the response vector is opposite to the target vector.
            alignment = np.dot(v_target, v_response) / accel
            alignment_map[i, j] = alignment

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.imshow(alignment_map, extent=[-1, 1, -1, 1], origin='lower', cmap='RdBu')
    plt.colorbar(label='Alignment (Positive = Toward, Negative = Away)')

    plt.title(f"Behavioral Alignment Map: {species.capitalize()} Brain")
    plt.xlabel("Stimulus Angle (Left to Right)")
    plt.ylabel("Stimulus Distance (Close to Far)")
    plt.grid(alpha=0.3)

    if to_save:
        results_full_folder = fr"{RESULTS_PATH}\{results_folder}"
        filename = f"{results_full_folder}\probe_vectors_{species}_update{update_milestone:03d}.png"
        plt.savefig(filename, dpi=150)
        print(f"Saved plot to '{filename}'")

    if to_show:
        plt.show()


if __name__ == '__main__':
    results_folder = 'marl_results_500_ent005_ex_eat_dist_reward'
    species = 'carn'
    update_milestone = 500
    probe_behavior_vectors(results_folder=results_folder, species=species, update_milestone=update_milestone,
                           to_show=True, to_save=False)
