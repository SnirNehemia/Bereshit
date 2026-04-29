import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import os

from ppo_training.ppo_brain import PPOBrain

# --- CONFIGURATION ---
INPUT_DIM = 13
ACTION_DIM = 3
GRID_RES = 40  # Resolution of the heatmap


# Make sure FFMPEG is in your system path, or set it explicitly here:
plt.rcParams[
    'animation.ffmpeg_path'] = r'C:\Users\saar.nehemia\PycharmProjects\Bereshit\a_utils\ffmpeg-2025-02-20-git-bc1a3bfd2c-essentials_build\bin\ffmpeg.exe'

RESULTS_PATH = rf'C:\Users\saar.nehemia\PycharmProjects\Bereshit\ppo_training\marl\results'

def generate_value_grid(model, value_type: str, a_type: float = 1.0, a_mass: float = 0.0):
    """Computes a grid of Critic values for the specified species."""
    dist_sweep = np.linspace(1.0, -1.0, GRID_RES)  # Far (1) to Close (-1)
    angle_sweep = np.linspace(-1.0, 1.0, GRID_RES)  # Left (-1) to Right (1)
    X, Y = np.meshgrid(angle_sweep, dist_sweep)
    Z = np.zeros_like(X)

    # Base "Neutral" State
    # [Energy, Health, Mass, Strength, Vel, F_Seen, F_D, F_A, A_Seen, A_D, A_A, A_Type, A_Mass]
    base_obs = [0.0, 1.0, 0.0, 0.0, -1.0]  # Self

    for i in range(GRID_RES):
        for j in range(GRID_RES):
            if value_type == 'Food':
                obs = base_obs + [1.0, Y[i, j], X[i, j]] + [0.0, 1.0, 0.0, 0.0, 0.0]
            else:
                obs = base_obs + [0.0, 1.0, 0.0] + [1.0, Y[i, j], X[i, j], a_type, a_mass]

            obs_t = torch.FloatTensor([obs])
            with torch.no_grad():
                value = model.critic(obs_t).item()
            Z[i, j] = value
    return X, Y, Z


def create_evolution_movie(results_folder: str, value_type: str, a_type: float = 1.0, a_mass: float = 0.0):
    # 1. Identify available updates
    results_full_folder = fr"{RESULTS_PATH}\{results_folder}"
    checkpoints = sorted([int(f.split('_')[-1].split('.')[0])
                          for f in os.listdir(f'{results_full_folder}\marl_checkpoints') if 'herb' in f])

    if not checkpoints:
        print("No checkpoints found in 'marl_checkpoints/'. Start training first!")
        return

    # Setup Figure
    fig, (ax_herb, ax_carn) = plt.subplots(1, 2, figsize=(14, 6))
    writer = FFMpegWriter(fps=4)  # Slow FPS so you can study the transitions

    herb_brain = PPOBrain(INPUT_DIM, ACTION_DIM)
    carn_brain = PPOBrain(INPUT_DIM, ACTION_DIM)

    if value_type == 'Food':
        video_type = 'Food'
    else:
        a_size = 'Tiny' if a_mass == -1.0 else 'Huge'
        a_type_str = 'Herb' if a_type == -1.0 else 'Carn'
        video_type = f'{a_size} {a_type_str}'
    video_name = fr"{results_full_folder}\{video_type}_value_map_video.mp4"
    print(f"Generating movie: {video_name}")

    with writer.saving(fig, video_name, dpi=100):
        for update in checkpoints:
            print(f"Processing Update {update}...")

            # Load Weights
            herb_brain.load_state_dict(torch.load(fr"{results_full_folder}/marl_checkpoints/herb_brain_{update:03d}.pth"))
            carn_brain.load_state_dict(torch.load(fr"{results_full_folder}/marl_checkpoints/carn_brain_{update:03d}.pth"))

            # --- Herbivore Plot ---
            ax_herb.clear()
            X, Y, Z_herb = generate_value_grid(model=herb_brain, value_type=value_type, a_type=a_type, a_mass=a_mass)
            im_h = ax_herb.contourf(X, Y, Z_herb, levels=20, cmap='viridis')
            ax_herb.set_title(f"Herbivore: {video_type} Value (Update {update})")
            ax_herb.set_xlabel("Angle")
            ax_herb.set_ylabel("Distance")

            # --- Carnivore Plot ---
            ax_carn.clear()
            X, Y, Z_carn = generate_value_grid(model=carn_brain, value_type=value_type, a_type=a_type, a_mass=a_mass)
            im_c = ax_carn.contourf(X, Y, Z_carn, levels=20, cmap='magma')
            ax_carn.set_title(f"Carnivore: {video_type} Value (Update {update})")
            ax_carn.set_xlabel("Angle")
            ax_herb.set_ylabel("Distance")

            # Add dynamic colorbars only on the first frame
            if update == checkpoints[0]:
                fig.colorbar(im_h, ax=ax_herb)
                fig.colorbar(im_c, ax=ax_carn)

            writer.grab_frame()

    plt.close()
    print("Done!")


if __name__ == "__main__":
    results_folder = '03_marl_results_500_ent005_ex_energy_dist_reward'
    value_type = 'Creature'
    a_type = 1.0  # -1.0 (herbivore), 1.0 (carnivore)
    a_mass = 1.0  # -1.0 (tiny), 1.0 (huge)
    create_evolution_movie(results_folder=results_folder,
                           value_type=value_type, a_type=a_type, a_mass=a_mass)
