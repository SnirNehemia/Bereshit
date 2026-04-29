"""
ECOSYSTEM DYNAMICS EVALUATOR
----------------------------
Renders the live simulation and tracks population counts over time.

What is GOOD to see:
- 'Oscillations': The red and blue lines should go up and down in a
  Lotka-Volterra cycle. Carnivores increase -> Herbivores decrease ->
  Carnivores starve/decrease -> Herbivores recover.
- 'Evasive Maneuvers': Blue dots actively turning 180 degrees
  away from red dots.

What is NOT GOOD to see:
- 'Immediate Extinction': If one species hits zero in the first 100 steps.
- 'Indifference': If a Carnivore walks right past a Herbivore without
  attempting an attack, or vice versa.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from ppo_training.marl.marl_env import Ecosystem
from ppo_training.ppo_brain import PPOBrain

# Make sure FFMPEG is in your system path, or set it explicitly here:
plt.rcParams[
    'animation.ffmpeg_path'] = r'C:\Users\saar.nehemia\PycharmProjects\Bereshit\a_utils\ffmpeg-2025-02-20-git-bc1a3bfd2c-essentials_build\bin\ffmpeg.exe'

RESULTS_PATH = rf'C:\Users\saar.nehemia\PycharmProjects\Bereshit\ppo_training\marl\results'


def render_ecosystem(results_folder: str, update_milestone: int):
    print(f"Loading Checkpoints from Update {update_milestone}...")

    # 1. Initialize Brains (13 inputs, 3 outputs)
    herb_brain = PPOBrain(13, 3)
    carn_brain = PPOBrain(13, 3)

    try:
        results_full_folder = rf"{RESULTS_PATH}\{results_folder}"
        herb_checkpoint = fr"{results_full_folder}\marl_checkpoints\herb_brain_{update_milestone:03d}.pth"
        carn_checkpoint = fr"{results_full_folder}\marl_checkpoints\carn_brain_{update_milestone:03d}.pth"
        herb_brain.load_state_dict(torch.load(herb_checkpoint))
        carn_brain.load_state_dict(torch.load(carn_checkpoint))
        herb_brain.eval()
        carn_brain.eval()
        print("Brains loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: Could not find checkpoint files. {e}")
        return

    # 2. Initialize Ecosystem
    # Feel free to change the starting populations here to test different scenarios
    env = Ecosystem(num_herbivores=15, num_carnivores=5, num_food=30)

    # Setup rendering
    fig, (ax_sim, ax_pop) = plt.subplots(1, 2, figsize=(12, 6))
    writer = FFMpegWriter(fps=20)
    filename = fr"{results_full_folder}\ecosystem_evaluation_{update_milestone:03d}.mp4"

    pop_history_herb = []
    pop_history_carn = []

    print(f"Recording simulation to {filename}...")

    with writer.saving(fig, filename, dpi=100):
        for t in range(600):  # Length of the video in frames
            ax_sim.clear()
            ax_pop.clear()

            # --- RENDER ARENA (LEFT) ---
            ax_sim.set_xlim(-4, 4)
            ax_sim.set_ylim(-4, 4)
            ax_sim.set_aspect('equal')
            ax_sim.set_title(f"Bereshit Ecosystem - Step {t}")

            # Draw Food
            if len(env.food_positions) > 0:
                ax_sim.plot(env.food_positions[:, 0], env.food_positions[:, 1], 'go', markersize=4, alpha=0.6)

            alive_herb = 0
            alive_carn = 0

            # Draw Agents
            for agent in env.agents:
                if not agent.alive: continue

                if agent.is_carnivore:
                    alive_carn += 1
                    color = 'red'
                else:
                    alive_herb += 1
                    color = 'blue'

                # Size relative to mass
                size = agent.mass * 8
                ax_sim.plot(agent.pos[0], agent.pos[1], marker='o', color=color, markersize=size)

                # Draw direction arrow
                dx = 0.2 * np.cos(agent.angle)
                dy = 0.2 * np.sin(agent.angle)
                ax_sim.arrow(agent.pos[0], agent.pos[1], dx, dy, head_width=0.08, color='black', alpha=0.5)

            # --- RENDER POPULATION TRACKER (RIGHT) ---
            pop_history_herb.append(alive_herb)
            pop_history_carn.append(alive_carn)

            ax_pop.set_title("Population Dynamics")
            ax_pop.set_xlim(0, 600)
            ax_pop.set_ylim(0, max(20, max(pop_history_herb + pop_history_carn) + 5))

            ax_pop.plot(pop_history_herb, color='blue', label='Herbivores', linewidth=2)
            ax_pop.plot(pop_history_carn, color='red', label='Carnivores', linewidth=2)
            if t == 0: ax_pop.legend()  # Only add legend once
            ax_pop.grid(True, linestyle='--', alpha=0.6)
            ax_pop.set_xlabel("Time Step")
            ax_pop.set_ylabel("Living Agents")

            writer.grab_frame()

            # --- GET ACTIONS & STEP ENV ---
            actions_dict = {}
            for agent in env.agents:
                if not agent.alive: continue

                obs = env._get_obs(agent)
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                brain = carn_brain if agent.is_carnivore else herb_brain

                with torch.no_grad():
                    # Deterministic action: Use mean (mu) for evaluation
                    mu, _, _ = brain(obs_t)

                actions_dict[agent.id] = mu[0].numpy()

            env.step(actions_dict)

            # If everything is dead, end the video early
            if alive_herb == 0 and alive_carn == 0:
                print("Total extinction reached. Ending recording early.")
                break

    print(f"Video saved successfully as '{filename}'.")


if __name__ == "__main__":
    results_folder = 'marl_results_500_ent001'
    update_milestone = 500
    render_ecosystem(results_folder=results_folder, update_milestone=update_milestone)
