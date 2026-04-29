import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from ppo_training.ppo_brain import PPOBrain
from ppo_training.ppo_env_eating_reward import ForagingSandbox

plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\saar.nehemia\PycharmProjects\Bereshit\a_utils\ffmpeg-2025-02-20-git-bc1a3bfd2c-essentials_build\bin\ffmpeg.exe'

# PPO Hyperparameters
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.01
UPDATE_EPOCHS = 4
BATCH_SIZE = 500
TOTAL_UPDATES = 300

OUTPUT_FOLDER_PATH = "eating_reward"


def evaluate_and_record(model, update_idx):
    """Runs a single deterministic episode and saves it as an MP4 with a live reward plot."""
    env = ForagingSandbox()
    state = env.reset()

    # Create a 1x2 grid: Left for simulation, Right for the plot
    fig, (ax_sim, ax_plot) = plt.subplots(1, 2, figsize=(10, 5))
    writer = FFMpegWriter(fps=20)
    filename = f"{OUTPUT_FOLDER_PATH}/progress_videos/update_{update_idx:03d}.mp4"

    print(f"--> Recording milestone: {filename}")

    cumulative_rewards = []
    current_cum_reward = 0.0

    with writer.saving(fig, filename, dpi=100):
        for t in range(300):
            # --- 1. Render the Simulation (Left Plot) ---
            ax_sim.clear()
            ax_sim.set_xlim(-1.5, 1.5)
            ax_sim.set_ylim(-1.5, 1.5)
            ax_sim.set_aspect('equal')
            ax_sim.set_title(f"Training Update {update_idx} | Energy: {env.energy:.2f}")

            # Draw food points and agent
            ax_sim.plot(env.food_positions[:, 0], env.food_positions[:, 1], 'go', markersize=10)
            ax_sim.plot(env.pos[0], env.pos[1], 'ro', markersize=6)
            dx = 0.1 * np.cos(env.angle)
            dy = 0.1 * np.sin(env.angle)
            ax_sim.arrow(env.pos[0], env.pos[1], dx, dy, head_width=0.05, color='red')

            # --- 2. Render the Live Reward Curve (Right Plot) ---
            ax_plot.clear()
            ax_plot.set_title("Cumulative Episode Reward")
            ax_plot.set_xlim(0, 300)

            # Dynamically adjust Y-axis to fit the curve beautifully
            min_y = min(cumulative_rewards) - 1 if cumulative_rewards else -2
            max_y = max(cumulative_rewards) + 1 if cumulative_rewards else 2
            ax_plot.set_ylim(min_y, max_y)

            ax_plot.plot(cumulative_rewards, color='blue', linewidth=2)
            ax_plot.set_xlabel("Step")
            ax_plot.set_ylabel("Total Reward")
            ax_plot.grid(True, linestyle='--', alpha=0.6)

            writer.grab_frame()

            # --- 3. Step the Environment ---
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                mu, std, value = model(state_t)

            state, reward, done = env.step(mu[0].numpy())

            # Record the reward for the next frame's plot
            current_cum_reward += reward
            cumulative_rewards.append(current_cum_reward)

            if done:
                # Grab one final frame to show the moment of death/completion
                ax_plot.clear()
                ax_plot.set_title("Cumulative Episode Reward")
                ax_plot.set_xlim(0, 300)
                ax_plot.set_ylim(min_y, max_y)
                ax_plot.plot(cumulative_rewards, color='blue', linewidth=2)
                ax_plot.grid(True, linestyle='--', alpha=0.6)
                writer.grab_frame()
                break

    plt.close(fig)


def train_ppo():
    os.makedirs(f"{OUTPUT_FOLDER_PATH}/progress_videos", exist_ok=True)
    os.makedirs(f"{OUTPUT_FOLDER_PATH}/checkpoints", exist_ok=True)

    env = ForagingSandbox()
    model = PPOBrain(env.state_dim, env.action_dim)
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    state = env.reset()
    all_episode_rewards = []
    episode_reward = 0
    episodes_completed = 0

    print("Starting Training with Periodic Video & Checkpoint Generation...")
    # Save the untrained "tabula rasa" brain
    evaluate_and_record(model, 0)
    torch.save(model.state_dict(), f"{OUTPUT_FOLDER_PATH}/checkpoints/brain_update_000.pth")

    for update in range(1, TOTAL_UPDATES + 1):
        obs = torch.zeros((BATCH_SIZE, env.state_dim))
        actions = torch.zeros((BATCH_SIZE, env.action_dim))
        logprobs = torch.zeros((BATCH_SIZE,))
        rewards = torch.zeros((BATCH_SIZE,))
        dones = torch.zeros((BATCH_SIZE,))
        values = torch.zeros((BATCH_SIZE,))

        # 1. Rollout Phase
        for step in range(BATCH_SIZE):
            obs[step] = torch.FloatTensor(state)
            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(obs[step].unsqueeze(0))

            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob.flatten()

            next_state, reward, done = env.step(action[0].numpy())
            rewards[step] = float(reward)  # Explicitly cast to Python float
            dones[step] = float(done)  # Explicitly cast boolean to float
            episode_reward += reward

            if done:
                state = env.reset()
                episodes_completed += 1
                all_episode_rewards.append(episode_reward)
                episode_reward = 0
            else:
                state = next_state

        # 2. GAE Phase
        with torch.no_grad():
            next_value = model.get_action_and_value(torch.FloatTensor(state).unsqueeze(0))[-1]
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(BATCH_SIZE)):
                if t == BATCH_SIZE - 1:
                    nextnonterminal = 1.0 - done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + values

        # 3. PPO Update Phase
        for epoch in range(UPDATE_EPOCHS):
            _, newlogprob, entropy, newvalue = model.get_action_and_value(obs, actions)
            logratio = newlogprob - logprobs
            ratio = logratio.exp()

            mb_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            v_loss = 0.5 * ((newvalue - returns) ** 2).mean()
            entropy_loss = entropy.mean()
            loss = pg_loss - ENT_COEF * entropy_loss + v_loss * 0.5

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        # Terminal Output
        avg_rew = np.mean(all_episode_rewards[-20:]) if len(all_episode_rewards) > 0 else 0
        print(
            f"Update: {update:03d}/{TOTAL_UPDATES} | Episodes: {episodes_completed:03d} | Avg Reward (Last 20): {avg_rew:.2f}")

        # --- PROGRESS RECORDING ---
        if update % 10 == 0:
            evaluate_and_record(model, update)

            # Save the specific weights for this update
            checkpoint_path = f"{OUTPUT_FOLDER_PATH}/checkpoints/brain_update_{update:03d}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"--> Checkpoint saved: {checkpoint_path}")

    # Save final artifacts
    np.save(f"{OUTPUT_FOLDER_PATH}/progress_videos/training_rewards.npy", np.array(all_episode_rewards))
    print("Training complete. Models and metrics saved.")


if __name__ == "__main__":
    train_ppo()