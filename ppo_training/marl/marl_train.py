import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os

from ppo_training.marl.marl_env import Ecosystem, Agent
from ppo_training.ppo_brain import PPOBrain

# Hyperparameters
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.05
UPDATE_EPOCHS = 4
BATCH_SIZE = 500  # Steps per update
TOTAL_UPDATES = 500

RESULTS_PATH = rf'C:\Users\saar.nehemia\PycharmProjects\Bereshit\ppo_training\marl\results'

def train_marl(results_folder: str):
    results_full_folder = f'{RESULTS_PATH}/{results_folder}'
    os.makedirs(f"{results_full_folder}/marl_checkpoints", exist_ok=True)

    # Instantiate the two Master Species Brains (13 inputs, 3 outputs)
    herb_brain = PPOBrain(13, 3)
    carn_brain = PPOBrain(13, 3)

    herb_optim = optim.Adam(herb_brain.parameters(), lr=LR, eps=1e-5)
    carn_optim = optim.Adam(carn_brain.parameters(), lr=LR, eps=1e-5)

    env = Ecosystem(num_herbivores=10, num_carnivores=5, num_food=20)

    print("Starting Multi-Agent Co-Evolution Training...")

    for update in range(1, TOTAL_UPDATES + 1):
        # Data buffers categorized by species
        buffers = {
            'herb': {'obs': [], 'actions': [], 'logprobs': [], 'rewards': [], 'values': [], 'dones': []},
            'carn': {'obs': [], 'actions': [], 'logprobs': [], 'rewards': [], 'values': [], 'dones': []}
        }

        # --- 1. ROLLOUT PHASE ---
        for step in range(BATCH_SIZE):
            actions_dict = {}
            step_data = {'herb': {'ids': [], 'obs': []}, 'carn': {'ids': [], 'obs': []}}

            # Gather observations for all alive agents
            for agent in env.agents:
                if not agent.alive: continue
                agent.energy_prev = agent.energy
                agent.health_prev = agent.health
                obs = env._get_obs(agent)
                species = 'carn' if agent.is_carnivore else 'herb'

                step_data[species]['ids'].append(agent.id)
                step_data[species]['obs'].append(obs)

            # Batch Forward Pass (Massive speedup over looping)
            for species, brain in [('herb', herb_brain), ('carn', carn_brain)]:
                if len(step_data[species]['obs']) > 0:
                    obs_tensor = torch.FloatTensor(np.array(step_data[species]['obs']))
                    with torch.no_grad():
                        action, logprob, _, value = brain.get_action_and_value(obs_tensor)

                    # Store data and map actions to agent IDs
                    for i, a_id in enumerate(step_data[species]['ids']):
                        actions_dict[a_id] = action[i].numpy()

                        buffers[species]['obs'].append(step_data[species]['obs'][i])
                        buffers[species]['actions'].append(action[i].numpy())
                        buffers[species]['logprobs'].append(logprob[i].item())
                        buffers[species]['values'].append(value[i].item())

            # Step the environment
            env.step(actions_dict)

            # Calculate Emergent Rewards and Dones
            for species in ['herb', 'carn']:
                for a_id in step_data[species]['ids']:
                    # Find the agent object
                    agent = next(a for a in env.agents if a.id == a_id)
                    reward = 0

                    # Reward for existing: +0.01 for living, -5.0 for dying
                    reward += 0.01 if agent.alive else -5.0
                    done = not agent.alive

                    # Reward for eating (positive) or moving (negative)
                    delta_energy = agent.energy - agent.energy_prev
                    if delta_energy > 0:
                        # Eating is a massive relief (+ dopamine)
                        reward += delta_energy * 5.0
                    else:
                        # Moving is a slight cost, but we keep it small
                        # so the death penalty (-5.0) remains the primary fear.
                        reward += delta_energy * 0.1

                    # Reward for seeing prey and moving toward him
                    # if agent.is_carnivore and agent.alive:
                    #     obs = env._get_obs(agent)
                    #     agent_seen = obs[8]  # Agent Seen Flag
                    #     agent_dist = obs[9]  # Normalized Distance (-1 to 1)
                    #
                    #     if agent_seen > 0:
                    #         # Give a small "scenting" reward for having prey in sight
                    #         # and getting closer to it.
                    #         distance_reward = (1.0 - agent_dist) * 0.05
                    #         reward += distance_reward

                    if agent.health < agent.health_prev:
                        reward -= 1.0  # Pain/Recoil penalty

                    buffers[species]['rewards'].append(reward)
                    buffers[species]['dones'].append(float(done))

            # Ecosystem maintenance: Replace dead agents to keep simulation running
            dead_agents = [a for a in env.agents if not a.alive]
            for dead in dead_agents:
                # Respawn as a new agent of the same species
                env.agents.remove(dead)
                env.agents.append(Agent(dead.id, dead.is_carnivore))

        # --- 2. UPDATE PHASE (For Both Species) ---
        for species, brain, optimizer in [('herb', herb_brain, herb_optim), ('carn', carn_brain, carn_optim)]:
            if len(buffers[species]['rewards']) == 0: continue

            # Convert buffers to tensors
            b_obs = torch.FloatTensor(np.array(buffers[species]['obs']))
            b_actions = torch.FloatTensor(np.array(buffers[species]['actions']))
            b_logprobs = torch.FloatTensor(np.array(buffers[species]['logprobs']))
            b_rewards = torch.FloatTensor(np.array(buffers[species]['rewards']))
            b_values = torch.FloatTensor(np.array(buffers[species]['values']))
            b_dones = torch.FloatTensor(np.array(buffers[species]['dones']))

            # Generalized Advantage Estimation (GAE)
            with torch.no_grad():
                advantages = torch.zeros_like(b_rewards)
                lastgaelam = 0
                for t in reversed(range(len(b_rewards))):
                    if t == len(b_rewards) - 1:
                        nextnonterminal = 1.0 - b_dones[t]
                        nextvalues = 0  # Approximate terminal value
                    else:
                        nextnonterminal = 1.0 - b_dones[t + 1]
                        nextvalues = b_values[t + 1]
                    delta = b_rewards[t] + GAMMA * nextvalues * nextnonterminal - b_values[t]
                    advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                returns = advantages + b_values

            # PPO Epochs
            for epoch in range(UPDATE_EPOCHS):
                _, newlogprob, entropy, newvalue = brain.get_action_and_value(b_obs, b_actions)
                logratio = newlogprob - b_logprobs
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
                nn.utils.clip_grad_norm_(brain.parameters(), 0.5)
                optimizer.step()

        print(f"Update: {update:03d}/{TOTAL_UPDATES} | Herbivory & Carnivory models updated.")

        # Save Checkpoints
        if update % 10 == 0:
            torch.save(herb_brain.state_dict(), f"{results_full_folder}/marl_checkpoints/herb_brain_{update:03d}.pth")
            torch.save(carn_brain.state_dict(), f"{results_full_folder}/marl_checkpoints/carn_brain_{update:03d}.pth")

    print("Co-Evolution Training Complete!")


if __name__ == "__main__":
    results_folder = "updates500_ent005_ex_energy_hp_reward_2"
    train_marl(results_folder=results_folder)
