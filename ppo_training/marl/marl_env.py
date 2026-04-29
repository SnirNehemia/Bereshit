import numpy as np

class Agent:
    def __init__(self, id, is_carnivore):
        self.id = id
        self.is_carnivore = is_carnivore
        self.color = 'red' if is_carnivore else 'blue'

        # Genetics / Stats
        self.mass = np.random.uniform(0.8, 1.5) if not is_carnivore else np.random.uniform(1.2, 2.5)
        self.strength = self.mass * np.random.uniform(0.8, 1.2)

        # Dynamics
        self.pos = np.random.uniform(-3, 3, size=2)
        self.angle = np.random.uniform(0, 2 * np.pi)
        self.vel = 0.0

        # Vitals
        self.max_health = 1.0 * self.mass
        self.health = self.max_health
        self.health_prev = self.health
        self.energy = 1.0
        self.energy_prev = self.energy
        self.alive = True

class Ecosystem:
    def __init__(self, num_herbivores=10, num_carnivores=5, num_food=20):
        self.vision_radius = 1.5
        self.fov = np.pi  # 180 degrees
        self.max_attack_dist = 0.2

        # Initialize Entities
        self.agents = []
        for i in range(num_herbivores):
            self.agents.append(Agent(i, is_carnivore=False))
        for i in range(num_herbivores, num_herbivores + num_carnivores):
            self.agents.append(Agent(i, is_carnivore=True))

        self.food_positions = np.random.uniform(-4, 4, size=(num_food, 2))

    def _get_obs(self, agent):
        """Generates the 13-dimensional sensory input for a specific agent."""
        # 1. SELF STATS (Normalized)
        self_stats = [
            (agent.energy * 2) - 1,
            (agent.health / agent.max_health) * 2 - 1,
            (agent.mass / 2.5) * 2 - 1,
            (agent.strength / 3.0) * 2 - 1,
            (agent.vel / 0.5) * 2 - 1
        ]

        # 2. FOOD VISION
        food_seen, f_dist, f_angle = 0.0, 1.0, 0.0
        if len(self.food_positions) > 0:
            f_rel_pos = self.food_positions - agent.pos
            f_distances = np.linalg.norm(f_rel_pos, axis=1)
            f_angles = (np.arctan2(f_rel_pos[:, 1], f_rel_pos[:, 0]) - agent.angle + np.pi) % (2 * np.pi) - np.pi

            f_visible = (f_distances < self.vision_radius) & (np.abs(f_angles) <= self.fov / 2)
            if np.any(f_visible):
                idx = np.argmin(f_distances[f_visible])
                food_seen = 1.0
                f_dist = (f_distances[f_visible][idx] / self.vision_radius) * 2 - 1
                f_angle = f_angles[f_visible][idx] / (self.fov / 2)

        # 3. AGENT VISION
        agent_seen, a_dist, a_angle, a_type, a_mass = 0.0, 1.0, 0.0, 0.0, 0.0
        other_agents = [a for a in self.agents if a.id != agent.id and a.alive]

        if len(other_agents) > 0:
            a_positions = np.array([a.pos for a in other_agents])
            a_rel_pos = a_positions - agent.pos
            a_distances = np.linalg.norm(a_rel_pos, axis=1)
            a_angles = (np.arctan2(a_rel_pos[:, 1], a_rel_pos[:, 0]) - agent.angle + np.pi) % (2 * np.pi) - np.pi

            a_visible = (a_distances < self.vision_radius) & (np.abs(a_angles) <= self.fov / 2)
            if np.any(a_visible):
                idx = np.argmin(a_distances[a_visible])
                closest_agent = other_agents[np.where(a_visible)[0][idx]]

                agent_seen = 1.0
                a_dist = (a_distances[a_visible][idx] / self.vision_radius) * 2 - 1
                a_angle = a_angles[a_visible][idx] / (self.fov / 2)
                a_type = 1.0 if closest_agent.is_carnivore else -1.0
                a_mass = (closest_agent.mass / 2.5) * 2 - 1

        obs = self_stats + [food_seen, f_dist, f_angle] + [agent_seen, a_dist, a_angle, a_type, a_mass]
        return np.array(obs, dtype=np.float32)

    def step(self, actions_dict):
        """Executes one simulation tick using externally provided actions."""
        # APPLY PHYSICS & METABOLISM
        for agent in self.agents:
            if not agent.alive or agent.id not in actions_dict: continue

            accel, turn, attack_signal = actions_dict[agent.id]

            agent.angle = (agent.angle + turn * 0.2) % (2 * np.pi)
            agent.vel = np.clip(agent.vel + accel * 0.05 - (0.02 * agent.vel), 0, 0.5)
            agent.pos += [agent.vel * np.cos(agent.angle), agent.vel * np.sin(agent.angle)]
            agent.pos = np.clip(agent.pos, -4, 4)

            agent.energy -= (0.002 + 0.003 * abs(accel))
            if agent.energy <= 0:
                agent.health -= 0.05
                if agent.health <= 0: agent.alive = False

        # 3. ENVIRONMENT INTERACTIONS (Eating Food)
        for agent in self.agents:
            if not agent.alive: continue
            if len(self.food_positions) > 0:
                dists = np.linalg.norm(self.food_positions - agent.pos, axis=1)
                eaten = np.where(dists < 0.15)[0]
                if len(eaten) > 0:
                    agent.energy = min(1.0, agent.energy + 0.3 * len(eaten))
                    # Respawn food
                    self.food_positions[eaten] = np.random.uniform(-4, 4, size=(len(eaten), 2))

        # 4. COMBAT LOGIC (Carnivore Attacks)
        for predator in self.agents:
            if not predator.alive or not predator.is_carnivore: continue

            _, _, attack_signal = actions_dict[predator.id]
            if attack_signal > 0:  # Agent chooses to attack
                predator.energy -= 0.05  # Attack costs energy

                # Find valid prey nearby
                other_agents = [a for a in self.agents if a.id != predator.id and a.alive]
                if not other_agents: continue

                dists = np.linalg.norm(np.array([a.pos for a in other_agents]) - predator.pos, axis=1)
                close_idx = np.where(dists < self.max_attack_dist)[0]

                for idx in close_idx:
                    prey = other_agents[idx]

                    # Probability Math: Based on Mass Ratio
                    mass_ratio = predator.mass / (predator.mass + prey.mass)
                    success_prob = mass_ratio * 0.9  # Max 90% success

                    roll = np.random.uniform(0, 1)
                    if roll < success_prob:
                        # SUCCESS: Prey loses health
                        damage = predator.strength * 0.4
                        prey.health -= damage
                        if prey.health <= 0:
                            prey.alive = False
                            predator.energy = min(1.0, predator.energy + prey.mass * 0.5)
                    else:
                        # FAILURE: Predator takes counter-attack damage based on how badly they failed
                        fail_margin = roll - success_prob
                        recoil_damage = prey.strength * fail_margin * 0.3
                        predator.health -= recoil_damage
                        if predator.health <= 0: predator.alive = False
                        break  # Attack sequence ends if predator gets hurt