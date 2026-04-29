import numpy as np


class ForagingSandbox:
    def __init__(self, num_food=10):
        self.num_food = num_food
        self.vision_radius = 1.0
        self.fov = np.pi

        # We keep the same state dim to maintain compatibility
        self.state_dim = 5
        self.action_dim = 2
        self.reset()

    def reset(self):
        self.pos = np.random.uniform(-1, 1, size=2)
        self.food_positions = np.random.uniform(-1.5, 1.5, size=(self.num_food, 2))
        self.angle = np.random.uniform(0, 2 * np.pi)
        self.energy = 1.0  # Setpoint is 1.0
        self.vel = 0.0
        return self._get_obs()

    def _get_obs(self):
        # [Same observation logic as before: seen, dist, angle, vel, energy]
        rel_positions = self.food_positions - self.pos
        distances = np.linalg.norm(rel_positions, axis=1)
        global_angles = np.arctan2(rel_positions[:, 1], rel_positions[:, 0])
        rel_angles = (global_angles - self.angle + np.pi) % (2 * np.pi) - np.pi

        visible_mask = (distances < self.vision_radius) & (np.abs(rel_angles) <= (self.fov / 2))

        food_seen, norm_dist, norm_angle = 0.0, 1.0, 0.0
        if np.any(visible_mask):
            idx = np.argmin(distances[visible_mask])
            food_seen = 1.0
            norm_dist = (distances[visible_mask][idx] / self.vision_radius) * 2 - 1
            norm_angle = rel_angles[visible_mask][idx] / (self.fov / 2)

        return np.array([food_seen, norm_dist, norm_angle, self.vel / 0.5, self.energy], dtype=np.float32)

    def step(self, action):
        accel, turn_vel = action

        # Store PREVIOUS energy state for homeostatic calculation
        prev_energy_error = abs(1.0 - self.energy)

        # Physics updates
        self.angle = (self.angle + turn_vel * 0.2) % (2 * np.pi)
        self.vel = np.clip(self.vel + accel * 0.05 - (0.02 * self.vel), 0, 0.5)
        self.pos += [self.vel * np.cos(self.angle), self.vel * np.sin(self.angle)]
        self.pos = np.clip(self.pos, -1.5, 1.5)

        # Metabolism: Base cost + movement cost
        cost = 0.002 + (0.003 * abs(accel)) + (0.001 * abs(turn_vel))
        self.energy -= cost

        # Eating
        dist_to_food = np.linalg.norm(self.food_positions - self.pos, axis=1)
        eaten = np.where(dist_to_food < 0.15)[0]
        if len(eaten) > 0:
            self.energy = min(1.0, self.energy + (0.4 * len(eaten)))
            self.food_positions[eaten] = np.random.uniform(-1.5, 1.5, size=(len(eaten), 2))

        # --- HOMEOSTATIC REWARD CALCULATION ---
        current_energy_error = abs(1.0 - self.energy)

        # Reward is the REDUCTION in error (positive if error gets smaller)
        # We multiply by a factor to make the signal strong enough for PPO
        reward = (prev_energy_error - current_energy_error) * 10.0

        done = self.energy <= 0
        if done: reward -= 2.0  # Extra penalty for total system failure (death)

        return self._get_obs(), float(reward), done