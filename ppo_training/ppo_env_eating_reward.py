import numpy as np


class ForagingSandbox:
    def __init__(self, num_food=10):
        self.num_food = num_food
        self.vision_radius = 1.0  # How far the agent can see
        self.fov = np.pi  # 180-degree Field of View

        # State: [food_seen_flag, target_dist, target_angle, velocity, energy]
        self.state_dim = 5
        self.action_dim = 2

        self.reset()

    def reset(self):
        self.pos = np.random.uniform(-1, 1, size=2)
        self.food_positions = np.random.uniform(-1.5, 1.5, size=(self.num_food, 2))
        self.angle = np.random.uniform(0, 2 * np.pi)
        self.energy = 1.0
        self.vel = 0.0
        return self._get_obs()

    def _get_obs(self):
        # 1. Calculate relative positions to ALL food points
        rel_positions = self.food_positions - self.pos
        distances = np.linalg.norm(rel_positions, axis=1)

        # Calculate angle to all food relative to the agent's current heading
        global_angles = np.arctan2(rel_positions[:, 1], rel_positions[:, 0])
        rel_angles = global_angles - self.angle
        rel_angles = (rel_angles + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]

        # 2. Filter by Physical Limits (Must be close enough AND inside FOV)
        in_range = distances < self.vision_radius
        in_fov = np.abs(rel_angles) <= (self.fov / 2)
        visible_mask = in_range & in_fov

        # 3. Default "Blind" State
        food_seen = 0.0
        norm_dist = 1.0  # If no food, tell the network distance is at 'maximum'
        norm_angle = 0.0  # If no food, angle is 0

        # 4. If ANY food is visible, acquire the 'Target Lock'
        if np.any(visible_mask):
            visible_distances = distances[visible_mask]
            visible_angles = rel_angles[visible_mask]

            # Find the closest food among the visible ones
            closest_idx = np.argmin(visible_distances)
            closest_dist = visible_distances[closest_idx]
            closest_angle = visible_angles[closest_idx]

            # Populate inputs with normalized values
            food_seen = 1.0
            norm_dist = (closest_dist / self.vision_radius) * 2 - 1  # Scales to [-1, 1]
            norm_angle = closest_angle / (self.fov / 2)  # Scales to [-1, 1]

        # Normalize internal states
        norm_vel = (self.vel / 0.5) * 2 - 1
        norm_energy = self.energy * 2 - 1

        return np.array([food_seen, norm_dist, norm_angle, norm_vel, norm_energy], dtype=np.float32)

    def step(self, action):
        accel, turn_vel = action

        # Physics
        self.angle = (self.angle + turn_vel * 0.2) % (2 * np.pi)
        self.vel = np.clip(self.vel + accel * 0.05 - (0.02 * self.vel), 0, 0.5)

        self.pos[0] = np.clip(self.pos[0] + self.vel * np.cos(self.angle), -1.5, 1.5)
        self.pos[1] = np.clip(self.pos[1] + self.vel * np.sin(self.angle), -1.5, 1.5)

        # Metabolism
        cost = 0.002 + (0.003 * abs(accel)) + (0.001 * abs(turn_vel))
        self.energy -= cost
        reward = -cost
        done = False

        # Eating logic
        distances = np.linalg.norm(self.food_positions - self.pos, axis=1)
        eaten_indices = np.where(distances < 0.15)[0]

        if len(eaten_indices) > 0:
            reward += 1.0 * len(eaten_indices)
            self.energy = min(1.0, self.energy + (0.4 * len(eaten_indices)))

            # Respawn
            new_spawns = np.random.uniform(-1.5, 1.5, size=(len(eaten_indices), 2))
            self.food_positions[eaten_indices] = new_spawns

        if self.energy <= 0:
            reward -= 1.0
            done = True

        return self._get_obs(), reward, done