"""
Temporal ecology environment: drought, seasons, two grass types active at different times.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

from b_basic.sim_config import sim_config


class Environment:
    """
    Temporal ecology environment with drought periods and seasonal grass growth.
    Two grass types (A and B) alternate activity; drought reduces growth chance.
    """

    def __init__(self, map_filename, grass_generation_rate, leaves_generation_rate):
        # Load the PNG image. Values are assumed to be in [0, 1].
        self.map_data = plt.imread(map_filename)
        self.size = self.map_data.shape[0:2]
        # Create an obstacle mask: black areas where all channels are very low.
        self.obstacle_mask = np.all(self.map_data[:, :, :3] < 0.1, axis=2)
        # Grass regions: yellow (red and green high, blue low).
        self.grass_mask = (self.map_data[:, :, 0] > 0.7) & (self.map_data[:, :, 1] > 0.7) & (
                self.map_data[:, :, 2] < 0.3)
        # Tree regions: green (green high, red and blue low).
        self.tree_mask = (self.map_data[:, :, 0] < 0.2) & (self.map_data[:, :, 1] > 0.6) & (
                self.map_data[:, :, 2] < 0.5)

        self.height, self.width, _ = self.map_data.shape
        self.water_source = (self.width // 2, self.height // 2, 50)

        self.grass_generation_rate = grass_generation_rate
        self.leaves_generation_rate = leaves_generation_rate

        self.grass_indices = np.argwhere(self.grass_mask)
        self.leaf_indices = np.argwhere(self.tree_mask)

        # Grass: single list (all types merged for eating); seasonal modulation via growth
        self.grass_points = []
        self.grass_indices_to_remove = []

        self.leaf_points = []
        self.leaf_indices_to_remove = []

        self.grass_kd_tree = self.build_grass_kd_tree()
        self.leaves_kd_tree = self.build_leaves_kd_tree()

        # Temporal ecology: drought and season config (with defaults if not in config)
        self.season_length = getattr(sim_config.config, 'SEASON_LENGTH', 5000)
        self.drought_probability = getattr(sim_config.config, 'DROUGHT_PROBABILITY', 0.1)
        self.drought_growth_factor = getattr(sim_config.config, 'DROUGHT_GROWTH_FACTOR', 0.2)
        self.drought_length = getattr(sim_config.config, 'DROUGHT_LENGTH', 500)
        self.grass_type_a_phase = getattr(sim_config.config, 'GRASS_TYPE_A_ACTIVE_PHASE', [0.0, 0.5])
        self.grass_type_b_phase = getattr(sim_config.config, 'GRASS_TYPE_B_ACTIVE_PHASE', [0.5, 1.0])

    def _is_drought(self, step_counter: int) -> bool:
        """Periodic drought: every DROUGHT_LENGTH steps, DROUGHT_PROBABILITY chance to be in drought."""
        if self.drought_length <= 0:
            return False
        cycle_pos = step_counter % (2 * self.drought_length)  # drought + normal
        if cycle_pos < self.drought_length:
            return np.random.rand() < self.drought_probability
        return False

    def _seasonal_growth_factor(self, step_counter: int) -> float:
        """
        Returns growth factor (0 or 1) based on season.
        Type A grows in first half of season, type B in second half.
        """
        if self.season_length <= 0:
            return 1.0
        phase = (step_counter % self.season_length) / self.season_length
        a_low, a_high = self.grass_type_a_phase
        b_low, b_high = self.grass_type_b_phase
        if a_low <= phase < a_high:
            return 1.0  # Type A active
        if b_low <= phase < b_high:
            return 1.0  # Type B active
        return 0.0  # Between seasons (no growth)

    def _generate_new_food_points(self, food_indices, food_points,
                                  max_food_num_points, food_generation_rate,
                                  food_growth_chance, step_counter: int):
        """Generate new food with drought and seasonal modulation."""
        num_food_indices = len(food_indices)
        new_food_points = []
        if num_food_indices > 0:
            drought_factor = self.drought_growth_factor if self._is_drought(step_counter) else 1.0
            seasonal_factor = self._seasonal_growth_factor(step_counter)
            effective_chance = food_growth_chance * drought_factor * seasonal_factor

            num_food_points_to_add = min([int(food_generation_rate),
                                          max_food_num_points - len(food_points)])

            if num_food_points_to_add > 0 and effective_chance > 0:
                if np.random.rand() <= effective_chance:
                    choices = food_indices[np.random.choice(
                        num_food_indices, num_food_points_to_add, replace=True)]
                    for pt in choices:
                        new_food_points.append([pt[1], pt[0]])
        return new_food_points

    def remove_grass_points(self):
        for grass_idx in sorted(self.grass_indices_to_remove, reverse=True):
            try:
                self.grass_points.pop(grass_idx)
            except IndexError:
                pass
        self.grass_indices_to_remove = []

    def get_extent(self):
        return [0, self.width, 0, self.height]

    def build_grass_kd_tree(self) -> KDTree:
        positions = [p for p in self.grass_points if isinstance(p, (list, np.ndarray)) and len(p) >= 2]
        if positions:
            return KDTree(np.array(positions)[:, :2])
        return KDTree([[0, 0]])

    def build_leaves_kd_tree(self) -> KDTree:
        if not self.leaf_points:
            return KDTree([[0, 0]])
        coords = []
        for p in self.leaf_points:
            if hasattr(p, '__getitem__'):
                coords.append([p[0], p[1]] if len(p) >= 2 else [0, 0])
            else:
                coords.append([0, 0])
        return KDTree(np.array(coords))

    def update_grass_kd_tree(self, step_counter: int):
        """Update grass with temporal ecology (drought, seasons)."""
        self.remove_grass_points()

        new_food_points = self._generate_new_food_points(
            food_indices=self.grass_indices,
            food_points=self.grass_points,
            max_food_num_points=sim_config.config.GRASS_MAX_NUM,
            food_generation_rate=self.grass_generation_rate,
            food_growth_chance=sim_config.config.GRASS_GROWTH_CHANCE,
            step_counter=step_counter)
        self.grass_points.extend(new_food_points)

        self.grass_kd_tree = self.build_grass_kd_tree()
