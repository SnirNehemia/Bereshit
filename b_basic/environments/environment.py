import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

from b_basic.sim_config.codes import sim_config


class Environment:
    """
    Represents the simulation environment (map) including obstacles, vegetation, and a water source.
    The PNG map uses:
      - Black pixels to indicate obstacles (forbidden areas).
      - Yellow pixels for grass regions.
      - Green pixels for tree regions.
    """

    def __init__(self, map_filename, grass_generation_rate, leaves_generation_rate):
        # Load the PNG image. Values are assumed to be in [0, 1].
        self.map_data = plt.imread(map_filename)
        self.size = self.map_data.shape[0:2]
        # Create an obstacle mask: black areas where all channels are very low.
        self.obstacle_mask = np.all(self.map_data[:, :, :3] < 0.1, axis=2)
        # Grass regions: yellow (red and green high, blue low). Adjust thresholds as needed.
        self.grass_mask = (self.map_data[:, :, 0] > 0.7) & (self.map_data[:, :, 1] > 0.7) & (
                self.map_data[:, :, 2] < 0.3)
        # Tree regions: green (green high, red and blue low). Adjust thresholds as needed.
        self.tree_mask = (self.map_data[:, :, 0] < 0.2) & (self.map_data[:, :, 1] > 0.6) & (
                self.map_data[:, :, 2] < 0.5)

        # Define a water source as a blue circle in the center of the map.
        self.height, self.width, _ = self.map_data.shape
        self.water_source = (self.width // 2, self.height // 2, 50)

        self.grass_generation_rate = grass_generation_rate
        self.leaves_generation_rate = leaves_generation_rate

        # Pre-calculate indices for grass and tree regions for fast random sampling.
        self.grass_indices = np.argwhere(self.grass_mask)
        self.leaf_indices = np.argwhere(self.tree_mask)

        # Lists to hold dynamically generated vegetation points.
        self.grass_points = []
        self.grass_indices_to_remove = []

        self.leaf_points = []
        self.leaf_indices_to_remove = []

        self.grass_kd_tree = self.build_grass_kd_tree()

    @staticmethod
    def _generate_new_food_points(food_indices, food_points,
                                  max_food_num_points, food_generation_rate,
                                  food_growth_chance):
        num_food_indices = len(food_indices)
        new_food_points = []
        if num_food_indices > 0:
            # Calculate how many new food points to add
            num_food_points_to_add = min([int(food_generation_rate),
                                          max_food_num_points - len(food_points)])

            # Add food points
            if num_food_points_to_add > 0:
                if np.random.rand() < food_growth_chance:
                    choices = food_indices[np.random.choice(
                        num_food_indices, num_food_points_to_add, replace=True)]
                    for pt in choices:
                        new_food_points.append([pt[1], pt[0]])  # Convert image coordinates (row, col) to (x, y)
            return new_food_points

    def remove_grass_points(self):
        for grass_idx in self.grass_indices_to_remove:
            self.grass_points.pop(grass_idx)
        self.grass_indices_to_remove = []

    def get_extent(self):
        """
        Returns the extent [xmin, xmax, ymin, ymax] to be used with imshow.
        """
        # height, width, _ = self.map_data.shape
        return [0, self.width, 0, self.height]

    def build_grass_kd_tree(self) -> KDTree:
        """
        Builds a KDTree from the positions of all creatures.
        """
        positions = [grass_point for grass_point in self.grass_points]
        if positions:
            return KDTree(positions)
        else:
            return KDTree([[0, 0]])

    def update_grass_kd_tree(self):
        """
        :return: remove points, add points and update the KDTree
        """
        # Remove points
        self.remove_grass_points()

        # add points
        # Generate new grass points
        new_food_points = self._generate_new_food_points(
            food_indices=self.grass_indices,
            food_points=self.grass_points,
            max_food_num_points=sim_config.config.MAX_GRASS_NUM,
            food_generation_rate=self.grass_generation_rate,
            food_growth_chance=sim_config.config.GRASS_GROWTH_CHANCE)
        self.grass_points.extend(new_food_points)

        # rebuild kdtree
        self.grass_kd_tree = self.build_grass_kd_tree()
