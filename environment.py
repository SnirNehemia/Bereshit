# environment.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

from config import Config as config


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
        self.tree_indices = np.argwhere(self.tree_mask)

        # Lists to hold dynamically generated vegetation points.
        self.grass_points = []
        self.leaf_points = []

        self.new_grass_points = []

        self.grass_kd_tree = self.build_grass_kd_tree()

    def update(self):
        """
        Generates new vegetation points (grass and leaves) based on generation rates.
        Note: Over time these lists may become large.
        """
        # Generate new grass points.
        if len(self.grass_points) + len(self.new_grass_points) >= config.MAX_GRASS_NUM:
            num_new_grass = 0
        else:
            num_new_grass = int(self.grass_generation_rate)
        if len(self.grass_indices) > 0 and num_new_grass > 0:
            if np.random.rand() < config.GRASS_GROWTH_CHANCE:
                choices = self.grass_indices[np.random.choice(len(self.grass_indices), num_new_grass, replace=True)]
                for pt in choices:
                    # Convert image coordinates (row, col) to (x, y)
                    self.new_grass_points.append([pt[1], pt[0]])

        # Generate new leaf points.
        if len(self.leaf_points) >= config.MAX_LEAVES_NUM:
            num_new_leaves = 0
        else:
            num_new_leaves = int(self.leaves_generation_rate)
        if len(self.tree_indices) > 0 and num_new_leaves > 0:
            choices = self.tree_indices[np.random.choice(len(self.tree_indices), num_new_leaves, replace=True)]
            for pt in choices:
                self.leaf_points.append([pt[1], pt[0]])

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
        :return: add the new grass points to the grass list and update the KDTree
        """
        self.grass_points.extend(self.new_grass_points)
        self.new_grass_points = []
        self.grass_kd_tree = self.build_grass_kd_tree()
