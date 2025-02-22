# environment.py
import numpy as np
import matplotlib.pyplot as plt


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
        # Create an obstacle mask: black areas where all channels are very low.
        self.obstacle_mask = np.all(self.map_data[:,:,:3] < 0.1, axis=2)
        # Grass regions: yellow (red and green high, blue low). Adjust thresholds as needed.
        self.grass_mask = (self.map_data[:, :, 0] > 0.7) & (self.map_data[:, :, 1] > 0.7) & (
                    self.map_data[:, :, 2] < 0.3)
        # Tree regions: green (green high, red and blue low). Adjust thresholds as needed.
        self.tree_mask = (self.map_data[:, :, 0] < 0.2) & (self.map_data[:, :, 1] > 0.6) & (
                    self.map_data[:, :, 2] < 0.5)

        # Define a water source as a blue circle in the center of the map.
        height, width, _ = self.map_data.shape
        self.water_source = (width // 2, height // 2, 50)

        self.grass_generation_rate = grass_generation_rate
        self.leaves_generation_rate = leaves_generation_rate

        # Pre-calculate indices for grass and tree regions for fast random sampling.
        self.grass_indices = np.argwhere(self.grass_mask)
        self.tree_indices = np.argwhere(self.tree_mask)

        # Lists to hold dynamically generated vegetation points.
        self.grass_points = []
        self.leaf_points = []

    def update(self):
        """
        Generates new vegetation points (grass and leaves) based on generation rates.
        Note: Over time these lists may become large.
        """
        # Generate new grass points.
        num_new_grass = int(self.grass_generation_rate)
        if len(self.grass_indices) > 0 and num_new_grass > 0:
            choices = self.grass_indices[np.random.choice(len(self.grass_indices), num_new_grass, replace=True)]
            for pt in choices:
                # Convert image coordinates (row, col) to (x, y)
                self.grass_points.append([pt[1], pt[0]])

        # Generate new leaf points.
        num_new_leaves = int(self.leaves_generation_rate)
        if len(self.tree_indices) > 0 and num_new_leaves > 0:
            choices = self.tree_indices[np.random.choice(len(self.tree_indices), num_new_leaves, replace=True)]
            for pt in choices:
                self.leaf_points.append([pt[1], pt[0]])

    def get_extent(self):
        """
        Returns the extent [xmin, xmax, ymin, ymax] to be used with imshow.
        """
        height, width, _ = self.map_data.shape
        return [0, width, 0, height]
