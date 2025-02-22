import config
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from scipy.spatial import KDTree

# ---------------- Environment Class ----------------
class Environment:
    def __init__(self, map_filename, grass_generation_rate, leaves_generation_rate):
        # Read the map PNG. Assume image is normalized (0-1).
        self.map_data = plt.imread(map_filename)
        # Create masks based on color.
        # Obstacles: black pixels (all channels < 0.1)
        self.obstacle_mask = np.all(self.map_data < 0.1, axis=2)
        # Grass regions: yellow (red and green high, blue low)
        self.grass_mask = np.logical_and(
            np.logical_and(self.map_data[:, :, 0] > 0.9, self.map_data[:, :, 1] > 0.9),
            self.map_data[:, :, 2] < 0.1
        )
        # Tree regions: green (green high, red and blue low)
        self.tree_mask = np.logical_and(
            np.logical_and(self.map_data[:, :, 0] < 0.1, self.map_data[:, :, 1] > 0.9),
            self.map_data[:, :, 2] < 0.1
        )
        # Define a water source manually (x, y, radius). Here, center water.
        height, width, _ = self.map_data.shape
        self.water_source = (width // 2, height // 2, 50)
        self.grass_generation_rate = grass_generation_rate
        self.leaves_generation_rate = leaves_generation_rate
        # Pre-calculate indices (row, col) for grass and tree regions.
        self.grass_indices = np.argwhere(self.grass_mask)
        self.tree_indices = np.argwhere(self.tree_mask)
        # Dynamic vegetation: store points as [x, y]
        self.grass_points = []
        self.leaf_points = []

    def update(self):
        # Generate new grass points.
        num_new_grass = int(self.grass_generation_rate)
        if len(self.grass_indices) > 0 and num_new_grass > 0:
            choices = self.grass_indices[np.random.choice(len(self.grass_indices), num_new_grass, replace=True)]
            for pt in choices:
                # Convert (row, col) to (x, y)
                self.grass_points.append([pt[1], pt[0]])
        # Generate new leaf points.
        num_new_leaves = int(self.leaves_generation_rate)
        if len(self.tree_indices) > 0 and num_new_leaves > 0:
            choices = self.tree_indices[np.random.choice(len(self.tree_indices), num_new_leaves, replace=True)]
            for pt in choices:
                self.leaf_points.append([pt[1], pt[0]])

    def get_extent(self):
        # Returns [xmin, xmax, ymin, ymax] for imshow.
        height, width, _ = self.map_data.shape
        return [0, width, 0, height]

# ---------------- Brain and Agent Classes ----------------
class Brain:
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        # Placeholder: random linear weights.
        self.weights = np.random.randn(input_size, output_size)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.dot(inputs, self.weights)

def prepare_eye_input(detection_result, vision_limit):
    """
    Converts a detection result (distance, signed_angle) or None
    into a 3-element vector: [detection_flag, distance, angle].
    """
    if detection_result is None:
        return np.array([0, vision_limit, 0])
    else:
        distance, angle = detection_result
        return np.array([1, distance, angle])

class StaticTraits:
    def __init__(self, position: np.ndarray, max_size: float, max_speed: float,
                 left_eye_params: tuple, right_eye_params: tuple,
                 vision_limit: float, brain: Brain):
        """
        left_eye_params and right_eye_params: (angle_offset, aperture)
        angle_offset is in radians (relative to creature's heading),
        and aperture is the total field-of-view for that eye.
        """
        self.position = position
        self.max_size = max_size
        self.max_speed = max_speed
        self.left_eye_params = left_eye_params
        self.right_eye_params = right_eye_params
        self.vision_limit = vision_limit
        self.brain = brain

    def get_heading(self):
        # Returns normalized heading from velocity if available; else default to (1, 0).
        if hasattr(self, 'speed') and np.linalg.norm(self.speed) > 0:
            return self.speed / np.linalg.norm(self.speed)
        else:
            return np.array([1.0, 0.0])

    def think(self, input_vector: np.ndarray) -> np.ndarray:
        return self.brain.forward(input_vector)

class Creature(StaticTraits):
    def __init__(self, position: np.ndarray, max_size: float, max_speed: float,
                 left_eye_params: tuple, right_eye_params: tuple,
                 vision_limit: float, brain: Brain, speed: np.ndarray,
                 hunger: float, thirst: float, color: np.ndarray):
        super().__init__(position, max_size, max_speed,
                         left_eye_params, right_eye_params, vision_limit, brain)
        self.speed = speed      # 2D velocity vector.
        self.hunger = hunger
        self.thirst = thirst
        self.color = color      # 3-element RGB vector.

# ---------------- Simulation Class ----------------
class Simulation:
    def __init__(self, creatures, environment):
        self.creatures = creatures
        self.env = environment
        self.kd_tree = self.build_kdtree()

    def build_kdtree(self) -> KDTree:
        positions = [creature.position for creature in self.creatures]
        return KDTree(positions)

    def update_kdtree(self):
        self.kd_tree = self.build_kdtree()

    def efficient_detect_target(self, creature: StaticTraits, eye_params: tuple, noise_std: float = 0.0):
        """
        Uses the specified eye (given by eye_params: (angle_offset, aperture))
        to detect a target. The eye's viewing direction is computed by rotating
        the creature's heading by the eye's angle_offset. A signed angle is computed
        (using arctan2) so that targets to the left and right yield different signs.
        Returns (distance, signed_angle) if a valid target is found, else None.
        """
        eye_position = creature.position  # Eye is at creature's position.
        heading = creature.get_heading()
        angle_offset, aperture = eye_params
        # Rotate the heading by angle_offset:
        cos_offset = np.cos(angle_offset)
        sin_offset = np.sin(angle_offset)
        eye_direction = np.array([
            heading[0]*cos_offset - heading[1]*sin_offset,
            heading[0]*sin_offset + heading[1]*cos_offset
        ])
        candidate_indices = self.kd_tree.query_ball_point(eye_position, creature.vision_limit)
        best_distance = float('inf')
        detected_info = None
        for idx in candidate_indices:
            target = self.creatures[idx]
            if target is creature:
                continue
            target_vector = target.position - eye_position
            distance = np.linalg.norm(target_vector)
            if distance == 0 or distance > creature.vision_limit:
                continue
            target_direction = target_vector / distance
            dot = np.dot(eye_direction, target_direction)
            # Compute signed angle using arctan2.
            det = eye_direction[0]*target_direction[1] - eye_direction[1]*target_direction[0]
            angle = np.arctan2(det, dot)
            # Accept only targets within half the aperture.
            if abs(angle) > (aperture / 2):
                continue
            if noise_std > 0:
                distance += np.random.normal(0, noise_std)
                angle += np.random.normal(0, noise_std)
            if distance < best_distance:
                best_distance = distance
                detected_info = (distance, angle)
        return detected_info

    def step(self, dt: float, noise_std: float = 0.0):
        # For each creature, perceive using both eyes, decide, and update velocity.
        for creature in self.creatures:
            left_detection = self.efficient_detect_target(creature, creature.left_eye_params, noise_std)
            right_detection = self.efficient_detect_target(creature, creature.right_eye_params, noise_std)
            left_eye_input = prepare_eye_input(left_detection, creature.vision_limit)
            right_eye_input = prepare_eye_input(right_detection, creature.vision_limit)
            brain_input = np.concatenate([
                np.array([creature.hunger, creature.thirst]),
                creature.speed,
                left_eye_input,
                right_eye_input
            ])
            decision = creature.think(brain_input)
            delta_angle, delta_speed = decision

            # Update velocity: rotate current heading by delta_angle and adjust speed.
            current_speed = creature.speed
            current_speed_mag = np.linalg.norm(current_speed)
            if current_speed_mag == 0:
                current_direction = np.array([1.0, 0.0])
            else:
                current_direction = current_speed / current_speed_mag

            cos_angle = np.cos(delta_angle)
            sin_angle = np.sin(delta_angle)
            new_direction = np.array([
                current_direction[0]*cos_angle - current_direction[1]*sin_angle,
                current_direction[0]*sin_angle + current_direction[1]*cos_angle
            ])
            new_speed_mag = np.clip(current_speed_mag + delta_speed, 0, creature.max_speed)
            creature.speed = new_direction * new_speed_mag

        # Move all creatures.
        for creature in self.creatures:
            creature.position += creature.speed * dt

        self.update_kdtree()
        # Update vegetation in the environment.
        self.env.update()

    def run_and_visualize(self, dt: float, noise_std: float,
                          frames: int, save_filename: str = "simulation.mp4"):
        """
        Runs the simulation for a number of frames and creates an animation.
        Draws:
          - The environment map (semi-transparent).
          - The water source as a blue circle.
          - Vegetation: grass (light green dots with thin black outlines)
            and leaves (dark green dots with thin black outlines).
          - Creatures as colored dots with heading arrows.
        Prints progress every 10 frames.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        extent = self.env.get_extent()
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_title("Evolution Simulation")

        # Display environment map.
        ax.imshow(self.env.map_data, extent=extent, alpha=0.3)

        # Draw water source.
        water_x, water_y, water_r = self.env.water_source
        water_circle = Circle((water_x, water_y), water_r, color='blue', alpha=0.3)
        ax.add_patch(water_circle)

        # Initial creature positions.
        positions = np.array([creature.position for creature in self.creatures])
        colors = [creature.color for creature in self.creatures]
        scat = ax.scatter(positions[:, 0], positions[:, 1], c=colors, s=20)
        # Quiver for creature heading.
        U, V = [], []
        for creature in self.creatures:
            if np.linalg.norm(creature.speed) > 0:
                U.append(creature.speed[0])
                V.append(creature.speed[1])
            else:
                U.append(0)
                V.append(0)
        quiv = ax.quiver(positions[:, 0], positions[:, 1], U, V,
                         color='black', scale=20, width=0.005)

        # Scatter plots for vegetation.
        grass_scat = ax.scatter([], [], c='lightgreen', edgecolors='black', s=20)
        leaves_scat = ax.scatter([], [], c='darkgreen', edgecolors='black', s=20)

        def update(frame):
            self.step(dt, noise_std)
            # Update creature positions.
            positions = np.array([creature.position for creature in self.creatures])
            scat.set_offsets(positions)
            U, V = [], []
            for creature in self.creatures:
                if np.linalg.norm(creature.speed) > 0:
                    U.append(creature.speed[0])
                    V.append(creature.speed[1])
                else:
                    U.append(0)
                    V.append(0)
            quiv.set_offsets(positions)
            quiv.set_UVC(U, V)
            # Update vegetation scatter data.
            if len(self.env.grass_points) > 0:
                grass_scat.set_offsets(np.array(self.env.grass_points))
            if len(self.env.leaf_points) > 0:
                leaves_scat.set_offsets(np.array(self.env.leaf_points))
            if frame % 10 == 0:
                print(f"Frame {frame} / {frames}")
            return scat, quiv, grass_scat, leaves_scat

        ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
        ani.save(save_filename, writer="ffmpeg", dpi=200)
        plt.close(fig)

# ---------------- Main Execution ----------------
if __name__ == "__main__":
    start_time = time.time()
    num_creatures = 200
    simulation_space = 1000
    creatures = []
    # Brain input: 10-dimensional; output: [delta_angle, delta_speed]
    input_size = 10
    output_size = 2
    # Define eye parameters: (angle_offset in radians, aperture in radians)
    left_eye_params = (np.radians(30), np.radians(60))    # Left eye: +30° offset, 60° aperture.
    right_eye_params = (np.radians(-30), np.radians(60))    # Right eye: -30° offset, 60° aperture.
    for _ in range(num_creatures):
        pos = np.random.rand(2) * simulation_space
        speed = (np.random.rand(2) - 0.5) * 5
        max_size = 10.0
        max_speed = 5.0
        vision_limit = 100.0
        brain = Brain(input_size, output_size)
        hunger = np.random.rand() * 10
        thirst = np.random.rand() * 10
        color = np.random.rand(3)  # Random RGB color.
        creature = Creature(pos, max_size, max_speed,
                            left_eye_params, right_eye_params,
                            vision_limit, brain, speed, hunger, thirst, color)
        creatures.append(creature)

    # Create environment from a PNG map.
    # (Ensure 'map.png' exists with black (walls), yellow (grass), green (trees))
    env = Environment("Penvs\\Env1.png", grass_generation_rate=5, leaves_generation_rate=3)
    sim = Simulation(creatures, env)

    noise_std = 0.5
    dt = 1.0
    frames = 200
    sim.run_and_visualize(dt, noise_std, frames, save_filename="simulation.mp4")
    total_time = time.time() - start_time
    print("Simulation animation saved as simulation.mp4")
    print(f"Total simulation time: {total_time:.2f} seconds")
