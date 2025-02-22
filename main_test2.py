import config
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import KDTree

def prepare_eye_input(detection_result, vision_limit):
    """
    Convert a detection result (distance, angle) or None into a consistent 3-element vector:
      [detection_flag, distance, angle].
    If no target is detected, flag=0, distance=vision_limit, angle=0.
    """
    if detection_result is None:
        return np.array([0, vision_limit, 0])
    else:
        distance, angle = detection_result
        return np.array([1, distance, angle])

class Brain:
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        # Simple linear weights as a placeholder for the neural network.
        self.weights = np.random.randn(input_size, output_size)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Linear transformation: inputs -> decision vector.
        return np.dot(inputs, self.weights)

class StaticTraits:
    def __init__(self, position: np.ndarray, max_size: float, max_speed: float,
                 left_eye_params: tuple, right_eye_params: tuple,
                 vision_limit: float, brain: Brain):
        """
        left_eye_params and right_eye_params are tuples: (angle_offset, aperture),
        where angle_offset is in radians (relative to creature's heading) and
        aperture is the total field-of-view of that eye.
        """
        self.position = position
        self.max_size = max_size
        self.max_speed = max_speed
        self.left_eye_params = left_eye_params  # e.g., (np.radians(30), np.radians(60))
        self.right_eye_params = right_eye_params  # e.g., (np.radians(-30), np.radians(60))
        self.vision_limit = vision_limit
        self.brain = brain

    def get_heading(self):
        """
        Returns the creature's heading as a normalized 2D vector.
        If the creature is stationary, defaults to (1, 0).
        """
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
        self.color = color      # 3-element (r, g, b) vector.

class Simulation:
    def __init__(self, creatures):
        self.creatures = creatures
        self.kd_tree = self.build_kdtree()

    def build_kdtree(self) -> KDTree:
        positions = [creature.position for creature in self.creatures]
        return KDTree(positions)

    def update_kdtree(self):
        self.kd_tree = self.build_kdtree()

    def efficient_detect_target(self, creature: StaticTraits, eye_params: tuple, noise_std: float = 0.0):
        """
        Uses an eye (specified by eye_params: (angle_offset, aperture)) to detect a target.
        The eye's viewing direction is computed by rotating the creature's heading by the given angle offset.
        Only candidates within half the aperture (to either side) are considered.
        Returns a tuple (distance, signed_angle) or None.
        """
        eye_position = creature.position  # Eyes are considered at the creature's position.
        heading = creature.get_heading()
        angle_offset, aperture = eye_params
        # Rotate heading by angle_offset to get the eye's viewing direction.
        cos_offset = np.cos(angle_offset)
        sin_offset = np.sin(angle_offset)
        eye_direction = np.array([
            heading[0]*cos_offset - heading[1]*sin_offset,
            heading[0]*sin_offset + heading[1]*cos_offset
        ])
        candidate_indices = self.kd_tree.query_ball_point(eye_position, creature.vision_limit)
        best_candidate = None
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
            # Only consider targets within half the aperture.
            if abs(angle) > (aperture / 2):
                continue
            if noise_std > 0:
                distance += np.random.normal(0, noise_std)
                angle += np.random.normal(0, noise_std)
            if distance < best_distance:
                best_distance = distance
                detected_info = (distance, angle)
                best_candidate = target
        return detected_info

    def step(self, dt: float, noise_std: float = 0.0):
        """
        Advances the simulation one time step.
        For each creature:
          - Perceive using both eyes (which now use angular offsets relative to heading).
          - Build the brain input vector: [hunger, thirst, speed_x, speed_y,
                                            left_eye (flag, distance, angle),
                                            right_eye (flag, distance, angle)]
          - Get the brain decision (assumed to be [delta_angle, delta_speed]).
          - Update the creature's velocity accordingly.
        Then update positions and refresh the KDTree.
        """
        delta_speeds = []
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
            delta_angle = np.clip(delta_angle, -0.1,0.1)
            delta_speed = np.clip(delta_speed, -1, 1)
            delta_speeds.append(delta_speed)
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
            new_speed_mag = current_speed_mag + delta_speed
            new_speed_mag = np.clip(new_speed_mag, 0, creature.max_speed)
            creature.speed = new_direction * new_speed_mag
        # plt.figure()
        plt.hist(delta_speeds)
        plt.show()
        # Move all creatures.
        for creature in self.creatures:
            creature.position += creature.speed * dt

        self.update_kdtree()

    def run_and_visualize(self, dt: float, noise_std: float,
                          frames: int, save_filename: str = "simulation.mp4"):
        """
        Runs the simulation for a given number of frames and creates an animation.
        Each creature is drawn as a colored dot with an arrow indicating its heading.
        Also prints progress every 10 frames.
        The resulting animation is saved to a video file.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        simulation_space = 1000
        ax.set_xlim(0, simulation_space)
        ax.set_ylim(0, simulation_space)
        ax.set_title("Evolution Simulation")
        positions = np.array([creature.position for creature in self.creatures])
        colors = [creature.color for creature in self.creatures]
        scat = ax.scatter(positions[:, 0], positions[:, 1], c=colors, s=20)
        # Quiver for heading arrows.
        U, V = [], []
        for creature in self.creatures:
            if np.linalg.norm(creature.speed) > 0:
                U.append(creature.speed[0])
                V.append(creature.speed[1])
            else:
                U.append(0)
                V.append(0)
        quiv = ax.quiver(positions[:, 0], positions[:, 1], U, V,
                         color='black', scale=100, width=0.005)

        def update(frame):
            self.step(dt, noise_std)
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
            if frame % 10 == 0:
                print(f"Frame {frame} / {frames}")
            return scat, quiv

        ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
        ani.save(save_filename, writer="ffmpeg", dpi=200)
        plt.close(fig)

if __name__ == "__main__":
    start_time = time.time()
    num_creatures = 200
    simulation_space = 1000
    creatures = []
    # Brain input: 10-dimensional; output: [delta_angle, delta_speed]
    input_size = 10
    output_size = 2
    # Define eye parameters: (angle_offset in radians, aperture in radians)
    left_eye_params = (np.radians(30), np.radians(60))    # Left eye: 30° offset, 60° aperture.
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

    sim = Simulation(creatures)
    noise_std = 0.5
    dt = 1.0
    frames = 200
    sim.run_and_visualize(dt, noise_std, frames, save_filename="simulation.mp4")
    total_time = time.time() - start_time
    print(f"Simulation animation saved as simulation.mp4")
    print(f"Total simulation time: {total_time:.2f} seconds")
