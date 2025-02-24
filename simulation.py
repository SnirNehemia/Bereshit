# simulation.py
import numpy as np
from scipy.spatial import KDTree
from static_traits import StaticTraits
from creature import Creature
from environment import Environment

FOOD_DISTANCE_THRESHOLD = 5
LEAF_HEIGHT = 10
GRASS_ENERGY = 2
LEAF_ENREGY = 5


def prepare_eye_input(detection_result, vision_limit):
    """
    Converts a detection result (distance, signed_angle) or None into a 3-element vector:
      [detection_flag, distance, angle].
    """
    if detection_result is None:
        return np.array([0, vision_limit, 0])
    else:
        distance, angle = detection_result
        return np.array([1, distance, angle])


class Simulation:
    """
    Manages the simulation of creatures within an environment.
    Handles perception, decision-making, movement, collision detection, and vegetation updates.
    """

    def __init__(self, creatures: list[Creature], environment: Environment):
        self.creatures = creatures
        self.env = environment
        self.kd_tree = self.build_kdtree()

    def build_kdtree(self) -> KDTree:
        """
        Builds a KDTree for fast spatial queries based on creature positions.
        """
        positions = [creature.position for creature in self.creatures]
        if positions:
            return KDTree(positions)
        else:
            return KDTree([[0, 0]])

    def update_kdtree(self):
        """
        Rebuilds the KDTree after creatures have moved.
        """
        self.kd_tree = self.build_kdtree()

    def efficient_detect_target(self, creature: StaticTraits, eye_params: tuple, noise_std: float = 0.0):
        """
        Uses the specified eye (given by eye_params: (angle_offset, aperture))
        to detect a nearby target.
        Computes the eye's viewing direction by rotating the creature's heading by angle_offset.
        Returns (distance, signed_angle) if a target is found within half the aperture, else None.
        """
        eye_position = creature.position  # Eye is assumed at the creature's position.
        heading = creature.get_heading()
        angle_offset, aperture = eye_params

        # Rotate the heading vector by angle_offset.
        cos_offset = np.cos(angle_offset)
        sin_offset = np.sin(angle_offset)
        eye_direction = np.array([
            heading[0] * cos_offset - heading[1] * sin_offset,
            heading[0] * sin_offset + heading[1] * cos_offset
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
            # Compute the signed angle using arctan2.
            det = eye_direction[0] * target_direction[1] - eye_direction[1] * target_direction[0]
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
        """
        Advances the simulation by one time step.
        For each creature:
          - Perceives its surroundings with both eyes.
          - Constructs an input vector for the brain.
          - Receives a decision (delta_angle, delta_speed) to update its velocity.
          - Checks for collisions with obstacles (black areas) and stops if necessary.
        Then, moves creatures and updates the vegetation.
        """
        # Update each creature's velocity.
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
            delta_angle = np.clip(delta_angle, -0.1, 0.1)
            delta_speed = np.clip(delta_speed, -1, 1)

            current_speed = creature.speed
            current_speed_mag = np.linalg.norm(current_speed)
            if current_speed_mag == 0:
                current_direction = np.array([1.0, 0.0])
            else:
                current_direction = current_speed / current_speed_mag

            cos_angle = np.cos(delta_angle)
            sin_angle = np.sin(delta_angle)
            new_direction = np.array([
                current_direction[0] * cos_angle - current_direction[1] * sin_angle,
                current_direction[0] * sin_angle + current_direction[1] * cos_angle
            ])
            new_speed_mag = np.clip(current_speed_mag + delta_speed, 0, creature.max_speed)
            creature.speed = new_direction * new_speed_mag

        # Collision detection: if a creature's new position would be inside an obstacle, stop it.
        for creature in self.creatures:
            new_position = creature.position + creature.speed * dt
            # Convert (x, y) to image indices (col, row).
            col = int(new_position[0])
            row = int(new_position[1])
            height, width = self.env.map_data.shape[:2]
            if col < 0 or col >= width or row < 0 or row >= height:
                creature.speed = np.array([0.0, 0.0])
            else:
                if self.env.obstacle_mask[row, col]:
                    creature.speed = np.array([0.0, 0.0])

        for creature in self.creatures:
            # death from age
            if creature.age >= creature.max_age:
                self.creatures.remove(creature)
                continue
            else:
                creature.age += 1

            # check energy
            energy_consumption = 0
            energy_consumption += creature.energy_efficiency  # idle energy
            energy_consumption += creature.speed_efficiency * np.linalg.norm(creature.speed)  # movement energy

            # update or kill creature
            if creature.energy > energy_consumption:
                # update creature energy and position
                creature.energy -= energy_consumption
                creature.position += creature.speed * dt

                # check for food
                is_found_food = self.eat_food(creature=creature, food_type='grass')
                if not is_found_food and creature.height >= LEAF_HEIGHT:
                    _ = self.eat_food(creature=creature, food_type='leaf')

            else:
                # death from energy
                self.creatures.remove(creature)

        self.update_kdtree()
        # Update environment vegetation.
        self.env.update()

    def eat_food(self, creature: Creature, food_type: str):
        is_found_food = False

        if food_type == 'grass':
            food_points = self.env.grass_points
            food_energy = GRASS_ENERGY
        elif food_type == 'leaf':
            food_points = self.env.leaf_points
            food_energy = LEAF_ENREGY

        if food_points:
            food_distances = [np.linalg.norm(food_point - creature.position)
                              for food_point in food_points]

            if np.min(food_distances) <= FOOD_DISTANCE_THRESHOLD:
                # update creature energy
                creature.energy += creature.food_efficiency * food_energy

                # remove food from board
                closest_food_point = self.env.grass_points[np.argmin(food_distances)]
                self.env.grass_points.remove(closest_food_point)
                is_found_food = True

        return is_found_food

    def run_and_visualize(self, dt: float, noise_std: float,
                          frames: int, save_filename: str = "simulation.mp4"):
        """
        Runs the simulation for a given number of frames and saves an animation.
        Visualizes:
          - The environment map with semi-transparent overlay (using origin='lower').
          - The water source, vegetation (grass and leaves) with outlines.
          - Creatures as colored dots with arrows indicating heading.
        Prints progress every 10 frames.
        """
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.patches import Circle
        global quiv, scat, grass_scat, leaves_scat

        fig, ax = plt.subplots(figsize=(8, 8))
        extent = self.env.get_extent()
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_title("Evolution Simulation")

        # Display the environment map with origin='lower' to avoid vertical mirroring.
        ax.imshow(self.env.map_data, extent=extent, alpha=0.3, origin='lower')  # , aspect='auto')

        # Draw the water source.
        water_x, water_y, water_r = self.env.water_source
        water_circle = Circle((water_x, water_y), water_r, color='blue', alpha=0.3)
        ax.add_patch(water_circle)

        # Initial creature positions.
        positions = np.array([creature.position for creature in self.creatures])
        colors = [creature.color for creature in self.creatures]
        scat = ax.scatter(positions[:, 0], positions[:, 1], c=colors, s=20)

        # Create quiver arrows for creature headings.
        U, V = [], []
        for creature in self.creatures:
            if np.linalg.norm(creature.speed) > 0:
                U.append(creature.speed[0])
                V.append(creature.speed[1])
            else:
                U.append(0)
                V.append(0)
        quiv = ax.quiver(positions[:, 0], positions[:, 1], U, V,
                         color='black', scale=150, width=0.005)

        # Scatter plots for vegetation.
        grass_scat = ax.scatter([], [], c='lightgreen', edgecolors='black', s=20)
        leaves_scat = ax.scatter([], [], c='darkgreen', edgecolors='black', s=20)

        def update(frame):
            self.step(dt, noise_std)

            # clear quiver and scatter
            global quiv, scat, grass_scat, leaves_scat
            if 'quiv' in globals():
                quiv.remove()
            if 'scat' in globals():
                scat.remove()
            if 'grass_scat' in globals():
                grass_scat.remove()
            if 'leaves_scat' in globals():
                leaves_scat.remove()

            # Update creature positions.
            positions = np.array([creature.position for creature in self.creatures])
            colors = [creature.color for creature in self.creatures]
            if len(positions) > 0:
                scat = ax.scatter(positions[:, 0], positions[:, 1], c=colors, s=20)
            # scat.set_offsets(positions)

                U, V = [], []
                for creature in self.creatures:
                    if np.linalg.norm(creature.speed) > 0:
                        U.append(creature.speed[0])
                        V.append(creature.speed[1])
                    else:
                        U.append(0)
                        V.append(0)

                quiv = ax.quiver(positions[:, 0], positions[:, 1], U, V,
                                 color='black', scale=150, width=0.005)
                # quiv.set_offsets(positions)
                # quiv.set_UVC(U, V)
            else:
                print('all creatures are dead :(')
                scat = ax.scatter([1], [1])
                quiv = ax.quiver([1], [1], [1], [1])

            # Update vegetation scatter data.
            if len(self.env.grass_points) > 0:
                grass_points = np.array(self.env.grass_points)
                grass_scat = ax.scatter(grass_points[:, 0], grass_points[:, 1], c='lightgreen', edgecolors='black', s=20)
                # grass_scat.set_offsets(np.array(self.env.grass_points))
            if len(self.env.leaf_points) > 0:
                leaf_points = np.array(self.env.leaf_points)
                leaves_scat = ax.scatter(leaf_points[:, 0], leaf_points[:, 1], c='darkgreen', edgecolors='black', s=20)
                # leaves_scat.set_offsets(np.array(self.env.leaf_points))
            if frame % 10 == 0:
                print(f"Frame {frame} / {frames}")

            return scat, quiv, grass_scat, leaves_scat

        ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
        ani.save(save_filename, writer="ffmpeg", dpi=200)
        plt.close(fig)
