# simulation.py
import numpy as np
from scipy.spatial import KDTree

from brain import Brain
from creature import Creature
from environment import Environment
from tqdm import tqdm
import config as config

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle


class Simulation:
    """
    Manages the simulation of creatures within an environment.
    Handles perception, decision-making, movement, collision detection, and vegetation updates.
    Implements multi-channel perception by using separate KDTree queries for each target type.
    """

    def __init__(self):
        # Create the environment. Ensure that 'map.png' exists and follows the color conventions.
        self.env = Environment(map_filename=config.ENV_PATH,
                               grass_generation_rate=config.GRASS_GENERATION_RATE,
                               leaves_generation_rate=config.LEAVES_GENERATION_RATE)

        # Initialize creatures (ensuring they are not in forbidden areas).
        self.creatures = self.initialize_creatures(num_creatures=config.NUM_CREATURES,
                                                   simulation_space=config.SIMULATION_SPACE,
                                                   input_size=config.INPUT_SIZE,
                                                   output_size=config.OUTPUT_SIZE,
                                                   eyes_params=config.EYES_PARAMS,
                                                   env=self.env)
        self.birthday = {id: 0 for id in self.creatures.keys()}
        # Build a KDTree for creature positions.
        self.creatures_kd_tree = self.build_creatures_kd_tree()
        self.max_creature_id = len(self.creatures.keys()) - 1

        # TODO: add dictionary for dead creatures

        # debug run
        self.creatures_energy_per_frame = dict([(id, list()) for id in range(len(self.creatures.keys()))])
        self.num_creatures_per_frame = []
        self.min_creature_energy_per_frame = []
        self.max_creature_energy_per_frame = []
        self.mean_creature_energy_per_frame = []
        self.std_creature_energy_per_frame = []

    @staticmethod
    def initialize_creatures(num_creatures, simulation_space, input_size, output_size,
                             eyes_params, env: Environment):
        """
        Initializes creatures ensuring they are not placed in a forbidden (black) area.
        """
        creatures = dict()
        for id in range(num_creatures):
            position = []
            valid_position = False
            while not valid_position:
                position = np.random.rand(2) * simulation_space
                # Convert (x, y) to indices (col, row)
                col, row = int(position[0]), int(position[1])
                height, width = env.map_data.shape[:2]
                # Check bounds and obstacle mask.
                if col < 0 or col >= width or row < 0 or row >= height:
                    continue
                if env.obstacle_mask[row, col]:
                    continue
                valid_position = True

            # static traits
            max_age = 50
            max_weight = 10.0
            max_height = 5.0
            max_speed = [5.0, 5.0]
            color = np.random.rand(3)  # Random RGB color.

            energy_efficiency = 1   # idle energy
            speed_efficiency = 0.1  # speed * speed_efficiency
            food_efficiency = 1     # energy from food * food_efficiency
            reproduction_energy = config.REPRODUCTION_ENERGY
            max_energy = config.MAX_INIT_ENERGY

            vision_limit = 100.0
            brain = Brain([input_size, output_size])

            # dynamic traits
            weight = np.random.rand() * max_weight
            height = np.random.rand() * max_height
            speed = (np.random.rand(2) - 0.5) * max_speed
            energy = np.random.rand() * max_energy
            hunger = np.random.rand() * 10
            thirst = np.random.rand() * 10

            # init creature
            creature = Creature(max_age=max_age, max_weight=max_weight, max_height=max_height, max_speed=max_speed,
                                color=color,
                                energy_efficiency=energy_efficiency, speed_efficiency=speed_efficiency,
                                food_efficiency=food_efficiency, reproduction_energy=reproduction_energy,
                                max_energy=max_energy,
                                eyes_params=eyes_params, vision_limit=vision_limit, brain=brain,
                                weight=weight, height=height,
                                position=position, speed=speed, energy=energy, hunger=hunger, thirst=thirst)

            creatures[id] = creature
        return creatures

    def build_creatures_kd_tree(self) -> KDTree:
        """
        Builds a KDTree from the positions of all creatures.
        """
        positions = [creature.position for creature in self.creatures.values()]
        if positions:
            return KDTree(positions)
        else:
            return KDTree([[0, 0]])

    def update_creatures_kd_tree(self):
        self.creatures_kd_tree = self.build_creatures_kd_tree()

    def seek(self, creature: Creature, noise_std: float = 0.0):
        """
        Uses the specified eye (given by eye_params: (angle_offset, aperture))
        to detect a nearby target.
        Computes the eye's viewing direction by rotating the creature's heading by angle_offset.
        Returns (distance, signed_angle) if a target is found within half the aperture, else None.
        """
        channel_results = {}
        channels_order = ['grass', 'leaves', 'water', 'creatures']
        channels_list = []
        for i_eye, eye_params in enumerate(creature.eyes_params):
            for channel in channels_order:
                candidate_points = np.array([])
                if channel == 'grass':
                    if len(self.env.grass_points) > 0:
                        candidate_points = np.array(self.env.grass_points)
                elif channel == 'leaves':
                    if len(self.env.leaf_points) > 0:
                        candidate_points = np.array(self.env.leaf_points)
                elif channel == 'water':
                    candidate_points = np.array([[self.env.water_source[0], self.env.water_source[1]]])
                elif channel == 'creatures':
                    candidate_points = np.array([c.position for c in self.creatures.values()])

                if len(candidate_points) > 0:
                    kd_tree = KDTree(candidate_points)
                    result = self.detect_target_from_kdtree(creature, kd_tree, candidate_points, noise_std)
                else:
                    result = None

                channel_name = f'{channel}_{i_eye}'
                channel_results[channel_name] = result
                channels_list.append(channel_name)

        return channel_results

    def use_brain(self, creature: Creature, noise_std: float = 0.0):
        try:
            seek_results = self.seek(creature, noise_std)
            eyes_inputs = [self.prepare_eye_input(seek_results, creature.vision_limit) for seek_results in
                           seek_results.values()]
            brain_input = np.concatenate([
                np.array([creature.hunger, creature.thirst]),
                creature.speed,
                np.concatenate(eyes_inputs)
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
        except Exception as e:
            print(e)
            breakpoint()

    @staticmethod
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

    @staticmethod
    def detect_target_from_kdtree(creature: Creature, kd_tree: KDTree, candidate_points: np.ndarray,
                                  noise_std: float = 0.0):
        """
        Generic function to detect the closest target from candidate_points using a KDTree.

        Parameters:
          creature: the creature performing the detection.
          eye_params: (angle_offset, aperture) specifying the eye's viewing direction relative to the creature's heading.
          kd_tree: a KDTree built from candidate_points.
          candidate_points: numpy array of shape (N, 2) containing candidate target positions.
          noise_std: standard deviation for optional Gaussian noise.

        Returns:
          A tuple (distance, signed_angle) for the detected target, or None if no target qualifies.
        """
        eye_position = creature.position
        heading = creature.get_heading()
        angle_offset, aperture = creature.eyes_params
        # Compute the eye's viewing direction by rotating the heading by angle_offset.
        cos_offset = np.cos(angle_offset)
        sin_offset = np.sin(angle_offset)
        eye_direction = np.array([
            heading[0] * cos_offset - heading[1] * sin_offset,
            heading[0] * sin_offset + heading[1] * cos_offset
        ])
        # Query the KDTree for candidate indices within the creature's vision range.
        candidate_indices = kd_tree.query_ball_point(eye_position, creature.vision_limit)
        best_distance = float('inf')
        detected_info = None
        # Evaluate each candidate.
        for idx in candidate_indices:
            candidate = candidate_points[idx]
            # Skip if the candidate is the creature itself.
            if np.allclose(candidate, creature.position):
                continue
            target_vector = candidate - eye_position
            distance = np.linalg.norm(target_vector)
            if distance == 0 or distance > creature.vision_limit:
                continue
            target_direction = target_vector / distance
            dot = np.dot(eye_direction, target_direction)
            det = eye_direction[0] * target_direction[1] - eye_direction[1] * target_direction[0]
            angle = np.arctan2(det, dot)
            # Only accept targets within half the aperture.
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
        for id, creature in self.creatures.items():
            # print(f'creature {i}: start brain use...')
            self.use_brain(creature=creature, noise_std=noise_std)
            # print(f'creature {i}: completed brain use!')

        # Collision detection: if a creature's new position would be inside an obstacle, stop it.
        for id, creature in self.creatures.items():
            new_position = creature.position + creature.speed * dt
            # Convert (x, y) to image indices (col, row).
            col = int(new_position[0])
            row = int(new_position[1])
            height, width = self.env.map_data.shape[:2]
            if col < 0 or col >= width or row < 0 or row >= height or self.env.obstacle_mask[row, col]:
                creature.speed = np.array([0.0, 0.0])
            else:  # TODO: check if neccessary
                if self.env.obstacle_mask[row, col]:
                    creature.speed = np.array([0.0, 0.0])

        creatures_reproduced = []
        died_creatured_id = []

        # energy consumption
        for id, creature in self.creatures.items():
            # death from age
            if creature.age >= creature.max_age:
                died_creatured_id.append(id)
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

                # check for food (first grass, if not found search for leaf if tall enough)
                is_found_food = self.eat_food(id=id, food_type='grass')
                if not is_found_food and creature.height >= config.LEAF_HEIGHT:
                    _ = self.eat_food(id=id, food_type='leaf')
                creature.log_eat.append(is_found_food)

                # reproduce
                if creature.energy > creature.reproduction_energy + config.MIN_LIFE_ENERGY:
                    creature.energy -= creature.reproduction_energy
                    creatures_reproduced.append(creature)
                    creature.log_reproduce.append(1)
                else:
                    creature.log_reproduce.append(0)
            else:
                # death from energy
                died_creatured_id.append(id)
            creature.log_energy.append(creature.energy)

        # kill creatures
        for id in died_creatured_id:
            del self.creatures[id]  # TODO: move to a cemetery dictionary
        child_ids = []
        # Reproduction
        for creature in creatures_reproduced:
            # update id
            id = self.max_creature_id + 1
            self.max_creature_id += 1

            # Mutate father attributes for child
            child_attributes = creature.__dict__.copy()
            del child_attributes['age']

            for key in creature.log_list:
                del child_attributes[key]
            del child_attributes['log_list']
            # child_attributes.pop()
            mutation_binary_mask = np.random.rand(len(child_attributes)) > config.MUTATION_THRESHOLD  # which traits to change

            for i, key in enumerate(child_attributes.keys()):
                if key.startswith('log'):  # clear logs for child
                    child_attributes[key] = []
                    continue
                do_mutate = mutation_binary_mask[i]
                max_mutation_factor = config.MAX_MUTATION_FACTORS[key]
                if do_mutate:
                    if key == "brain":
                        brain_mutation_rate = dict()
                        max_brain_mutation_rate = max_mutation_factor
                        for brain_mutation_rate_key in max_mutation_factor.keys():
                            mutation_roll = np.random.rand()
                            brain_mutation_rate[brain_mutation_rate_key] = \
                                mutation_roll * max_brain_mutation_rate[brain_mutation_rate_key]
                        creature.brain.mutate_brain(brain_mutation_rate=brain_mutation_rate)
                    else:
                        # check if attribute contain number or array
                        try:
                            num_to_rand = len(child_attributes[key])
                        except:
                            num_to_rand = 1

                        # mutate attribute
                        mutation_roll = np.random.rand(num_to_rand) - 0.5  # so it will be between -0.5 and 0.5
                        child_attributes[key] += mutation_roll * max_mutation_factor

                        # change array with size 1 back to float
                        if num_to_rand == 1:
                            child_attributes[key] = float(child_attributes[key])

            # Add child to creatures
            child_creature = Creature(**child_attributes)
            self.creatures[id] = child_creature
            child_ids.append(id)

        self.update_creatures_kd_tree()
        # Update environment vegetation.
        self.env.update()

        return child_ids

    def eat_food(self, id: id, food_type: str):
        creature = self.creatures[id]
        is_found_food = False
        if creature.energy >= creature.max_energy:
            return False

        # get food points
        food_points, food_energy = [], 0
        if food_type == 'grass':
            food_points = self.env.grass_points
            food_energy = config.GRASS_ENERGY
        elif food_type == 'leaf':
            food_points = self.env.leaf_points
            food_energy = config.LEAF_ENERGY

        if len(food_points) > 0:
            food_distances = [np.linalg.norm(food_point - creature.position)
                              for food_point in food_points]

            if np.min(food_distances) <= config.FOOD_DISTANCE_THRESHOLD:
                # update creature energy
                creature.energy += creature.food_efficiency * food_energy

                # remove food from board
                closest_food_point = self.env.grass_points[np.argmin(food_distances)]
                self.env.grass_points.remove(closest_food_point)
                is_found_food = True

        return is_found_food

    def run_and_visualize(self):
        """
        Runs the simulation for a given number of frames and saves an animation.
        Visualizes:
          - The environment map with semi-transparent overlay (using origin='lower').
          - The water source, vegetation (grass and leaves) with outlines.
          - Creatures as colored dots with arrows indicating heading.
        Prints progress every 10 frames.
        """

        # -------------------------- init relevant parameters for simulation -------------------------- #
        global quiv, scat, grass_scat, leaves_scat

        dt = config.DT
        noise_std = config.NOISE_STD
        num_frames = config.NUM_FRAMES

        # Initialize the progress bar outside of the update function
        progress_bar = tqdm(total=num_frames, desc="Simulation progress")
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
        positions = np.array([creature.position for creature in self.creatures.values()])
        colors = [creature.color for creature in self.creatures.values()]
        scat = ax.scatter(positions[:, 0], positions[:, 1], c=colors, s=20)

        # Create quiver arrows for creature headings.
        U, V = [], []
        for creature in self.creatures.values():
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

        # -------------------------------- function for simulation progress -------------------------------- #

        print('starting simulation')

        def update(frame):

            # --------------------------- run frame --------------------------- #
            child_ids = self.step(dt, noise_std)
            # update birthdays
            for child_id in child_ids:
                self.birthday[child_id] = frame
            # ------------------------- update debug parameters ------------------------- #
            current_num_creatures = len(self.creatures.keys())
            if current_num_creatures > config.MAX_NUM_CREATURES:
                raise Exception(f'{frame=}: Too many creatures, simulation is stuck.')
            if current_num_creatures > 0:
                self.num_creatures_per_frame.append(current_num_creatures)
                creatures_energy = [creature.energy for creature in self.creatures.values()]
                self.min_creature_energy_per_frame.append(np.min(creatures_energy))
                self.max_creature_energy_per_frame.append(np.max(creatures_energy))
                self.mean_creature_energy_per_frame.append(np.mean(creatures_energy))
                self.std_creature_energy_per_frame.append(np.std(creatures_energy))

                for id, creature in self.creatures.items():
                    if id not in self.creatures_energy_per_frame.keys():
                        self.creatures_energy_per_frame[id] = np.zeros(frame).tolist()
                        self.creatures_energy_per_frame[id].append(creature.energy)
                    else:
                        self.creatures_energy_per_frame[id].append(creature.energy)

                print(f'{frame=}: ended with {current_num_creatures} creatures (+{len(child_ids)}), '
                      f' max energy = {round(self.max_creature_energy_per_frame[-1], 2)}.')
            else:
                print(f'{frame=}: all creatures are dead :(.')

            # --------------------------- plot --------------------------- #
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
            positions = np.array([creature.position for creature in self.creatures.values()])
            colors = [creature.color for creature in self.creatures.values()]
            if len(positions) > 0:
                scat = ax.scatter(positions[:, 0], positions[:, 1], c=colors, s=20)
                # scat.set_offsets(positions)

                U, V = [], []
                for creature in self.creatures.values():
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
                scat = ax.scatter([1], [1])
                quiv = ax.quiver([1], [1], [1], [1])

            # Update vegetation scatter data.
            if len(self.env.grass_points) > 0:
                grass_points = np.array(self.env.grass_points)
                grass_scat = ax.scatter(grass_points[:, 0], grass_points[:, 1], c='lightgreen', edgecolors='black',
                                        s=20)
                # grass_scat.set_offsets(np.array(self.env.grass_points))
            if len(self.env.leaf_points) > 0:
                leaf_points = np.array(self.env.leaf_points)
                leaves_scat = ax.scatter(leaf_points[:, 0], leaf_points[:, 1], c='darkgreen', edgecolors='black', s=20)
                # leaves_scat.set_offsets(np.array(self.env.leaf_points))
            # if frame % 10 == 0:
            #     print(f"Frame {frame} / {num_frames}")
            # Update the progress bar
            progress_bar.update(1)
            return scat, quiv, grass_scat, leaves_scat

        # ----------------------------------- run simulation and save animation ------------------------------------ #
        ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)
        ani.save(config.ANIMATION_FILEPATH, writer="ffmpeg", dpi=200)
        plt.close(fig)
        print('finished simulation.')
        print(f'Simulation animation saved as {config.ANIMATION_FILEPATH.stem}.')

        # ----------------------------------- Plot graphs after simulation ended ----------------------------------- #
        # creature energy fig
        plt.figure()
        creature_ids_to_plot = [0, 1, 2, 3, 4, 5]
        for id in creature_ids_to_plot:
            plt.plot(self.creatures_energy_per_frame[id], '.-', label=f'{id=}')
        plt.axhline(y=config.REPRODUCTION_ENERGY + config.MIN_LIFE_ENERGY,
                    linestyle='--', color='r', label='reproduction threshold')
        plt.legend()
        plt.title('creature energy per frame')
        plt.ylabel('creature energy')
        plt.xlabel('frame number')
        plt.savefig(fname=config.SPECIFIC_FIG_FILEPATH)
        print(f'specific fig saved as {config.SPECIFIC_FIG_FILEPATH.stem}.')

        # Final fig
        fig, ax = plt.subplots(2, 1, sharex='all')
        ax[0].plot(self.num_creatures_per_frame, '.-')
        ax[0].set_title('num creatures per frame')
        ax[1].plot(self.min_creature_energy_per_frame, '.-', label='min energy')
        ax[1].plot(self.max_creature_energy_per_frame, '.-', label='max energy')
        ax[1].errorbar(x=np.arange(len(self.mean_creature_energy_per_frame)),
                       y=self.mean_creature_energy_per_frame,
                       yerr=self.std_creature_energy_per_frame, linestyle='-', marker='.', label='mean and std energy')
        ax[1].axhline(y=list(self.creatures.values())[0].reproduction_energy + config.MIN_LIFE_ENERGY,
                      linestyle='--', color='r', label='reproduction threshold')
        ax[1].set_title('energy statistics per frame')
        ax[1].set_xlabel('frame number')
        ax[1].legend()
        fig.savefig(fname=config.STATISTICS_FIG_FILEPATH)
        print(f'statistics fig saved as {config.STATISTICS_FIG_FILEPATH.stem}.')

        plt.show()
