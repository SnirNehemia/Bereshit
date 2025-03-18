# simulation.py
import copy

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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes, mark_inset
import matplotlib.gridspec as gridspec

from lineage_tree import plot_lineage_tree


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
        self.dead_creatures = dict()

        self.birthday = {id: 0 for id in self.creatures.keys()}
        # Build a KDTree for creature positions.
        self.creatures_kd_tree = self.build_creatures_kd_tree()

        # self.max_creature_id = len(self.creatures.keys()) - 1
        self.children_num = 0
        # debug run
        self.creatures_energy_per_frame = dict([(id, list()) for id in range(len(self.creatures.keys()))])
        self.num_creatures_per_frame = []
        self.min_creature_energy_per_frame = []
        self.max_creature_energy_per_frame = []
        self.mean_creature_energy_per_frame = []
        self.std_creature_energy_per_frame = []
        self.num_new_creatures_per_frame = []
        self.num_dead_creatures_per_frame = []
        self.abort_simulation = False
        self.kdtree_update_interval = config.UPDATE_KDTREE_INTERVAL  # Set update interval for KDTree
        self.animation_update_interval = config.UPDATE_ANIMATION_INTERVAL  # Set update interval for animation frames
        self.frame_counter = 0  # Initialize frame counter  # TODO - should be steps_counter?
        self.id_count = config.NUM_CREATURES - 1
        self.focus_ID = 0
        # TODO: maybe add to the creature class a flag indicating if it survived a purge event and if so, it will be immune in the future
        self.purge = True  # flag for purge events
        self.creatures_history = []

    @staticmethod
    def initialize_creatures(num_creatures, simulation_space, input_size, output_size,
                             eyes_params, env: Environment):
        """
        Initializes creatures ensuring they are not placed in a forbidden (black) area.
        """
        creatures = dict()
        for creature_id in range(num_creatures):
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
            gen = 0
            parent_id = None
            birth_frame = 0
            max_age = np.random.randint(low=config.INIT_MAX_AGE * 0.8, high=config.INIT_MAX_AGE)
            max_weight = 10.0
            max_height = 5.0
            max_energy = config.INIT_MAX_ENERGY
            max_speed = config.MAX_SPEED  # TODO - should be max velocity
            color = np.random.rand(3)  # Random RGB color.

            energy_efficiency = 0.1  # idle energy
            motion_efficiency = 0.01  # speed * speed_efficiency
            food_efficiency = 1  # energy from food * food_efficiency
            reproduction_energy = config.REPRODUCTION_ENERGY

            vision_limit = config.VISION_LIMIT
            brain = Brain([input_size, output_size])

            # dynamic traits
            weight = np.random.rand() * max_weight
            height = np.random.rand() * max_height
            velocity = (np.random.rand(2) - 0.5) * max_speed

            # init creature
            creature = Creature(
                creature_id=creature_id, gen=gen, parent_id=parent_id, birth_frame=birth_frame,
                max_age=max_age, max_weight=max_weight, max_height=max_height,
                max_speed=max_speed, max_energy=max_energy, color=color,
                energy_efficiency=energy_efficiency, motion_efficiency=motion_efficiency,
                food_efficiency=food_efficiency, reproduction_energy=reproduction_energy,
                eyes_params=eyes_params, vision_limit=vision_limit, brain=brain,
                weight=weight, height=height,
                position=position, velocity=velocity)

            creatures[creature_id] = creature
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

        channels_list = []
        for i_eye, eye_params in enumerate(creature.eyes_params):
            for channel in config.EYE_CHANNEL:
                candidate_points = np.array([])
                if channel == 'grass':
                    if len(self.env.grass_points) > 0:
                        kd_tree = self.env.grass_kd_tree
                        candidate_points = np.array(self.env.grass_points)
                # elif channel == 'leaves':
                #     if len(self.env.leaf_points) > 0:
                #         candidate_points = np.array(self.env.leaf_points)
                # elif channel == 'water':
                #     candidate_points = np.array([[self.env.water_source[0], self.env.water_source[1]]])
                elif channel == 'creatures':
                    kd_tree = self.creatures_kd_tree
                    candidate_points = np.array([c.position for c in self.creatures.values()])

                if len(candidate_points) > 0:
                    # kd_tree = KDTree(candidate_points)
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
                creature.velocity,
                np.concatenate(eyes_inputs)
            ])
            decision = creature.think(brain_input)
            delta_angle, delta_speed = decision
            delta_angle = np.clip(delta_angle, -config.MAX_D_ANGLE, config.MAX_D_ANGLE)
            delta_speed = np.clip(delta_speed, -config.MAX_D_SPEED, config.MAX_D_SPEED)

            current_speed = creature.velocity
            current_speed_mag = creature.speed
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
            creature.velocity = new_direction * new_speed_mag
            creature.calc_speed()
        except Exception as e:
            print(e)

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

    def kill(self, sim_id):
        self.dead_creatures[sim_id] = self.creatures[sim_id]
        del self.creatures[sim_id]

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
        self.creatures_history.append([getattr(creature, 'creature_id') for creature in self.creatures.values()])
        # Update each creature's velocity.
        if config.DEBUG_MODE: print('seek')
        for creature_id, creature in self.creatures.items():
            # print(f'creature {creature_id}: start brain use...')
            self.use_brain(creature=creature, noise_std=noise_std)
            # print(f'creature {creature_id}: completed brain use!')

        # Collision detection: if a creature's new position would be inside an obstacle, stop it.
        if config.DEBUG_MODE: print('collision detection')
        for creature_id, creature in self.creatures.items():
            new_position = creature.position + creature.velocity * dt
            # Convert (x, y) to image indices (col, row).
            col = int(new_position[0])
            row = int(new_position[1])
            height, width = self.env.map_data.shape[:2]
            if col < 0 or col >= width or row < 0 or row >= height or self.env.obstacle_mask[row, col]:
                creature.velocity = np.array([0.0, 0.0])

        creatures_reproduced = []
        died_creatured_id = []

        # energy consumption
        if config.DEBUG_MODE: print('energy consumption')
        for creature_id, creature in self.creatures.items():
            # death from age
            if creature.age >= creature.max_age:
                died_creatured_id.append(creature_id)
                continue
            else:
                creature.age += 1

            # check energy
            energy_consumption = 0
            energy_consumption += creature.energy_efficiency  # idle energy
            energy_consumption += creature.motion_efficiency * creature.speed  # movement energy

            # test whether to update or kill creature
            if creature.energy > energy_consumption:
                # update creature energy and position
                creature.energy -= energy_consumption
                creature.position += creature.velocity * dt

                # check for food (first grass, if not found search for leaf if tall enough)
                is_found_food = self.eat_food(creature=creature, food_type='grass')
                if not is_found_food and creature.height >= config.LEAF_HEIGHT:
                    _ = self.eat_food(creature=creature, food_type='leaf')
                if is_found_food:
                    creature.log_eat.append(self.frame_counter)

                # reproduce
                if creature.energy > creature.reproduction_energy + config.MIN_LIFE_ENERGY:
                    # creature.energy -= creature.reproduction_energy
                    creatures_reproduced.append(creature)
                # else:
                #     creature.log_reproduce.append(0)
            else:
                # death from energy
                died_creatured_id.append(creature_id)
            creature.log_energy.append(creature.energy)

        # the purge
        # if (self.purge and len(creatures_reproduced) > 0) or len(self.creatures) > config.MAX_NUM_CREATURES * 0.75:
        if self.purge or len(self.creatures) > config.MAX_NUM_CREATURES * 0.75:
            purge_count = 0
            self.purge = False
            for id, creature in self.creatures.items():
                if (len(self.creatures) > config.MAX_NUM_CREATURES * 0.95 and np.random.rand(1) < 0.1
                        and id not in died_creatured_id):
                    purge_count += 1
                    died_creatured_id.append(id)
                if creature.max_speed_exp <= config.PURGE_SPEED_THRESHOLD and id not in died_creatured_id:
                    purge_count += 1
                    died_creatured_id.append(id)
            print(f'Purging {purge_count} creatures.')

        # kill creatures
        dead_ids = []
        for sim_id in died_creatured_id:
            self.kill(sim_id)
            dead_ids.append(sim_id)

        # Reproduction
        child_ids = []
        for creature in creatures_reproduced:
            child = creature.reproduce()
            creature.log_reproduce.append(self.frame_counter)
            child.birth_frame = self.frame_counter
            self.id_count += 1
            child.creature_id = self.id_count
            # self.creatures.append(child) # TODO: why do we use dict instead of list?
            self.creatures[self.id_count] = child
            child_ids.append(self.id_count)
            self.children_num += 1

        if config.DEBUG_MODE: print('update kdtree')
        # **Update KDTree every N frames**
        self.frame_counter += 1
        if self.frame_counter % self.kdtree_update_interval == 0:
            self.update_creatures_kd_tree()
            self.env.update_grass_kd_tree()
        if config.DEBUG_MODE: print('update environment')
        # Update environment vegetation.
        self.env.update()
        if config.DEBUG_MODE: print('grass num is:', len(self.env.grass_points))

        return child_ids, dead_ids

    def eat_food(self, creature: Creature, food_type: str):
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
            # candidate_indices = kd_tree.query_ball_point(eye_position, creature.vision_limit) TODO: use the kd_tree here too!
            food_distances = [np.linalg.norm(food_point - creature.position)
                              for food_point in food_points]

            if np.min(food_distances) <= config.FOOD_DISTANCE_THRESHOLD:
                # update creature energy
                creature.energy += creature.food_efficiency * food_energy

                # remove food from board
                closest_food_point = self.env.grass_points[np.argmin(food_distances)]
                self.env.grass_points.remove(closest_food_point)
                is_found_food = True
        if is_found_food:
            self.env.update_grass_kd_tree()  # TODO: add here the leaf kd_tree too
        return is_found_food

    def update_debug_logs(self, child_ids, dead_ids, frame):
        # update birthdays
        for child_id in child_ids:
            self.birthday[child_id] = frame
        # ------------------------- update debug parameters ------------------------- #
        current_num_creatures = len(self.creatures.keys())
        if current_num_creatures > config.MAX_NUM_CREATURES:
            print(f'{frame=}: Too many creatures, simulation is too slow.')
            self.abort_simulation = True
        if current_num_creatures > 0:
            # update total/new/dead number of creatures
            self.num_creatures_per_frame.append(current_num_creatures)
            self.num_new_creatures_per_frame.append(len(child_ids))
            self.num_dead_creatures_per_frame.append(len(dead_ids))

            # update energy statistics
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

            # print(f'{frame=}: ended with {current_num_creatures} creatures '
            #       f'(+{len(child_ids)}, -{len(dead_ids)}), '
            #       f'max energy = {round(self.max_creature_energy_per_frame[-1], 2)}.')
        else:
            print(f'{frame=}: all creatures are dead :(.')
            self.abort_simulation = True

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
        global quiv, scat, grass_scat, leaves_scat, agent_scat

        fig = plt.figure(figsize=(16, 8))
        # Define the grid layout with uneven ratios
        gs = gridspec.GridSpec(nrows=2, ncols=3,
                               width_ratios=[1, 2, 1], height_ratios=[2, 1])  # 2:1 ratio for both axes
        ax_ancestors = fig.add_subplot(gs[0, 0])  # ancestor tree?
        ax_env = fig.add_subplot(gs[0, 1])  # Large subplot (3/4 of figure)
        ax_brain = fig.add_subplot(gs[0, 2])  # Smaller subplot (1/4 width, full height)
        ax_pass = fig.add_subplot(gs[1, 0])  # placeholder
        ax_agent_info = fig.add_subplot(gs[1, 1])  # Smaller subplot (1/4 height, full width)
        ax_zoom = fig.add_subplot(gs[1, 2])  # Smallest subplot (1/4 x 1/4)
        fig.figsize = (16, 8)
        extent = self.env.get_extent()
        ax_env.set_xlim(extent[0], extent[1])
        ax_env.set_ylim(extent[2], extent[3])
        ax_env.set_title("Evolution Simulation")

        # -------------------------------- function for simulation progress -------------------------------- #

        print('starting simulation')

        # Display the environment map with origin='lower' to avoid vertical mirroring.
        ax_env.imshow(self.env.map_data, extent=extent, alpha=0.3, origin='lower')  # , aspect='auto')

        # Draw the water source.
        water_x, water_y, water_r = self.env.water_source
        water_circle = Circle((water_x, water_y), water_r, color='blue', alpha=0.3)
        ax_env.add_patch(water_circle)

        # Initial creature positions.
        positions = np.array([creature.position for creature in self.creatures.values()])
        colors = [creature.color for creature in self.creatures.values()]
        scat = ax_env.scatter(positions[:, 0], positions[:, 1], c=colors, s=config.FOOD_SIZE,
                              transform=ax_env.transData)

        # Create quiver arrows for creature headings.
        U, V = [], []
        for creature in self.creatures.values():
            if creature.speed > 0:
                U.append(creature.velocity[0])
                V.append(creature.velocity[1])
            else:
                U.append(0)
                V.append(0)
        quiv = ax_env.quiver(positions[:, 0], positions[:, 1], U, V,
                             color=colors, scale=150, width=0.005)  # 'black'

        # Scatter plots for vegetation.
        grass_scat = ax_env.scatter([], [], c='lightgreen', edgecolors='black', s=10)
        leaves_scat = ax_env.scatter([], [], c='darkgreen', edgecolors='black', s=10)
        agent_scat = ax_env.scatter([], [], s=20, facecolors='none', edgecolors='r')

        def init_func():
            return scat, quiv, grass_scat, leaves_scat

        # -------------------------------- function for simulation progress -------------------------------- #
        # Initialize the progress bar outside of the update function
        progress_bar = tqdm(total=config.NUM_FRAMES * config.UPDATE_ANIMATION_INTERVAL,
                            desc=f"Alive num: {len(self.creatures)}\n"
                                 f"Total children {self.children_num} \n"
                                 f"Total dead {len(self.dead_creatures)}"
                                 f"\nSimulation progress:")

        def update(frame):
            # Skip extra initial calls (because blit=True)
            global quiv, scat, grass_scat, leaves_scat, agent_scat
            if len(self.creatures) == 0 or self.abort_simulation:
                ax_env.set_title(f"Evolution Simulation ({frame=})")
                progress_bar.update(self.animation_update_interval)
                self.frame_counter += self.animation_update_interval
                return scat, quiv, grass_scat, leaves_scat, agent_scat

            # --------------------------- run frame --------------------------- #
            for _ in range(self.animation_update_interval):
                child_ids, dead_ids = self.step(dt=config.DT, noise_std=config.NOISE_STD)

                # update debug logs
                self.update_debug_logs(child_ids, dead_ids, frame)

                # Update the progress bar
                progress_bar.set_description(f"Alive: {len(self.creatures)} | "
                                             f"Children: {self.children_num} | "
                                             f"Dead: {len(self.dead_creatures)} | Progress:")
                progress_bar.update(1)  # or self.animation_update_interval outside the for loop

            # Purge every so often to clear static agents
            if frame % 50 == 0:
                self.purge = True

            for creature_id, creature in self.creatures.items():
                if creature_id not in self.creatures_energy_per_frame.keys():
                    self.creatures_energy_per_frame[creature_id] = np.zeros(frame).tolist()
                    self.creatures_energy_per_frame[creature_id].append(creature.energy)
                else:
                    self.creatures_energy_per_frame[creature_id].append(creature.energy)

            # # Track animation update frames
            # if self.frame_counter % self.animation_update_interval != 0:
            #     # **Return the previous animation objects even if skipping updates**
            #     return scat, quiv, grass_scat, leaves_scat, agent_scat

            # --------------------------- plot --------------------------- #
            # clear quiver and scatter
            if 'quiv' in globals():
                quiv.remove()
            if 'scat' in globals():
                scat.remove()
            if 'grass_scat' in globals():
                try:  # in case it's empty
                    grass_scat.remove()
                except:
                    pass
            if 'leaves_scat' in globals():
                try:  # in case it's empty
                    leaves_scat.remove()
                except:
                    pass
            if 'agent_scat' in globals():
                agent_scat.remove()

            # Update creature positions.
            positions = np.array([creature.position for creature in self.creatures.values()])
            colors = [creature.color for creature in self.creatures.values()]
            if len(positions) > 0:
                scat = ax_env.scatter(positions[:, 0], positions[:, 1], c=colors, s=config.FOOD_SIZE)
                # scat.set_offsets(positions)

                U, V = [], []
                for creature in self.creatures.values():
                    if creature.speed > 0:
                        U.append(creature.velocity[0])
                        V.append(creature.velocity[1])
                    else:
                        U.append(0)
                        V.append(0)

                quiv = ax_env.quiver(positions[:, 0], positions[:, 1], U, V,
                                     color=colors, scale=150, width=0.005)
                # quiv.set_offsets(positions)
                # quiv.set_UVC(U, V)
            else:
                scat = ax_env.scatter([1], [1])
                quiv = ax_env.quiver([1], [1], [1], [1])

            # Update vegetation scatter data.
            if len(self.env.grass_points) > 0:
                grass_points = np.array(self.env.grass_points)
                grass_scat = ax_env.scatter(grass_points[:, 0], grass_points[:, 1], c='lightgreen', edgecolors='black',
                                            s=10)
                # grass_scat.set_offsets(np.array(self.env.grass_points))
            if len(self.env.leaf_points) > 0:
                leaf_points = np.array(self.env.leaf_points)
                leaves_scat = ax_env.scatter(leaf_points[:, 0], leaf_points[:, 1], c='darkgreen', edgecolors='black',
                                             s=20)
                # leaves_scat.set_offsets(np.array(self.env.leaf_points))

            ax_env.set_title(f"Evolution Simulation ({frame=})")
            # --------------------- focus on one agent ----------------------------
            if len(self.creatures) > 0:
                ids = [creature.creature_id for creature in self.creatures.values()]
                if self.focus_ID not in ids:
                    if self.id_count in ids:
                        self.focus_ID = self.id_count
                    else:
                        self.focus_ID = np.random.choice(list(self.creatures.keys()))
                agent = self.creatures[self.focus_ID]
                agent_scat = ax_env.scatter(
                    [agent.position[0]] * 3, [agent.position[1]] * 3,  # Repeat position for multiple rings
                    s=[50, 20, 500],  # Different sizes for bullseye rings # config.FOOD_SIZE
                    facecolors=['black', 'red', 'black'], edgecolors=['black', 'red', 'yellow'],
                    # Different colors for bullseye rings
                    linewidth=5, alpha=[0.9, 1, 0.5], marker='*'
                )
                agent.brain.plot(ax_brain)
                # ax_agent_info.clear()
                agent.plot_live_status(ax_agent_info)
                agent.plot_acc_status(ax_zoom, plot_type=1, curr_frame=self.frame_counter)
                # Create zoomed-in inset
                # axins = zoomed_inset_axes(ax_env, zoom=100, loc="upper right")  # zoom=2 means 2x zoom
                # axins = inset_axes(ax_env, width="30%", height="30%", loc="upper right")
                # axins.set_xlim(agent.position[0] - 100, agent.position[0] + 100)  # Set zoom-in limits
                # axins.set_ylim(agent.position[1] - 100, agent.position[1] + 100)  # Adjust zoom region
                # axins.set_xticks([])  # Hide x-axis ticks
                # axins.set_yticks([])  # Hide y-axis ticks
            return scat, quiv, grass_scat, leaves_scat, agent_scat

        # ----------------------------------- run simulation and save animation ------------------------------------ #
        ani = animation.FuncAnimation(fig, update, frames=config.NUM_FRAMES, interval=config.FRAME_INTERVAL,
                                      init_func=init_func, blit=True)
        ani.save(config.ANIMATION_FILEPATH, writer="ffmpeg", dpi=100)
        plt.close(fig)
        print('finished simulation.')
        print(f'Simulation animation saved as {config.ANIMATION_FILEPATH.stem}.')

        # ----------------------------------- Plot graphs after simulation ended ----------------------------------- #
        # all_creatures = {**self.creatures, **self.dead_creatures}
        plot_lineage_tree(self.creatures)

        # specific fig
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

        # statistics fig
        fig, ax_env = plt.subplots(2, 1, sharex='all')
        ax2 = ax_env[0].twinx()
        ax2.plot(self.num_creatures_per_frame, 'b.-', label='total')
        ax_env[0].plot(self.num_new_creatures_per_frame, 'g.-', label='new')
        ax_env[0].plot(self.num_dead_creatures_per_frame, 'r.-', label='dead')
        ax_env[0].set_title('num creatures per frame')
        ax_env[0].legend()

        ax_env[1].plot(self.min_creature_energy_per_frame, '.-', label='min energy')
        ax_env[1].plot(self.max_creature_energy_per_frame, '.-', label='max energy')
        ax_env[1].errorbar(x=np.arange(len(self.mean_creature_energy_per_frame)),
                           y=self.mean_creature_energy_per_frame,
                           yerr=self.std_creature_energy_per_frame, linestyle='-', marker='.',
                           label='mean and std energy')
        ax_env[1].axhline(y=list(self.dead_creatures.values())[0].reproduction_energy + config.MIN_LIFE_ENERGY,
                          linestyle='--', color='r', label='reproduction threshold')
        ax_env[1].set_title('energy statistics per frame')
        ax_env[1].set_xlabel('frame number')
        ax_env[1].legend()

        fig.savefig(fname=config.STATISTICS_FIG_FILEPATH)
        print(f'statistics fig saved as {config.STATISTICS_FIG_FILEPATH.stem}.')

        plt.show()
