import numpy as np
from scipy.spatial import KDTree

# from brain_models.fully_connected_brain import Brain
from creature import Creature
from environment import Environment
from tqdm import tqdm
from config import Config as config

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes, mark_inset

import importlib
brain_module = importlib.import_module(f"brain_models.{config.BRAIN_TYPE}")
Brain = getattr(brain_module, 'Brain')

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
                                                   simulation_space=self.env.size,
                                                   input_size=config.INPUT_SIZE,
                                                   output_size=config.OUTPUT_SIZE,
                                                   eyes_params=config.EYES_PARAMS,
                                                   env=self.env)
        self.dead_creatures = dict()
        self.positions = []

        # Build a KDTree for creature positions.
        self.creatures_kd_tree = self.build_creatures_kd_tree()
        self.children_num = 0

        # simulation log parameters
        self.num_creatures_per_step = []
        self.num_new_creatures_per_step = []
        self.num_dead_creatures_per_step = []
        self.creatures_history = []
        self.num_grass_history = []
        self.num_leaves_history = []
        self.log_num_eats = []

        # statistics log parameters
        self.stat_dict = {'min': np.min, 'max': np.max, 'mean': np.mean, 'std': np.std}
        self.stat_attributes = ['energy', 'speed']
        for stat_name in self.stat_dict.keys():
            for stat_attribute in self.stat_attributes:
                setattr(self, f'{stat_name}_creature_{stat_attribute}_per_step', [])

        # simulation control parameters
        self.abort_simulation = False
        self.kdtree_update_interval = config.UPDATE_KDTREE_INTERVAL  # Set update interval for KDTree
        self.animation_update_interval = config.UPDATE_ANIMATION_INTERVAL  # Set update interval for animation frames
        self.step_counter = 0  # Initialize step counter
        self.id_count = config.NUM_CREATURES - 1
        self.focus_ID = 0
        self.purge = True  # flag for purge events

        if config.DEBUG_MODE:
            np.seterr(all='raise')  # Convert NumPy warnings into exceptions

    @staticmethod
    def initialize_creatures(num_creatures, simulation_space, input_size, output_size,
                             eyes_params, env: Environment):
        """
        Initializes creatures ensuring they are not placed in a forbidden (black) area.
        """
        creatures = dict()

        for creature_id in range(num_creatures):
            # get a valid position.
            position = []
            valid_position = False
            while not valid_position:
                position = np.random.rand(2) * simulation_space
                # Convert (x, y) to indices (col, row)
                col, row = int(position[0]), int(position[1])
                # height, width = env.map_data.shape[:2]
                # Check bounds and obstacle mask.
                if col < 0 or col >= env.width or row < 0 or row >= env.height:
                    continue
                if env.obstacle_mask[row, col]:
                    continue
                valid_position = True

            # static traits
            gen = 0
            parent_id = None
            birth_step = 0
            max_age = np.random.randint(low=config.INIT_MAX_AGE * 0.8, high=config.INIT_MAX_AGE)
            color = np.random.rand(3)  # Random RGB color.

            max_mass = np.random.uniform(low=0.8, high=1) * config.INIT_MAX_MASS
            max_height = np.random.uniform(low=0.8, high=1) * config.INIT_MAX_HEIGHT
            max_strength = np.random.uniform(low=0.8, high=1) * config.INIT_MAX_STRENGTH

            max_speed = np.random.uniform(low=0.8, high=1) * config.MAX_SPEED
            max_energy = np.random.uniform(low=0.8, high=1) * config.INIT_MAX_ENERGY

            digest_dict = config.INIT_DIGEST_DICT
            reproduction_energy = config.REPRODUCTION_ENERGY

            vision_limit = config.VISION_LIMIT
            brain = Brain([input_size, output_size])

            # init creature
            creature = Creature(
                creature_id=creature_id, gen=gen, parent_id=parent_id, birth_step=birth_step, color=color,
                max_age=max_age, max_mass=max_mass, max_height=max_height,
                max_strength=max_strength, max_speed=max_speed, max_energy=max_energy,
                digest_dict=digest_dict, reproduction_energy=reproduction_energy,
                eyes_params=eyes_params, vision_limit=vision_limit, brain=brain,
                position=position)

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
                    result = self.detect_target_from_kdtree(creature, eye_params, kd_tree, candidate_points, noise_std)
                else:
                    result = None

                channel_name = f'{channel}_{i_eye}'
                channel_results[channel_name] = result
                channels_list.append(channel_name)

        return channel_results

    def use_brain(self, creature: Creature, dt: float, noise_std: float = 0.0):
        try:
            # get brain input
            seek_results = self.seek(creature=creature, noise_std=noise_std)
            eyes_inputs = [self.prepare_eye_input(seek_results, creature.vision_limit) for seek_results in
                           seek_results.values()]
            brain_input = []
            brain_input.append(np.array([creature.hunger, creature.thirst]))
            brain_input.append(creature.speed)
            brain_input.append(np.concatenate(eyes_inputs))
            brain_input = np.hstack(brain_input)
            decision = creature.think(brain_input)
            creature.move(decision=decision, dt=dt)
        except Exception as e:
            print(f'Error in Simulation (use_brain, movement) for creature: {creature.creature_id}:\n{e}')
            breakpoint()

        # Collision detection: handle cases where creature's new position is inside an obstacle or outbound.
        try:
            col, row = map(int, creature.position)  # Convert (x, y) to image indices (col, row)
            height, width = self.env.map_data.shape[:2]
            if col < 0 or col >= width or row < 0 or row >= height or self.env.obstacle_mask[row, col]:
                # choose if the velocity is set to zero or get mirrored
                if config.BOUNDARY_CONDITION == 'zero':
                    creature.velocity = np.array([0.0, 0.0])
                elif config.BOUNDARY_CONDITION == 'mirror':
                    creature.velocity = -creature.velocity
        except Exception as e:
            print(f'exceptiom in use_brain for creature: {creature.creature_id}\n{e}')
            print(f'Error in Simulation (use_brain, collision detection) for creature: {creature.creature_id}:\n{e}')
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
    def detect_target_from_kdtree(creature: Creature, eye, kd_tree: KDTree, candidate_points: np.ndarray,
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
        angle_offset, aperture = eye
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

    def kill(self, creature_id):
        self.dead_creatures[creature_id] = self.creatures[creature_id]
        del self.creatures[creature_id]

    def do_step(self, dt: float, noise_std: float = 0.0):
        """
        Advances the simulation by one time step.
        For each creature:
          - Perceives its surroundings with both eyes.
          - Constructs an input vector for the brain.
          - Receives a decision (delta_angle, delta_speed) to update its velocity.
          - Checks for collisions with obstacles (black areas) and stops if necessary.
        Then, moves creatures and updates the vegetation.
        """

        # -------------------- Use brain and update creature velocities --------------------------

        for creature_id, creature in self.creatures.items():
            self.use_brain(creature=creature, dt=dt, noise_std=noise_std)

        # ----------------- die / eat / reproduce (+create list of creatures to die or reproduce) ----------------------

        list_creatures_reproduce = []
        list_creature_die = []

        for creature_id, creature in self.creatures.items():
            # death from age or fatigue
            if creature.age >= creature.max_age or creature.energy <= 0:
                list_creature_die.append(creature_id)
                continue
            else:
                creature.age += config.DT

                # check for food (first grass, if not found search for leaf if tall enough)
                is_found_food = self.eat_food(creature=creature, food_type='grass')
                if not is_found_food and creature.height >= config.LEAF_HEIGHT:
                    _ = self.eat_food(creature=creature, food_type='leaf')

                # reproduce if able
                if creature.energy > creature.reproduction_energy + config.MIN_LIFE_ENERGY:
                    list_creatures_reproduce.append(creature_id)

        # ------------------------ add the purge to the killing list ----------------------------

        if config.DO_PURGE:
            if self.purge or len(self.creatures) > config.MAX_NUM_CREATURES * config.STUCK_PERCENTAGE:
                purge_count = 0
                self.purge = False
                for creature_id, creature in self.creatures.items():
                    if (len(self.creatures) > config.MAX_NUM_CREATURES * 0.95 and np.random.rand(1) < 0.1
                            and creature_id not in list_creature_die):
                        purge_count += 1
                        list_creature_die.append(creature_id)
                    if creature.max_speed_exp <= config.PURGE_SPEED_THRESHOLD and creature_id not in list_creature_die:
                        purge_count += 1
                        list_creature_die.append(creature_id)
                print(f'\nStep {self.step_counter}: Purging {purge_count} creatures.')

        # ------------------------ use the list to kill ----------------------------

        # kill creatures
        dead_ids = []
        for creature_id in list_creature_die:
            self.kill(creature_id)
            dead_ids.append(creature_id)

        # ------------------------ use the list to reproduce ----------------------------

        # Reproduction
        child_ids = []
        for creature_id in list_creatures_reproduce:
            # update creature
            creature = self.creatures[creature_id]
            child = creature.reproduce()
            creature.log_reproduce.append(self.step_counter)

            # update child
            self.id_count += 1
            child.creature_id = self.id_count
            child.birth_step = self.step_counter

            # add to simulation
            self.creatures[self.id_count] = child
            child_ids.append(self.id_count)
            self.children_num += 1

        # ------------------------------- Update creatures log -------------------------------
        for creature in self.creatures.values():
            creature.log_energy.append(creature.energy)
            creature.log_speed.append(creature.speed)

        # ------------------------ Update KDtree (in some frames) ----------------------------

        # Update environment vegetation (generate new points if conditions are met)
        self.env.update()

        # Update KDTree every "kdtree_update_interval" frames
        if self.step_counter % self.kdtree_update_interval == 0:
            self.update_creatures_kd_tree()
            self.env.update_grass_kd_tree()

        self.step_counter += 1

        return child_ids, dead_ids

    def eat_food(self, creature: Creature, food_type: str):
        # check if creature is full
        if creature.energy >= creature.max_energy:
            return False

        # get food points
        is_found_food = False
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

            closest_food_distance = np.min(food_distances)
            closest_food_point = self.env.grass_points[np.argmin(food_distances)]

            if closest_food_distance <= config.FOOD_DISTANCE_THRESHOLD:
                # creature eat food
                creature.eat(food_type=food_type, food_energy=food_energy)
                creature.log_eat.append(self.step_counter)

                # remove food from environment
                self.env.grass_points.remove(closest_food_point)
                self.env.update_grass_kd_tree()
                is_found_food = True

        return is_found_food

    def update_statistics_logs(self, child_ids, dead_ids, step):
        try:
            # abort simulation if there are too many creatures or no creatures left
            current_num_creatures = len(self.creatures)
            if current_num_creatures > config.MAX_NUM_CREATURES:
                print(f'{step=}: Too many creatures, simulation is too slow.')
                self.abort_simulation = True
            elif current_num_creatures <= 0:
                if not self.abort_simulation:
                    print(f'\n{step=}: all creatures are dead :(.')
                    self.abort_simulation = True
            else:
                # update number of alive/new/dead creatures (and also id of all alive)
                self.num_creatures_per_step.append(current_num_creatures)
                self.num_new_creatures_per_step.append(len(child_ids))
                self.num_dead_creatures_per_step.append(len(dead_ids))
                self.creatures_history.append(
                    [getattr(creature, 'creature_id') for creature in self.creatures.values()])

                # update environment log
                self.num_grass_history.append(len(self.env.grass_points))
                self.num_leaves_history.append(len(self.env.leaf_points))

                # update energy and speed statistics
                self.update_attribute_statistics_logs(attribute='energy')
                self.update_attribute_statistics_logs(attribute='speed')

                # Update eating logs
                self.log_num_eats.append(np.sum([len(creature.log_eat) for creature in self.creatures.values()]))

        except Exception as e:
            print(f'Error in Simulation (update_statistics_logs):\n{e}')
            breakpoint()

    def update_attribute_statistics_logs(self, attribute: str):
        creatures_attribute = [getattr(creature, attribute) for creature in self.creatures.values()]

        for stat_name, stat_func in self.stat_dict.items():
            getattr(self, f'{stat_name}_creature_{attribute}_per_step').append(stat_func(creatures_attribute))

    def run_and_visualize(self):
        """
        Runs the simulation for a given number of frames and saves an animation.
        Visualizes:
          - The environment map with semi-transparent overlay (using origin='lower').
          - The water source, vegetation (grass and leaves) with outlines.
          - Creatures as colored dots with arrows indicating heading.
        Prints progress every 10 frames.
        """
        global quiv, scat, grass_scat, leaves_scat, agent_scat
        global fig, ax_env, ax_brain, ax_agent_info, ax_zoom, progress_bar

        def init_fig():
            """
            Init the simulation figure.
            :return:
            """
            global quiv, scat, grass_scat, leaves_scat, agent_scat
            global fig, ax_env, ax_brain, ax_agent_info, ax_zoom, progress_bar

            # init fig with the grid layout with uneven ratios
            fig = plt.figure(figsize=(16, 8))
            fig_grid = gridspec.GridSpec(2, 3, width_ratios=[1, 2, 1], height_ratios=[2, 1])  # 2:1 ratio for both axes
            ax_lineage = fig.add_subplot(fig_grid[0, 0])  # ancestor tree?
            ax_env = fig.add_subplot(fig_grid[0, 1])  # Large subplot (3/4 of figure)
            ax_brain = fig.add_subplot(fig_grid[0, 2])  # Smaller subplot (1/4 width, full height)
            ax_pass = fig.add_subplot(fig_grid[1, 0])  # placeholder
            ax_agent_info = fig.add_subplot(fig_grid[1, 1])  # Smaller subplot (1/4 height, full width)
            ax_zoom = fig.add_subplot(fig_grid[1, 2])  # Smallest subplot (1/4 x 1/4)
            extent = self.env.get_extent()
            ax_env.set_xlim(extent[0], extent[1])
            ax_env.set_ylim(extent[2], extent[3])
            ax_env.set_title("Evolution Simulation")

            # Display the environment map with origin='lower' to avoid vertical mirroring
            ax_env.imshow(self.env.map_data, extent=extent, alpha=0.3, origin='lower')  # , aspect='auto')

            # Draw the water source
            water_x, water_y, water_r = self.env.water_source
            water_circle = Circle((water_x, water_y), water_r, color='blue', alpha=0.3)
            ax_env.add_patch(water_circle)

            # Initial creature positions
            self.positions = np.array([creature.position for creature in self.creatures.values()])
            colors = [creature.color for creature in self.creatures.values()]
            sizes = np.array([creature.mass for creature in self.creatures.values()]) * config.FOOD_SIZE / 100
            scat = ax_env.scatter(self.positions[:, 0], self.positions[:, 1], c=colors, s=sizes,
                                  transform=ax_env.transData)

            # Create quiver arrows for creature headings
            U, V = [], []
            for creature in self.creatures.values():
                if creature.speed > 0:
                    U.append(creature.velocity[0])
                    V.append(creature.velocity[1])
                else:
                    U.append(0)
                    V.append(0)
            quiv = ax_env.quiver(self.positions[:, 0], self.positions[:, 1], U, V,
                                 color=colors, scale=150, width=0.005)  # 'black'

            # Scatter food points for vegetation
            grass_scat = ax_env.scatter([], [], c='lightgreen', edgecolors='black', s=10)
            leaves_scat = ax_env.scatter([], [], c='darkgreen', edgecolors='black', s=10)
            agent_scat = ax_env.scatter([], [], s=20, facecolors='none', edgecolors='r')

            # Initialize the progress bar to print
            if config.STATUS_EVERY_STEP:
                update_num = config.NUM_FRAMES * config.UPDATE_ANIMATION_INTERVAL
            else:
                update_num = config.NUM_FRAMES

            progress_bar = tqdm(total=update_num, desc=f"Alive: {len(self.creatures)} | "
                                             f"Children: {self.children_num} | "
                                             f"Dead: {len(self.dead_creatures)} | "
                                             f"leaves: {len(self.env.leaf_points)} | "
                                             f"grass: {len(self.env.grass_points)} | "
                                             f"Progress")

        def init_func():
            """
            Function for simulation initialization.
            A way to make sure animation doesn't call update multiple times for initialization.
            Once we will learn how to update scat/quiv instead of redrawing them it can also
            reduce computation time (because blit=True)
            :return:
            """
            global quiv, scat, grass_scat, leaves_scat, agent_scat
            global fig, ax_env, ax_brain, ax_agent_info, ax_zoom, progress_bar

            return scat, quiv, grass_scat, leaves_scat, agent_scat

        # Function for simulation progress
        def update_func(frame):
            """
            The main function of the animation.
            This function runs a single frame of the animation.
            Each frame contain multiple simulation steps according to config.
            :param frame:
            :return: the variables that are updated (right now we are redrawing them)
            """
            global quiv, scat, grass_scat, leaves_scat, agent_scat
            global fig, ax_env, ax_brain, ax_agent_info, ax_zoom, progress_bar

            # abort simulation if no creatures left or there are too many creatures
            if self.abort_simulation:
                from matplotlib import use

                use('TkAgg')
                for stat_attribute in self.stat_attributes:
                    statistics_fig = self.plot_statistics_graph(stat_attribute=stat_attribute)
                    statistics_fig.show()

                breakpoint()

                ax_env.set_title(f"Evolution Simulation ({frame=}, step={self.step_counter})")
                progress_bar.update(self.animation_update_interval)
                self.step_counter += self.animation_update_interval

                return scat, quiv, grass_scat, leaves_scat, agent_scat

            # Run steps of frame
            for step in range(self.animation_update_interval):
                # Do simulation step
                child_ids, dead_ids = self.do_step(dt=config.DT, noise_std=config.NOISE_STD)

                # Update statistics logs
                self.update_statistics_logs(child_ids=child_ids, dead_ids=dead_ids, step=self.step_counter)

                # Update the progress bar every step
                if config.STATUS_EVERY_STEP:
                    progress_bar.set_description(f"Alive: {len(self.creatures)} | "
                                             f"Children: {self.children_num} | "
                                             f"Dead: {len(self.dead_creatures)} | "
                                             f"leaves: {len(self.env.leaf_points)} | "
                                             f"grass: {len(self.env.grass_points)} | "
                                             f"Progress")
                    progress_bar.update(1)  # or self.animation_update_interval outside the for loop

            # update the progress bar every frame
            if not config.STATUS_EVERY_STEP:
                progress_bar.set_description(f"Alive: {len(self.creatures)} | "
                                             f"Children: {self.children_num} | "
                                             f"Dead: {len(self.dead_creatures)} | "
                                             f"leaves: {len(self.env.leaf_points)} | "
                                             f"grass: {len(self.env.grass_points)} | "
                                             f"Progress")
                progress_bar.update(1)  # or self.animation_update_interval outside the for loop

            # Do purge if PURGE_FRAME_FREQUENCY frames passed (to clear static agents)
            if config.DO_PURGE:
                if frame % config.PURGE_FRAME_FREQUENCY == 0:
                    if len(self.creatures) > config.MAX_NUM_CREATURES * config.PURGE_PERCENTAGE:
                        self.purge = True

            if config.DEBUG_MODE:
                from matplotlib import use

                use('TkAgg')
                for stat_attribute in self.stat_attributes:
                    statistics_fig = self.plot_statistics_graph(stat_attribute=stat_attribute)
                    statistics_fig.show()

                breakpoint()
            # --------------------------- Plot --------------------------- #

            try:
                # Update creature positions and directions
                num_creatures_in_last_frame = len(self.positions)
                self.positions = np.array([creature.position for creature in self.creatures.values()])
                sizes = np.array([creature.mass for creature in self.creatures.values()]) * config.FOOD_SIZE / 100
                colors = [creature.color for creature in self.creatures.values()]

                U, V = [], []
                for creature in self.creatures.values():
                    if creature.speed > 0:
                        U.append(creature.velocity[0])
                        V.append(creature.velocity[1])
                    else:
                        U.append(0)
                        V.append(0)

                # Update or redraw scat and quiver plots
                num_creatures_after_step = len(self.creatures)
                if num_creatures_after_step > 0:
                    if num_creatures_after_step == num_creatures_in_last_frame:
                        # Update scatter and quiver plot (positions & directions)
                        scat.set_offsets(self.positions)
                        quiv.set_offsets(self.positions)
                        quiv.set_UVC(U, V)  # Update U (x-component) and V (y-component)
                    else:
                        # Clear scatter and quiver plots (positions & directions)
                        for obj_name in ['quiv', 'scat', 'grass_scat', 'leaves_scat', 'agent_scat']:
                            obj = globals()[obj_name]
                            try:  # in case it's empty
                                obj.remove()
                            except:
                                pass

                        # Redraw scatter and quiver plots (positions & directions)
                        scat = ax_env.scatter(self.positions[:, 0], self.positions[:, 1],
                                              c=colors, s=sizes)
                        quiv = ax_env.quiver(self.positions[:, 0], self.positions[:, 1], U, V,
                                             color=colors, scale=150, width=0.005)
                else:
                    # plot place holder
                    scat = ax_env.scatter([1], [1])
                    quiv = ax_env.quiver([1], [1], [1], [1])

                # Update vegetation scatter data
                # num_grass_points_in_last_frame = TODO
                num_grass_points_after_step = len(self.env.grass_points)
                if num_grass_points_after_step > 0:
                    # if num_grass_points_after_step == num_grass_points_in_last_frame:
                    #     grass_scat.set_offsets(np.array(self.env.grass_points))
                    # else:
                    grass_points = np.array(self.env.grass_points)
                    grass_scat = ax_env.scatter(grass_points[:, 0], grass_points[:, 1], c='lightgreen', edgecolors='black',
                                                s=10)

                # num_leaf_points_in_last_frame = TODO
                num_leaf_points_after_step = len(self.env.leaf_points)
                if num_leaf_points_after_step > 0:
                    # if num_leaf_points_after_step == num_grass_points_in_last_frame:
                    #     leaves_scat.set_offsets(np.array(self.env.leaf_points))
                    # else:
                    leaf_points = np.array(self.env.leaf_points)
                    leaves_scat = ax_env.scatter(leaf_points[:, 0], leaf_points[:, 1], c='darkgreen', edgecolors='black',
                                                 s=20)

                ax_env.set_title(f"Evolution Simulation ({frame=}, step={self.step_counter})")
                # --------------------- focus on one agent ----------------------------

                if len(self.creatures) > 0:
                    ids = self.creatures_history[-1]
                    if self.focus_ID not in ids:
                        if self.id_count in ids:
                            self.focus_ID = self.id_count
                        else:
                            self.focus_ID = np.random.choice(list(self.creatures.keys()))
                    agent = self.creatures[self.focus_ID]
                    agent_scat = ax_env.scatter(
                        [agent.position[0]] * 2, [agent.position[1]] * 2,  # Repeat position for N=2 rings
                        s=[60, 500],  # Different sizes for bullseye rings # config.FOOD_SIZE
                        facecolors=['none', 'none'],
                        edgecolors=['black', 'black'],
                        linewidth=2.5,
                        marker='o'  # or 'x'
                    )
                    agent.brain.plot(ax_brain)
                    # ax_agent_info.clear()
                    agent.plot_live_status(ax_agent_info)
                    agent.plot_acc_status(ax_zoom, plot_type=1, curr_step=self.step_counter)
                    # Create zoomed-in inset
                    # axins = zoomed_inset_axes(ax_env, zoom=100, loc="upper right")  # zoom=2 means 2x zoom
                    # axins = inset_axes(ax_env, width="30%", height="30%", loc="upper right")
                    # axins.set_xlim(agent.position[0] - 100, agent.position[0] + 100)  # Set zoom-in limits
                    # axins.set_ylim(agent.position[1] - 100, agent.position[1] + 100)  # Adjust zoom region
                    # axins.set_xticks([])  # Hide x-axis ticks
                    # axins.set_yticks([])  # Hide y-axis ticks

            except Exception as e:
                print(f'Error in simulation (update_func): cannot plot because {e}.')
                breakpoint()

            return scat, quiv, grass_scat, leaves_scat, agent_scat

        # Run simulation
        try:
            init_fig()
            self.update_statistics_logs(child_ids=[], dead_ids=[], step=-1)
            ani = animation.FuncAnimation(fig=fig, func=update_func, init_func=init_func, blit=True,
                                      frames=config.NUM_FRAMES, interval=config.FRAME_INTERVAL)
            print('\nSimulation completed successfully. saving progress...')

        except KeyboardInterrupt:
            print('Simulation interrupted. saving progress...')

        finally:
            # Save animation
            ani.save(config.ANIMATION_FILEPATH, writer="ffmpeg", dpi=100)
            plt.close(fig)
            print(f'Simulation animation saved as {config.ANIMATION_FILEPATH.stem}.')

            # Plot statistics summary graph
            from matplotlib import use
            use('TkAgg')
            for stat_attribute in self.stat_attributes:
                statistics_fig = self.plot_statistics_graph(stat_attribute=stat_attribute)
                statistics_fig.show()
                statistics_fig_filepath = config.OUTPUT_FOLDER.joinpath(
                    config.STATISTICS_FIG_FILEPATH.stem + f'_{stat_attribute}.png')
                statistics_fig.savefig(fname=statistics_fig_filepath)
                print(f'statistics fig saved as {statistics_fig_filepath.stem}.')

            env_fig, ax = plt.subplots(1, 1)
            ax[0].plot(self.num_grass_history, 'g.-', label='num grass')
            ax[0].plot(self.num_leaves_history, 'k.-', label='num leaves')
            ax[0].legend()
            env_fig.show()
            env_fig.savefig(fname=config.ENV_FIG_FILE_PATH)

    def plot_statistics_graph(self, stat_attribute: str):
        """
        Plot a graph showing the number of current/new/dead creatures in every step and
        a graph showing the min/mstatistics_ax/mean/std energy of all creatures alive in every step.
        :return:
        """

        # Plot number of alive/new/dead creatures in every step
        num_creatures_per_step = np.array(self.num_creatures_per_step)
        num_new_creatures_per_step = np.array(self.num_new_creatures_per_step)
        num_dead_creatures_per_step = np.array(self.num_dead_creatures_per_step)

        statistics_fig, statistics_ax = plt.subplots(2, 1, sharex='all')
        statistics_ax[0].plot(num_creatures_per_step, 'b.-', label='alive')
        statistics_ax[0].axhline(y=config.MAX_NUM_CREATURES,
                                 linestyle='--', color='b', label='max num creatures')
        statistics_ax[0].tick_params(axis='y', colors='b')
        statistics_ax[0].set_title('num creatures per step')
        statistics_ax[0].legend()

        # plot number of alive creatures in second y-axis for clarity
        statistics_ax2 = statistics_ax[0].twinx()
        statistics_ax2.plot(num_new_creatures_per_step, 'g.-', label='new')
        statistics_ax2.plot(num_dead_creatures_per_step, 'r.-', label='dead')
        statistics_ax2.legend()

        # plot min/max/mean/std attribute (speed or energy) of all creatures alive in every step
        statistics_ax[1].plot(getattr(self, f'min_creature_{stat_attribute}_per_step'), '.-',
                              label=f'min {stat_attribute}')
        statistics_ax[1].plot(getattr(self, f'max_creature_{stat_attribute}_per_step'), '.-',
                              label=f'max {stat_attribute}')
        statistics_ax[1].errorbar(x=np.arange(len(getattr(self, f'mean_creature_{stat_attribute}_per_step'))),
                                  y=getattr(self, f'mean_creature_{stat_attribute}_per_step'),
                                  yerr=getattr(self, f'std_creature_{stat_attribute}_per_step'), linestyle='-',
                                  marker='.',
                                  label=f'mean and std {stat_attribute}')
        if stat_attribute == 'energy':
            statistics_ax[1].axhline(y=config.REPRODUCTION_ENERGY + config.MIN_LIFE_ENERGY,
                                     linestyle='--', color='r', label='reproduction threshold')
        statistics_ax[1].set_title(f'{stat_attribute} statistics per step')
        statistics_ax[1].set_xlabel('step number')
        statistics_ax[1].legend()

        return statistics_fig


# for debug (run by selecting all line and press Alt+Shift+E)
if False:
    from matplotlib import use

    use('TkAgg')
    for stat_attribute in self.stat_attributes:
        statistics_fig = self.plot_statistics_graph(stat_attribute=stat_attribute)
        statistics_fig.show()
