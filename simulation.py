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

from statistics_logs import StatisticsLogs
from traits_evolution.trait_stacked_colored_histogram import trait_stacked_colored_histogram
from traits_evolution.traits_scatter import plot_traits_scatter

brain_module = importlib.import_module(f"brain_models.{config.BRAIN_TYPE}")
Brain = getattr(brain_module, 'Brain')

global lineage_graph, traits_scat, quiv, scat, grass_scat, leaves_scat, agent_scat
global fig, ax_lineage, ax_traits, ax_env, ax_brain, ax_agent_info_1, ax_agent_info_2, ax_agent_events, ax_life, progress_bar


class Simulation:
    """
    Manages the simulation of creatures within an environment.
    Handles perception, decision-making, movement, collision detection, and vegetation updates.
    Implements multichannel perception by using separate KDTree queries for each target type.
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

        # simulation control parameters
        self.abort_simulation = False
        self.kdtree_update_interval = config.UPDATE_KDTREE_INTERVAL  # Set update interval for KDTree
        self.animation_update_interval = config.UPDATE_ANIMATION_INTERVAL  # Set update interval for animation frames
        self.step_counter = 0  # Initialize step counter
        self.id_count = config.NUM_CREATURES - 1
        self.focus_id = 0
        self.make_agent(focus_id=0)

        self.purge = True  # flag for purge events

        # statistics logs
        self.statistics_logs = StatisticsLogs()

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

    def make_agent(self, focus_id: int = 0):
        self.focus_id = focus_id
        self.creatures[self.focus_id].make_agent()

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
        kd_tree = []
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

    def use_brain(self, creature: Creature, dt: float, seek_results: dict, noise_std: float = 0.0):
        try:
            # get brain input
            # seek_results = self.seek(creature=creature, noise_std=noise_std)  # REMOVE
            eyes_inputs = [self.prepare_eye_input(seek_result, creature.vision_limit) for seek_result in
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
            # breakpoint()

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
            print(f'exception in use_brain for creature: {creature.creature_id}\n{e}')
            print(f'Error in Simulation (use_brain, collision detection) for creature: {creature.creature_id}:\n{e}')
            # breakpoint()

    @staticmethod
    def prepare_eye_input(detection_result, vision_limit):
        """
        Converts a detection result (distance, signed_angle) or None into a 3-element vector:
          [detection_flag, distance, angle].
        """
        if detection_result is None:
            return np.array([0, vision_limit, 0])
        else:
            distance, angle = detection_result[0:2]
            return np.array([1, distance, angle])

    @staticmethod
    def detect_target_from_kdtree(creature: Creature, eye_params,
                                  kd_tree: KDTree, candidate_points: np.ndarray,
                                  noise_std: float = 0.0):
        """
        Generic function to detect the closest target from candidate_points using a KDTree.

        Parameters:
          creature: the creature performing the detection.
          eye_params: (angle_offset,aperture) specifying the eye's viewing direction relative to the creature's heading.
          kd_tree: a KDTree built from candidate_points.
          candidate_points: numpy array of shape (N, 2) containing candidate target positions.
          noise_std: standard deviation for optional Gaussian noise.

        Returns:
          A tuple (distance, signed_angle) for the detected target, or None if no target qualifies.
        """
        eye_position = creature.position
        heading = creature.get_heading()
        angle_offset, aperture = eye_params
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
        # Evaluate each candidate - which was sorted by the KDtree.
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
                detected_info = (distance, angle, idx)  # TODO: make sure to save the index too for fast removal
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

        # ------------------ seek creatures' targets ------------------------------
        seek_results = {}
        for creature_id, creature in self.creatures.items():
            seek_results[creature_id] = self.seek(creature=creature, noise_std=noise_std)
        # -------------------- Use brain and update creature velocities --------------------------

        for creature_id, creature in self.creatures.items():
            self.use_brain(creature=creature, dt=dt, seek_results=seek_results[creature_id], noise_std=noise_std)

        # ----------------- die / eat / reproduce (+create list of creatures to die or reproduce) ----------------------

        list_creatures_reproduce = []
        list_creature_die = []
        is_eat_grass = False

        for creature_id, creature in self.creatures.items():
            # death from age or fatigue
            if creature.age >= creature.max_age or creature.energy <= 0:
                list_creature_die.append(creature_id)
                continue
            else:
                creature.age += config.DT

                # check for food (first grass, if not found search for leaf if tall enough)
                is_eat = self.eat_food(creature=creature, seek_result=seek_results[creature_id], food_type='grass')
                if not is_eat and creature.height >= config.LEAF_HEIGHT:
                    _ = self.eat_food(creature=creature, seek_result=seek_results[creature_id], food_type='leaf')

                if is_eat:  # record if something was eaten
                    is_eat_grass = True

                # reproduce if able
                if (creature.energy > creature.reproduction_energy + config.MIN_LIFE_ENERGY and
                        creature.can_reproduce(self.step_counter)):
                    list_creatures_reproduce.append(creature_id)
                    creature.reproduced_at = self.step_counter
        if is_eat_grass:
            self.env.remove_grass_points()
            self.env.update_grass_kd_tree()
        # ------------------------ add the purge to the killing list ----------------------------

        if config.DO_PURGE:
            if self.purge or len(self.creatures) > config.MAX_NUM_CREATURES * config.PURGE_POP_PERCENTAGE:
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

    def eat_food(self, creature: Creature, seek_result: dict, food_type: str):
        # check if creature is full
        if creature.energy >= creature.max_energy:
            return False

        # get food points
        is_eat = False
        food_points, food_energy = [], 0

        # This for serves the case of several eyes
        for key, value in seek_result.items():
            if key.startswith(food_type) and not value == None:
                food_points.append(value)

        if food_type == 'grass':
            food_energy = config.GRASS_ENERGY
        elif food_type == 'leaf':
            food_energy = config.LEAF_ENERGY

        if len(food_points) > 0:
            if len(food_points) > 1:
                # candidate_indices = kd_tree.query_ball_point(eye_position, creature.vision_limit) TODO: use the kd_tree here too!
                food_distances = [food_point[:2] for food_point in food_points]
                # DELETE: food_distances = [np.linalg.norm(food_point[:2] - creature.position)
                #                   for food_point in food_points]
                closest_food_index = np.argmin(food_distances)
            else:
                food_distances = food_points[0][:2]
                closest_food_index = 0
            closest_food_distance = np.min(food_distances[closest_food_index])
            closest_food_point = self.env.grass_points[food_points[closest_food_index][2]]
            # if someone got there first
            if closest_food_point in self.env.grass_remove_list:
                return False

            if closest_food_distance <= config.FOOD_DISTANCE_THRESHOLD:
                # creature eat food
                creature.eat(food_type=food_type, food_energy=food_energy)
                creature.log_eat.append(self.step_counter)

                # remove food from environment
                self.env.grass_remove_list.append(closest_food_point)
                # self.env.grass_points.remove(closest_food_point)  # TODO:moved outside - check if it's fine
                # self.env.update_grass_kd_tree()  # TODO:moved outside - check if it's fine
                is_eat = True

        return is_eat

    def check_abort_simulation(self):
        if len(self.creatures) > config.MAX_NUM_CREATURES:
            print(f'step={self.step_counter}: Too many creatures, simulation is too slow.')
            self.abort_simulation = True
        elif len(self.creatures) <= 0:
            if not self.abort_simulation:
                print(f'\nstep={self.step_counter}: all creatures are dead :(.')
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
        global lineage_graph, traits_scat, quiv, scat, grass_scat, leaves_scat, agent_scat
        global fig, ax_lineage, ax_traits, ax_env, ax_brain, ax_agent_info_1, ax_agent_info_2, ax_agent_events, ax_life, progress_bar

        def init_fig():
            """
            Init the simulation figure.
            :return:
            """
            global lineage_graph, traits_scat, quiv, scat, grass_scat, leaves_scat, agent_scat
            global fig, ax_lineage, ax_traits, ax_env, ax_brain, ax_agent_info_1, ax_agent_info_2, ax_agent_events, ax_life, progress_bar

            # init fig with the grid layout with uneven ratios
            # TODO: fig, axes = set_animation_figure()
            fig = plt.figure(figsize=(16, 8))
            fig_grid = gridspec.GridSpec(2, 3, width_ratios=[1, 2, 1], height_ratios=[2, 1])  # 2:1 ratio for both axes
            ax_lineage = fig.add_subplot(fig_grid[0, 0])  # ancestor tree?
            ax_env = fig.add_subplot(fig_grid[0, 1])  # Large subplot (3/4 of figure)
            ax_brain = fig.add_subplot(fig_grid[0, 2])  # Smaller subplot (1/4 width, full height)
            ax_traits = fig.add_subplot(fig_grid[1, 0])  # placeholder
            ax_agent_info = fig.add_subplot(fig_grid[1, 1])  # Smaller subplot (1/4 height, full width)
            ax_agent_info_1 = ax_agent_info
            ax_agent_info_2 = ax_agent_info_1.twinx()
            # ax_agent_status = fig.add_subplot(fig_grid[1, 2])  # Smallest subplot (1/4 x 1/4)
            subgrid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=fig_grid[1, 2], width_ratios=[1, 4])
            ax_life = fig.add_subplot(subgrid[0, 0])
            ax_agent_events = fig.add_subplot(subgrid[0, 1])
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
            agent_scat = ax_env.scatter(
                [self.creatures[0].position[0]] * 2, [self.creatures[0].position[1]] * 2,
                # Repeat position for N=2 rings
                s=[60, 500],  # Different sizes for bullseye rings # config.FOOD_SIZE
                facecolors=['none', 'none'],
                edgecolors=['black', 'black'],
                linewidth=2.5,
                marker='o',
                zorder=4  # or 'x'
            )

            # Init lineage plot
            lineage_graph = ax_lineage.scatter([], [], c=[], s=50)

            # Init traits plot
            traits_scat = ax_traits.scatter([], [], c=[], s=50)

            # Initialize the progress bar to print
            if config.STATUS_EVERY_STEP:
                update_num = config.NUM_FRAMES * config.UPDATE_ANIMATION_INTERVAL
            else:
                update_num = config.NUM_FRAMES

            progress_bar = tqdm(total=update_num, desc=f"Alive: {len(self.creatures):4} | "
                                                       f"Children: {self.children_num:4} | "
                                                       f"Dead: {len(self.dead_creatures):4} | "
                                                       f"leaves: {len(self.env.leaf_points):3} | "
                                                       f"grass: {len(self.env.grass_points):3} | "
                                                       f"Progress")

        def init_func():
            """
            Function for simulation initialization.
            A way to make sure animation doesn't call update multiple times for initialization.
            Once we will learn how to update scat/quiv instead of redrawing them it can also
            reduce computation time (because blit=True)
            :return:
            """
            global lineage_graph, traits_scat, quiv, scat, grass_scat, leaves_scat, agent_scat
            global fig, ax_lineage, ax_traits, ax_env, ax_brain, ax_agent_info_1, ax_agent_info_2, ax_agent_events, ax_life, progress_bar

            return scat, quiv, grass_scat, leaves_scat, agent_scat, traits_scat

        # Function for simulation progress
        def update_func(frame):
            """
            The main function of the animation.
            This function runs a single frame of the animation.
            Each frame contain multiple simulation steps according to config.
            :param frame:
            :return: the variables that are updated (right now we are redrawing them)
            """
            global lineage_graph, traits_scat, quiv, scat, grass_scat, leaves_scat, agent_scat
            global fig, ax_lineage, ax_traits, ax_env, ax_brain, ax_agent_info_1, ax_agent_info_2, ax_agent_events, ax_life, progress_bar

            # abort simulation if no creatures left or there are too many creatures
            if self.abort_simulation:
                if config.DEBUG_MODE:
                    from matplotlib import use

                    use('TkAgg')
                    self.statistics_logs.plot_and_save_statistics_graphs(to_save=False)
                    # breakpoint()

                ax_env.set_title(f"Evolution Simulation ({frame=}, step={self.step_counter})")
                progress_bar.update(self.animation_update_interval)
                self.step_counter += self.animation_update_interval

                return scat, quiv, grass_scat, leaves_scat, agent_scat, traits_scat

            # Run steps of frame
            for step in range(self.animation_update_interval):
                # Do simulation step
                child_ids, dead_ids = self.do_step(dt=config.DT, noise_std=config.NOISE_STD)

                # abort simulation if there are too many creatures or no creatures left
                self.check_abort_simulation()

                # Update statistics logs
                self.statistics_logs.update_statistics_logs(creatures=self.creatures, env=self.env,
                                                            child_ids=child_ids, dead_ids=dead_ids)

                # Update the progress bar every step
                if config.STATUS_EVERY_STEP:
                    progress_bar.set_description(f"Alive: {len(self.creatures):4} | "
                                                 f"Children: {self.children_num:4} | "
                                                 f"Dead: {len(self.dead_creatures):4} | "
                                                 f"leaves: {len(self.env.leaf_points):3} | "
                                                 f"grass: {len(self.env.grass_points):3} | "
                                                 f"Progress")
                    progress_bar.update(1)  # or self.animation_update_interval outside the for loop

            # update the progress bar every frame
            if not config.STATUS_EVERY_STEP:
                progress_bar.set_description(f"Alive: {len(self.creatures):4} | "
                                             f"Children: {self.children_num:4} | "
                                             f"Dead: {len(self.dead_creatures):4} | "
                                             f"leaves: {len(self.env.leaf_points):3} | "
                                             f"grass: {len(self.env.grass_points):3} | "
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
                self.statistics_logs.plot_and_save_statistics_graphs(to_save=False)

                # breakpoint()
            # --------------------------- Plot --------------------------- #
            if 'grass_scat' in globals():
                try:  # in case it's empty
                    grass_scat.remove()
                except:
                    pass
            try:
                # Update creature positions and directions
                num_creatures_in_last_frame = len(self.positions)
                self.positions = np.array([creature.position for creature in self.creatures.values()])
                sizes = np.array([creature.mass for creature in self.creatures.values()]) * config.FOOD_SIZE  # / 10
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
                        scat.set_facecolor(colors)
                        scat.set_sizes(sizes)
                        quiv.set_offsets(self.positions)
                        quiv.set_facecolor(colors)
                        quiv.set_UVC(U, V)  # Update U (x-component) and V (y-component)
                    else:
                        # Clear scatter and quiver plots (positions & directions)
                        for obj_name in ['quiv', 'scat', 'grass_scat', 'leaves_scat']:  # , 'agent_scat']:
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
                    grass_scat = ax_env.scatter(grass_points[:, 0], grass_points[:, 1], c='lightgreen',
                                                edgecolors='black',
                                                s=10)

                # num_leaf_points_in_last_frame = TODO
                num_leaf_points_after_step = len(self.env.leaf_points)
                if num_leaf_points_after_step > 0:
                    # if num_leaf_points_after_step == num_grass_points_in_last_frame:
                    #     leaves_scat.set_offsets(np.array(self.env.leaf_points))
                    # else:
                    leaf_points = np.array(self.env.leaf_points)
                    leaves_scat = ax_env.scatter(leaf_points[:, 0], leaf_points[:, 1], c='darkgreen',
                                                 edgecolors='black',
                                                 s=20)

                ax_env.set_title(f"Evolution Simulation ({frame=}, step={self.step_counter})")

                # ----------------- update lineage scat ------------------

                trait_stacked_colored_histogram(
                    ax=ax_lineage,
                    creatures=self.creatures,
                    trait_name='mass',
                    num_bins=30, min_value=0, max_value=2)

                # ----------------- update traits scat -------------------
                traits_scat = plot_traits_scatter(ax=ax_traits,
                                                  creatures=self.creatures,
                                                  trait_x='mass', trait_y='height',
                                                  trait_x_min=0, trait_x_max=2,
                                                  trait_y_min=0, trait_y_max=0.2)

                # --------------------- focus on one agent ----------------------------

                if len(self.creatures) > 0:
                    ids = self.statistics_logs.creatures_id_history[-1]
                    if self.focus_id not in ids:
                        if self.id_count in ids:  # trying to use the youngest creature as agent
                            self.make_agent(self.id_count)
                        else:
                            self.make_agent(np.random.choice(list(self.creatures.keys())))
                    agent = self.creatures[self.focus_id]
                    agent_scat.set_offsets([agent.position, agent.position])
                    agent.brain.plot(ax_brain)
                    # ax_agent_info.clear()
                    agent.plot_rebalance(ax_agent_info_1, mode='energy_use')
                    agent.plot_rebalance(ax_agent_info_2, mode='speed')
                    agent.plot_live_status(ax_life, plot_horizontal=False)
                    agent.plot_acc_status(ax_agent_events, plot_type=1, curr_step=self.step_counter)

            except Exception as e:
                print(f'Error in simulation (update_func): cannot plot because {e}.')
                # breakpoint()

            return scat, quiv, grass_scat, leaves_scat, agent_scat, traits_scat

        # Run simulation
        try:
            init_fig()
            ani = animation.FuncAnimation(fig=fig, func=update_func, init_func=init_func, blit=True,
                                          frames=config.NUM_FRAMES, interval=config.FRAME_INTERVAL)
            # print('\nSimulation completed successfully. saving progress...')

        except KeyboardInterrupt:
            print('Simulation interrupted. saving progress...')

        finally:
            # Save animation
            ani.save(config.ANIMATION_FILEPATH, writer="ffmpeg", dpi=100)
            plt.close(fig)
            print(f'Simulation animation saved as {config.ANIMATION_FILEPATH.stem}.')

            # Plot and save creature statistics and env statistics summary graphs
            self.statistics_logs.plot_and_save_statistics_graphs(to_save=True)
