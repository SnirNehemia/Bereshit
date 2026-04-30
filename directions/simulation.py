import time

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec

import importlib

from b_basic.creatures.creature import Creature
from b_basic.sim_config import sim_config
from c_models.physical_model_factory import PhysicalModelFactory
from directions import simulation_utils
from e_logs import traits_evolution
from e_logs.statistics_logs import StatisticsLogs

global lineage_graph, traits_scat, quiv, scat, grass_scat, leaves_scat, agent_scat
global fig, axes, progress_bar


class Simulation:
    """
    Manages the simulation of creatures within an environment.
    Handles perception, decision-making, movement, collision detection, and vegetation updates.
    Implements multichannel perception by using separate KDTree queries for each target type.
    """

    def __init__(self):
        # Init physical model
        self.physical_model = PhysicalModelFactory.create()

        # Init environment
        self.env = simulation_utils.init_environment()

        # Init creatures (ensuring they are not in forbidden areas)
        brain_module = importlib.import_module(f"c_models.brain_models.{sim_config.config.BRAIN_TYPE}")
        brain_obj = getattr(brain_module, 'Brain')
        self.creatures = simulation_utils.init_creatures(env=self.env, brain_obj=brain_obj)

        self.dead_creatures = dict()
        self.creatures_ids = list(self.creatures.keys())
        self.positions = np.array([creature.position
                                   for creature in self.creatures.values()])

        # Build a KDTree for creature positions.
        self.creatures_kd_tree = simulation_utils.build_creatures_kd_tree(positions=self.positions)
        self.num_children = 0

        # simulation control parameters
        self.num_creatures_in_last_frame = len(self.creatures)
        self.id_count = self.num_creatures_in_last_frame - 1
        self.num_creatures_threshold = \
            int(sim_config.config.PURGE_POPULATION_PERCENTAGE * sim_config.config.MAX_NUM_CREATURES)
        self.abort_simulation = False
        self.num_steps_per_frame = 0
        self.step_counter = 0
        self.focus_id = 0
        self.creatures[self.focus_id].make_agent()
        sim_config.config.OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)

        # statistics logs
        self.statistics_logs = StatisticsLogs()
        self.statistics_logs.num_frames = sim_config.config.NUM_FRAMES
        self.statistics_logs.total_num_steps = \
            simulation_utils.calc_total_num_steps(
                num_steps_from_frame_dict=sim_config.config.NUM_STEPS_FROM_FRAME_DICT,
                up_to_frame=sim_config.config.NUM_FRAMES)

        print('\n----------------------------------------')
        print(f'Simulation {sim_config.config.OUTPUT_FOLDER.stem}: '
              f'num frames = {self.statistics_logs.num_frames}, '
              f'num steps = {self.statistics_logs.total_num_steps}')
        print('----------------------------------------')

        if sim_config.config.DEBUG_MODE:
            np.seterr(all='raise')  # Convert NumPy warnings into exceptions

    def seek(self,
             creature: Creature, i_creature: int,
             creatures_positions: list | np.ndarray,
             creatures_indices_to_kill: list):
        """
        Search targets for each eye (angle_offset, aperture) in each channel (grass/creature/...)
        Returns (distance, signed_angle, idx) if a target is found within half the aperture, else None.
        """
        kd_tree = []
        seek_results = {}
        candidates_indices_to_remove = []

        for i_eye in range(len(creature.eyes)):
            for channel_name in sim_config.config.CHANNELS_LIST:
                candidate_points = np.array([])
                if channel_name == 'grass':
                    if len(self.env.grass_points) > 0:
                        kd_tree = self.env.grass_kd_tree
                        candidate_points = np.array(self.env.grass_points)
                        candidates_indices_to_remove = self.env.grass_indices_to_remove
                        i_creature = -1
                elif channel_name == 'leaf':
                    if len(self.env.leaf_points) > 0:
                        kd_tree = self.env.leaves_kd_tree
                        candidate_points = np.array(self.env.leaf_points)
                        candidates_indices_to_remove = self.env.leaf_indices_to_remove
                        i_creature = -1
                # elif channel_name == 'water':
                #     candidate_points = np.array([[self.env.water_source[0], self.env.water_source[1]]])
                #     candidates_indices_to_remove = self.env.water_remove_list
                elif channel_name == 'creature':
                    kd_tree = self.creatures_kd_tree
                    candidate_points = creatures_positions
                    candidates_indices_to_remove = creatures_indices_to_kill

                if len(candidate_points) > 0:
                    result = simulation_utils.detect_target_from_kdtree(
                        creature=creature,
                        i_creature=i_creature,
                        eye_idx=i_eye,
                        kd_tree=kd_tree,
                        candidate_points=candidate_points,
                        candidates_indices_to_remove=candidates_indices_to_remove)
                else:
                    result = None

                eye_channel_name = f'{i_eye}_{channel_name}'
                seek_results[eye_channel_name] = result

        return seek_results

    def do_step(self):
        """
        Advances the simulation by one time step.
        For each creature:
          - Perceives its surroundings with both eyes.
          - Constructs an input vector for the brain.
          - Receives a decision (delta_angle, delta_speed) to update its velocity.
          - Checks for collisions with obstacles (black areas) and stops if necessary.
        Then, moves creatures and updates the vegetation.
        """

        # ------------------------------------ Creatures actions ------------------------------------
        creatures_indices_to_kill = []
        creatures_ids_to_reproduce = []
        to_update_kd_tree = {food_type: False
                             for food_type in sim_config.config.INIT_HERBIVORE_DIGEST_DICT.keys()}

        for i_creature, creature in enumerate(self.creatures.values()):
            creature_id = creature.creature_id
            # Death from age
            if creature.age >= creature.max_age:
                self.statistics_logs.death_causes_dict['age'].append(creature_id)
                creatures_indices_to_kill.append(i_creature)
            # Death from fatigue
            elif creature.energy <= 0:
                self.statistics_logs.death_causes_dict['fatigue'].append(creature_id)
                creatures_indices_to_kill.append(i_creature)

            # Check if creature is already killed (age/fatigue/eaten/purged)
            if i_creature in creatures_indices_to_kill:
                continue
            else:
                # Creature get older
                creature.age += sim_config.config.DT

                # Seek in environment
                seek_result = self.seek(
                    creature=creature, i_creature=i_creature,
                    creatures_positions=self.positions,
                    creatures_indices_to_kill=creatures_indices_to_kill)

                # Use brain to move
                self.use_brain(
                    creature=creature,
                    seek_result=seek_result)

                # Check for nearby food
                eaten_food_type, food_idx = self.eat_food(
                    creature=creature,
                    seek_result=seek_result,
                    creatures_indices_to_kill=creatures_indices_to_kill,
                    step_counter=self.step_counter)

                # record eaten_food_type to update kd tree afterward
                if eaten_food_type is not None:
                    to_update_kd_tree[eaten_food_type] = True
                    if eaten_food_type == 'creature':
                        self.statistics_logs.death_causes_dict['eaten'] \
                            .append(self.creatures_ids[food_idx])

                # reproduce if able
                energy_needed_to_reproduce = creature.reproduction_energy + sim_config.config.MIN_LIFE_ENERGY
                if creature.energy > energy_needed_to_reproduce and \
                        creature.can_reproduce(self.step_counter):
                    creatures_ids_to_reproduce.append(creature_id)

        # ---------------------------- After all creatures actions ----------------------------------
        # Convert from indices to ids
        creatures_ids_to_kill = list(np.array(self.creatures_ids)[creatures_indices_to_kill])

        # Reproduction
        new_children_ids, self.num_children, self.id_count = \
            simulation_utils.reporduce_creatures(creatures_ids_to_reproduce=creatures_ids_to_reproduce,
                                                 creatures=self.creatures,
                                                 id_count=self.id_count,
                                                 num_children=self.num_children,
                                                 step_counter=self.step_counter)

        # Kill creatures
        simulation_utils.kill_creatures(
            creatures_ids_to_kill=creatures_ids_to_kill,
            creatures=self.creatures,
            dead_creatures=self.dead_creatures)

        # Purge
        if sim_config.config.DO_PURGE:
            purged_creatures_ids = simulation_utils.do_purge(
                num_creatures_threshold=self.num_creatures_threshold,
                creatures=self.creatures,
                dead_creatures=self.dead_creatures,
                step_counter=self.step_counter,
                statistics_logs=self.statistics_logs)
            creatures_ids_to_kill.extend(purged_creatures_ids)

        # Ensure creatures kd tree is updated if creatures are added/killed.
        if len(new_children_ids) > 0 or len(creatures_ids_to_kill) > 0:
            to_update_kd_tree['creature'] = True

        # Update creatures ids and positions
        self.creatures_ids = list(self.creatures.keys())
        self.positions = np.array([creature.position for creature in self.creatures.values()])

        # Update KD trees if conditions are met
        self.creatures_kd_tree = simulation_utils.update_kd_trees(
            env=self.env,
            positions=self.positions,
            creatures_kd_tree=self.creatures_kd_tree,
            to_update_kd_tree=to_update_kd_tree,
            step_counter=self.step_counter)

        # Update step counter
        self.step_counter += 1

        return new_children_ids, creatures_ids_to_kill

    def update_logs(self, child_ids, dead_ids):
        # Update creatures logs (after movement, eating and reproduction)
        simulation_utils.update_creatures_logs(creatures=self.creatures)

        # Update statistics logs
        self.statistics_logs.update_statistics_logs(creatures=self.creatures, env=self.env,
                                                    child_ids=child_ids, dead_ids=dead_ids,
                                                    step_counter=self.step_counter)
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
        global fig, axes, progress_bar

        def init_fig():
            """
            Init the simulation figure.
            :return:
            """
            global lineage_graph, traits_scat, quiv, scat, grass_scat, leaves_scat, agent_scat
            global fig, axes, progress_bar

            # init fig with the grid layout with uneven ratios
            fig = plt.figure(figsize=(16, 8))
            fig_grid = gridspec.GridSpec(2, 3, width_ratios=[1, 2, 1], height_ratios=[1, 1])
            axes = []
            # [0] -> color histogram
            # [1] -> environment
            # [2] -> brain
            # [3] -> traits scatter
            # [4] -> live status
            # force
            # force angle
            # speed
            # energy
            # [5] -> event status
            # energy and age
            # reproduce and meals
            axes.append(fig.add_subplot(fig_grid[0, 0]))  # ancestor tree?
            axes.append(fig.add_subplot(fig_grid[0, 1]))  # Large subplot (3/4 of figure)
            axes.append(fig.add_subplot(fig_grid[0, 2]))  # Smaller subplot (1/4 width, full height)
            axes.append(fig.add_subplot(fig_grid[1, 0]))  # placeholder
            # ax_agent_info = fig.add_subplot(fig_grid[1, 1])  # Smaller subplot (1/4 height, full width)
            # axes[4][0][0] = ax_agent_info
            # axes[4][1] = axes[4][0][0].twinx()
            axes.append([])
            subgrid = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=fig_grid[1, 1], height_ratios=[1, 1])
            axes[-1].append(fig.add_subplot(subgrid[0, 0]))
            axes[-1].append(fig.add_subplot(subgrid[1, 0]))
            # axes[-1].append([])
            # axes[-1][-1].append(fig.add_subplot(subgrid[0, 0]))
            # twin_axis = axes[-1][-1][-1].twinx()
            # axes[-1][-1].append(twin_axis)
            axes[-1].append(fig.add_subplot(subgrid[0, 1]))
            axes[-1].append(fig.add_subplot(subgrid[1, 1]))
            # axes[-1].append(fig.add_subplot(subgrid[1, 1]))
            # ax_agent_status = fig.add_subplot(fig_grid[1, 2])  # Smallest subplot (1/4 x 1/4)
            axes.append([])
            subgrid = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=fig_grid[1, 2], height_ratios=[1, 1, 1])
            axes[-1].append(fig.add_subplot(subgrid[0, 0]))
            axes[-1].append(fig.add_subplot(subgrid[1, 0]))
            axes[-1].append(fig.add_subplot(subgrid[2, 0]))

            extent = self.env.get_extent()
            axes[1].set_xlim(extent[0], extent[1])
            axes[1].set_ylim(extent[2], extent[3])
            axes[1].set_title("Evolution Simulation")

            # axes[0] = fig.add_subplot(fig_grid[0, 0])  # ancestor tree?
            # axes[1] = fig.add_subplot(fig_grid[0, 1])  # Large subplot (3/4 of figure)
            # axes[2] = fig.add_subplot(fig_grid[0, 2])  # Smaller subplot (1/4 width, full height)
            # axes[3] = fig.add_subplot(fig_grid[1, 0])  # placeholder
            # # ax_agent_info = fig.add_subplot(fig_grid[1, 1])  # Smaller subplot (1/4 height, full width)
            # # axes[4][0][0] = ax_agent_info
            # # axes[4][1] = axes[4][0][0].twinx()
            # subgrid = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=fig_grid[1, 1], height_ratios=[1, 1, 1])
            # axes[4][0][0] = fig.add_subplot(subgrid[0, 0])
            # axes[4][0][1] = axes[4][0][0].twinx()
            # axes[4][1] = fig.add_subplot(subgrid[1, 0])
            # axes[4][2] = fig.add_subplot(subgrid[2, 0])
            # # ax_agent_status = fig.add_subplot(fig_grid[1, 2])  # Smallest subplot (1/4 x 1/4)
            # subgrid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=fig_grid[1, 2], width_ratios=[1, 4])
            # axes[5][0] = fig.add_subplot(subgrid[0, 0])
            # axes[5][1] = fig.add_subplot(subgrid[0, 1])
            # extent = self.env.get_extent()
            # ax_env.set_xlim(extent[0], extent[1])
            # ax_env.set_ylim(extent[2], extent[3])
            # ax_env.set_title("Evolution Simulation")

            # Display the environment map with origin='lower' to avoid vertical mirroring
            axes[1].imshow(self.env.map_data, extent=extent, alpha=0.3, origin='lower')  # , aspect='auto')

            # Draw the water source
            water_x, water_y, water_r = self.env.water_source
            water_circle = Circle((water_x, water_y), water_r, color='blue', alpha=0.3)
            axes[1].add_patch(water_circle)

            # Initial creature positions
            colors = [creature.color for creature in self.creatures.values()]
            edge_colors = ['r' if creature.digest_dict['creature'] > 0 else 'g' for creature in self.creatures.values()]
            sizes = np.array(
                [creature.mass for creature in self.creatures.values()]) * sim_config.config.FOOD_SIZE / 100
            scat = axes[1].scatter(self.positions[:, 0], self.positions[:, 1],
                                   c=colors, s=sizes, edgecolor=edge_colors, linewidth=1.5,
                                   transform=axes[1].transData)

            # Create quiver arrows for creature headings
            U, V = [], []
            for creature in self.creatures.values():
                if creature.speed > 0:
                    U.append(creature.velocity[0])
                    V.append(creature.velocity[1])
                else:
                    U.append(0)
                    V.append(0)
            quiv = axes[1].quiver(self.positions[:, 0], self.positions[:, 1], U, V,
                                  color=colors, scale=500, width=0.005)  # 'black'

            # Scatter food points for vegetation
            grass_scat = axes[1].scatter([], [], c='lightgreen', edgecolors='black', s=10)
            leaves_scat = axes[1].scatter([], [], c='darkgreen', edgecolors='black', s=10)
            agent_scat = axes[1].scatter(
                [self.creatures[0].position[0]] * 2, [self.creatures[0].position[1]] * 2,
                # Repeat position for N=2 rings
                s=[60, 500],  # Different sizes for bullseye rings # sim_config.config.FOOD_SIZE
                facecolors=['none', 'none'],
                edgecolors=['black', 'black'],
                linewidth=2.5,
                marker='o',
                zorder=4  # or 'x'
            )

            # Init lineage plot
            lineage_graph = axes[0].scatter([], [], c=[], s=50)

            # Init traits plot
            traits_scat = axes[3].scatter([], [], c=[], s=50)

            # Initialize the progress bar to print
            if sim_config.config.STATUS_EVERY_STEP:
                update_num = simulation_utils.calc_total_num_steps(
                    num_steps_from_frame_dict=sim_config.config.NUM_STEPS_FROM_FRAME_DICT,
                    up_to_frame=sim_config.config.NUM_FRAMES)
            else:
                update_num = sim_config.config.NUM_FRAMES

            progress_bar = tqdm(total=update_num,
                                desc=f"Herbivores: {0:4} | "
                                     f"Carnivores: {0:4} | "
                                     f"Alive: {0:4} | "
                                     f"Children: {0:4} | "
                                     f"Dead: {0:4} | "
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
            global fig, axes, progress_bar

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
            global fig, axes, progress_bar

            # check num steps per frame
            self.num_steps_per_frame = simulation_utils.calc_num_steps_per_frame(frame=frame)

            # abort simulation if no creatures left or there are too many creatures
            if self.abort_simulation:
                axes[1].set_title(f"Evolution Simulation ({frame=}, step={self.step_counter})")
                progress_bar.update(self.num_steps_per_frame)
                self.step_counter += self.num_steps_per_frame

                return scat, quiv, grass_scat, leaves_scat, agent_scat, traits_scat

            # ------------------- EVERY STEP ------------------- #
            # Run steps of frame
            for step in range(self.num_steps_per_frame):
                # Do simulation step
                child_ids, dead_ids = self.do_step()

                # Update logs
                self.update_logs(child_ids=child_ids, dead_ids=dead_ids)

                # abort simulation if there are too many creatures or no creatures left
                if not self.abort_simulation:
                    self.abort_simulation = simulation_utils.check_abort_simulation(
                        creatures=self.creatures,
                        step_counter=self.step_counter)

                # Update the progress bar every step
                if sim_config.config.STATUS_EVERY_STEP:
                    progress_bar.set_description(
                        f"Herbivores: {self.statistics_logs.num_herbivores_per_step[-1]:4} | "
                        f"Carnivores: {self.statistics_logs.num_carnivores_per_step[-1]:4} | "
                        f"Alive: {len(self.creatures):4} | "
                        f"Children: {self.num_children:4} | "
                        f"Dead: {len(self.dead_creatures):4} | "
                        f"Progress")
                    progress_bar.update(1)  # or self.num_steps_per_frame outside the for loop

            # -------------------- EVERY FRAME ------------------------- #
            # update the progress bar every frame
            if not sim_config.config.STATUS_EVERY_STEP:
                progress_bar.set_description(
                    f"Herbivores: {self.statistics_logs.num_herbivores_per_step[-1]:4} | "
                    f"Carnivores: {self.statistics_logs.num_carnivores_per_step[-1]:4} | "
                    f"Alive: {len(self.creatures):4} | "
                    f"Children: {self.num_children:4} | "
                    f"Dead: {len(self.dead_creatures):4} | "
                    f"Progress")
                progress_bar.update(1)  # or self.num_steps_per_frame outside the for loop

            # Plot
            if 'grass_scat' in globals():
                try:  # in case it's empty
                    grass_scat.remove()
                except:
                    pass
            try:
                # Update creature positions and directions
                colors = [creature.color for creature in self.creatures.values()]
                sizes = np.array([creature.mass for creature in self.creatures.values()]) \
                        * sim_config.config.FOOD_SIZE
                edge_colors = ['r' if creature.digest_dict['creature'] > 0
                               else 'g' for creature in self.creatures.values()]

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
                    if num_creatures_after_step == self.num_creatures_in_last_frame:
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
                        scat = axes[1].scatter(self.positions[:, 0], self.positions[:, 1],
                                               c=colors, s=sizes, edgecolor=edge_colors, linewidth=1.5)
                        quiv = axes[1].quiver(self.positions[:, 0], self.positions[:, 1], U, V,
                                              color=colors, scale=500, width=0.005)
                else:
                    # plot place holder
                    scat = axes[1].scatter([1], [1])
                    quiv = axes[1].quiver([1], [1], [1], [1])

                # Update vegetation scatter data
                num_grass_points_after_step = len(self.env.grass_points)
                if num_grass_points_after_step > 0:
                    # if num_grass_points_after_step == num_grass_points_in_last_frame:
                    #     grass_scat.set_offsets(np.array(self.env.grass_points))
                    # else:
                    grass_points = np.array(self.env.grass_points)
                    grass_scat = axes[1].scatter(grass_points[:, 0], grass_points[:, 1], c='lightgreen',
                                                 edgecolors='black',
                                                 s=10)

                num_leaf_points_after_step = len(self.env.leaf_points)
                if num_leaf_points_after_step > 0:
                    # if num_leaf_points_after_step == num_grass_points_in_last_frame:
                    #     leaves_scat.set_offsets(np.array(self.env.leaf_points))
                    # else:
                    leaf_points = np.array(self.env.leaf_points)
                    leaves_scat = axes[1].scatter(leaf_points[:, 0], leaf_points[:, 1], c='darkgreen',
                                                  edgecolors='black',
                                                  s=20)

                axes[1].set_title(f"Evolution Simulation ({frame=}, step={self.step_counter})")

                # ----------------- update lineage scat ------------------

                traits_evolution.trait_stacked_colored_histogram(
                    ax=axes[0],
                    creatures=self.creatures,
                    trait_name='mass',
                    num_bins=30, min_value=0, max_value=2)

                # ----------------- update traits scat -------------------
                traits_scat = traits_evolution.plot_traits_scatter(ax=axes[3],
                                                                   creatures=self.creatures,
                                                                   trait_x='mass', trait_y='height',
                                                                   trait_x_min=0, trait_x_max=2,
                                                                   trait_y_min=0, trait_y_max=0.2)

                # --------------------- focus on one agent ----------------------------

                # change agent if current agent is dead (try to change to youngest, if not possible choose randomly)
                if len(self.creatures) > 0:
                    # check next agent id
                    ids = self.statistics_logs.creatures_id_history[-1]
                    if self.focus_id not in ids:
                        if self.id_count in ids:
                            self.focus_id = self.id_count
                        else:
                            self.focus_id = np.random.choice(list(self.creatures.keys()))

                    # choose agent
                    agent = self.creatures[self.focus_id]
                    agent.make_agent()

                    # plot agent
                    agent_scat.set_offsets([agent.position, agent.position])
                    agent.brain.plot(axes[2])
                    # ax_agent_info.clear()
                    # # plot.plot_rebalance(axes[4][0][0], agent, mode='force', add_title=True, ax_secondary=None)  # ax_secondary=axes[4][0][1]
                    # plot.plot_rebalance(axes[4][0], agent, mode='force', add_title=True)
                    # plot.plot_rebalance(axes[4][1], agent, mode='friction angle', add_x_label=False)
                    # plot.plot_rebalance(axes[4][2], agent, mode='speed')
                    # plot.plot_rebalance(axes[4][3], agent, mode='energy_use', add_x_label=True)
                    # # plot.plot_rebalance(axes[4][3], agent, mode='power', add_x_label=True)
                    # plot.plot_live_status(axes[5][2], agent, plot_horizontal=True)
                    # plot.plot_live_status_power(axes[5][1], agent, plot_horizontal=True)
                    # plot.plot_acc_status(axes[5][0], agent, plot_type=1, curr_step=self.step_counter)

                # Update num creatures in last frame (to update plot efficiently)
                self.num_creatures_in_last_frame = len(self.creatures)

            except Exception as e:
                # breakpoint('Error in simulation (update_func): cannot plot')
                print(f'Error in simulation (update_func): cannot plot because {e}.')
                # breakpoint()

            return scat, quiv, grass_scat, leaves_scat, agent_scat, traits_scat

        # Run simulation
        start_time = time.time()
        try:
            init_fig()
            ani = animation.FuncAnimation(fig=fig, func=update_func, init_func=init_func, blit=True,
                                          frames=sim_config.config.NUM_FRAMES,
                                          interval=sim_config.config.FRAME_INTERVAL)
            # print('\nSimulation completed successfully. saving progress...')

        except KeyboardInterrupt:
            print('Simulation interrupted. saving progress...')

        finally:
            # Save animation
            ani.save(sim_config.config.ANIMATION_FILEPATH, writer="ffmpeg", dpi=100)
            plt.close(fig)
            # print(f'Simulation animation saved as {sim_config.config.ANIMATION_FILEPATH.stem}.')

            # copy config to output folder
            simulation_utils.copy_config_file_to_output_folder()

            # save total time in statistics logs
            total_time = time.time() - start_time
            self.statistics_logs.total_time = total_time

            # Save statistics logs to json file
            self.statistics_logs.to_json(filepath=sim_config.config.STATISTICS_LOGS_JSON_FILEPATH)

            return total_time

    def use_brain(self, creature: Creature,
                  seek_result: dict):
        brain_input = simulation_utils.get_brain_input(creature=creature, seek_result=seek_result)
        decision = creature.think(brain_input)
        self.physical_model.move_creature(creature=creature, env=self.env, decision=decision)

    def eat_food(self,
                 creature: Creature,
                 seek_result: dict,
                 creatures_indices_to_kill: list[int],
                 step_counter: int):
        """

        :param creature:
        :param seek_result: dict of '{channel}_{eye_idx}': [distance, angle, idx]
        :param creatures_indices_to_kill
        :param step_counter:
        :return: eaten_food_type: 'grass'/'leaf'/'creature' or None if no food was eaten
        """
        # check if creature is full
        if creature.energy >= creature.max_energy:
            return None, None

        # init relevant variables
        food_energy = 0
        food_idx = None
        eaten_food_type = None
        is_food_condition_met = False
        food_list, food_to_remove_list = [], []

        # Eat food if conditions are met
        for key, result in seek_result.items():
            # Check if eye found something and if creature can eat it
            food_type = key.split('_')[1]
            if result is None or creature.digest_dict[food_type] == 0:
                continue
            else:
                food_distance, food_angle, food_idx = result

                if food_type == 'grass':
                    food_to_remove_list = self.env.grass_indices_to_remove
                    food_energy = sim_config.config.GRASS_ENERGY
                    is_food_condition_met = True
                elif food_type == 'leaf':
                    food_to_remove_list = self.env.leaf_indices_to_remove
                    food_energy = sim_config.config.LEAF_ENERGY
                    is_food_condition_met = \
                        creature.height >= self.env.leaf_points[food_idx].height
                elif food_type == 'creature':
                    food_to_remove_list = creatures_indices_to_kill
                    prey_id = self.creatures_ids[food_idx]
                    prey = self.creatures[prey_id]
                    food_energy = self.physical_model.energy_conversion_factors['mass_energy'] * prey.mass

                    is_child = creature.creature_id == prey.parent_id
                    is_father = creature.parent_id == prey.creature_id
                    is_food_condition_met = creature.mass >= prey.mass and not is_child and not is_father

                # Check if conditions to eat food are met (if so, eat and add to food_remove_list)
                is_food_available = food_idx not in food_to_remove_list
                is_food_close_enough = food_distance <= sim_config.config.FOOD_DISTANCE_THRESHOLD

                if is_food_available and is_food_close_enough and is_food_condition_met:
                    # eat food and record it
                    self.physical_model.digest_food(
                        creature=creature, food_type=food_type, food_energy=food_energy)
                    creature.log.add_record(f'eat_{food_type}', step_counter)

                    # remove food
                    food_to_remove_list.append(food_idx)
                    eaten_food_type = food_type
                    break

        return eaten_food_type, food_idx
