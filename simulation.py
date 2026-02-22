import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec

import simulation_utils
from creature import Creature
from input.codes import sim_config
import plot_utils as plot

from input.codes.physical_model_factory import PhysicalModelFactory
from input.codes import repos_utils
from statistics_logs import StatisticsLogs
from traits_evolution.trait_stacked_colored_histogram import trait_stacked_colored_histogram
from traits_evolution.traits_scatter import plot_traits_scatter

import importlib

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
        self.physical_model, self.physical_model_path = \
            PhysicalModelFactory.create(config_name=sim_config.config.PHYSICAL_MODEL_CONFIG_NAME)

        # Init environment
        self.env = simulation_utils.init_environment()

        # Init creatures (ensuring they are not in forbidden areas)
        brain_module = importlib.import_module(f"brain_models.{sim_config.config.BRAIN_TYPE}")
        brain_obj = getattr(brain_module, 'Brain')
        self.creatures = simulation_utils.init_creatures(env=self.env, brain_obj=brain_obj)
        self.dead_creatures = dict()
        self.positions = []

        # Build a KDTree for creature positions.
        self.creatures_kd_tree = simulation_utils.build_creatures_kd_tree(creatures=self.creatures)
        self.children_num = 0

        # simulation control parameters
        self.abort_simulation = False
        self.num_steps_per_frame = 0
        self.step_counter = 0  # Initialize step counter
        self.id_count = sim_config.config.NUM_CREATURES - 1
        self.focus_id = 0
        self.creatures[self.focus_id].make_agent()
        sim_config.config.OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)

        self.do_purge = True  # flag for purge events

        # statistics logs
        self.statistics_logs = StatisticsLogs()

        if sim_config.config.DEBUG_MODE:
            np.seterr(all='raise')  # Convert NumPy warnings into exceptions

        self.statistics_logs.num_frames = sim_config.config.NUM_FRAMES
        self.statistics_logs.total_num_steps = \
            simulation_utils.calc_total_num_steps(sim_config.config.NUM_FRAMES)
        print('\n----------------------------------------')
        print(f'Simulation {sim_config.config.OUTPUT_FOLDER.stem}: '
              f'num frames = {self.statistics_logs.num_frames}, '
              f'num steps = {self.statistics_logs.total_num_steps}')
        print('----------------------------------------')

    def seek(self,
             creatures_ids_to_kill: list,
             creatures_positions: list,
             creature: Creature, noise_std: float = 0.0,
             ):
        """
        Uses the specified eye (given by eye_params: (angle_offset, aperture))
        to detect a nearby target.
        Computes the eye's viewing direction by rotating the creature's heading by angle_offset.
        Returns (distance, signed_angle, idx) if a target is found within half the aperture, else None.
        """
        channel_results = {}

        kd_tree = []
        for i_eye, eye_params in enumerate(creature.eyes_params):
            for eye_channel in creature.eyes_channels:
                candidate_points = np.array([])
                if eye_channel == 'grass':
                    if len(self.env.grass_points) > 0:
                        kd_tree = self.env.grass_kd_tree
                        candidate_points = np.array(self.env.grass_points)
                        candidates_to_remove_list = self.env.grass_remove_list
                elif eye_channel == 'leaf':
                    if len(self.env.leaf_points) > 0:
                        candidate_points = np.array(self.env.leaf_points)
                        candidates_to_remove_list = self.env.leaf_remove_list
                # elif eye_channel == 'water':
                #     candidate_points = np.array([[self.env.water_source[0], self.env.water_source[1]]])
                #     candidates_to_remove_list = self.env.water_remove_list
                elif eye_channel == 'creature':
                    kd_tree = self.creatures_kd_tree
                    candidate_points = creatures_positions
                    candidates_to_remove_list = creatures_ids_to_kill

                if len(candidate_points) > 0:
                    result = simulation_utils.detect_target_from_kdtree(
                        creature=creature,
                        eye_params=eye_params,
                        kd_tree=kd_tree,
                        candidate_points=candidate_points,
                        candidates_to_remove_list=candidates_to_remove_list,
                        noise_std=noise_std)
                else:
                    result = None

                channel_name = f'{eye_channel}_{i_eye}'
                channel_results[channel_name] = result

        return channel_results

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

        # ------------------------------------ Creatures actions ------------------------------------

        creatures_ids_to_reproduce = []
        creatures_ids_to_kill = []
        to_update_kd_tree = {food_type: False
                             for food_type in sim_config.config.INIT_HERBIVORE_DIGEST_DICT.keys()}
        creatures_positions = np.array([c.position for c in self.creatures.values()])

        for creature_id, creature in self.creatures.items():
            # death from age/fatigue/eaten by another creature
            if creature.age >= creature.max_age:
                self.statistics_logs.death_causes_dict['age'].append(creature_id)
                creatures_ids_to_kill.append(creature_id)
            elif creature.energy <= 0:
                self.statistics_logs.death_causes_dict['fatigue'].append(creature_id)
                creatures_ids_to_kill.append(creature_id)

            if creature_id in creatures_ids_to_kill:
                continue
            else:
                # Creature get older
                creature.age += sim_config.config.DT

                # Seek in environment
                seek_result = self.seek(
                    creature=creature, noise_std=noise_std,
                    creatures_ids_to_kill=creatures_ids_to_kill,
                    creatures_positions=creatures_positions)

                # Use brain to move
                self.use_brain(
                    creature=creature,
                    seek_result=seek_result,
                    dt=dt)

                # Check for nearby food
                eaten_food_type, food_point = self.eat_food(
                    creature=creature,
                    seek_result=seek_result,
                    creatures_ids_to_kill=creatures_ids_to_kill,
                    step_counter=self.step_counter)

                # record eaten_food_type to update kd tree afterward
                if eaten_food_type is not None:
                    to_update_kd_tree[eaten_food_type] = True
                    if eaten_food_type == 'creature':
                        self.statistics_logs.death_causes_dict['eaten'].append(food_point)

                # reproduce if able
                energy_needed_to_reproduce = creature.reproduction_energy + sim_config.config.MIN_LIFE_ENERGY
                if (creature.energy > energy_needed_to_reproduce and
                        creature.can_reproduce(self.step_counter)):
                    creatures_ids_to_reproduce.append(creature_id)

        # ---------------------------- After all creatures actions ----------------------------------
        # Reproduction
        new_child_ids, self.children_num, self.id_count = \
            simulation_utils.reporduce_creatures(creatures_ids_to_reproduce=creatures_ids_to_reproduce,
                                                 creatures=self.creatures,
                                                 id_count=self.id_count,
                                                 children_num=self.children_num,
                                                 step_counter=self.step_counter)

        # Purge
        self.do_purge, creatures_ids_to_purge = \
            simulation_utils.do_purge(
                do_purge=self.do_purge,
                creatures=self.creatures,
                creatures_ids_to_kill=creatures_ids_to_kill,
                creatures_ids_to_reproduce=creatures_ids_to_reproduce,
                step_counter=self.step_counter)
        self.statistics_logs.death_causes_dict['purge'].extend(creatures_ids_to_purge)
        creatures_ids_to_kill.extend(creatures_ids_to_purge)

        # kill creatures
        new_dead_ids = simulation_utils.kill_creatures(
            creatures_ids_to_kill=creatures_ids_to_kill,
            creatures=self.creatures,
            dead_creatures=self.dead_creatures)

        # Ensure creatures kd tree is updated if creatures are added/killed.
        if len(new_child_ids) > 0 or len(new_dead_ids) > 0:
            to_update_kd_tree['creature'] = True

        # Update environment (generate new food points) and update KD trees if conditions are met
        self.creatures_kd_tree = simulation_utils.update_environment_and_kd_trees(
            env=self.env,
            creatures=self.creatures,
            creatures_kd_tree=self.creatures_kd_tree,
            to_update_kd_tree=to_update_kd_tree,
            step_counter=self.step_counter)

        # Update step counter
        self.step_counter += 1

        return new_child_ids, new_dead_ids

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
            self.positions = np.array([creature.position for creature in self.creatures.values()])
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
                update_num = simulation_utils.calc_total_num_steps(up_to_frame=sim_config.config.NUM_FRAMES)
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

            # Run steps of frame
            for step in range(self.num_steps_per_frame):
                # Do simulation step
                child_ids, dead_ids = self.do_step(dt=sim_config.config.DT, noise_std=sim_config.config.NOISE_STD)

                # Update creatures logs (after movement, eating and reproduction)
                simulation_utils.update_creatures_logs(creatures=self.creatures)

                # Update statistics logs
                self.statistics_logs.update_statistics_logs(creatures=self.creatures, env=self.env,
                                                            child_ids=child_ids, dead_ids=dead_ids,
                                                            step_counter=self.step_counter)

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
                        f"Children: {self.children_num:4} | "
                        f"Dead: {len(self.dead_creatures):4} | "
                        f"Progress")
                    progress_bar.update(1)  # or self.num_steps_per_frame outside the for loop

            # update the progress bar every frame
            if not sim_config.config.STATUS_EVERY_STEP:
                progress_bar.set_description(
                    f"Herbivores: {self.statistics_logs.num_herbivores_per_step[-1]:4} | "
                    f"Carnivores: {self.statistics_logs.num_carnivores_per_step[-1]:4} | "
                    f"Alive: {len(self.creatures):4} | "
                    f"Children: {self.children_num:4} | "
                    f"Dead: {len(self.dead_creatures):4} | "
                    f"Progress")
                progress_bar.update(1)  # or self.num_steps_per_frame outside the for loop

            # Do purge if PURGE_FRAME_FREQUENCY frames passed (to clear static agents)
            if sim_config.config.DO_PURGE:
                is_time_to_purge = frame % sim_config.config.PURGE_FRAME_FREQUENCY == 0
                is_too_many_creatures = \
                    len(self.creatures) > sim_config.config.MAX_NUM_CREATURES * sim_config.config.PURGE_POP_PERCENTAGE
                if is_time_to_purge or is_too_many_creatures:
                    self.do_purge = True

            if sim_config.config.DEBUG_MODE:
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
                sizes = np.array(
                    [creature.mass for creature in self.creatures.values()]) * sim_config.config.FOOD_SIZE  # / 10
                colors = [creature.color for creature in self.creatures.values()]
                edge_colors = ['r' if creature.digest_dict['creature'] > 0 else 'g' for creature in
                               self.creatures.values()]

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

                trait_stacked_colored_histogram(
                    ax=axes[0],
                    creatures=self.creatures,
                    trait_name='mass',
                    num_bins=30, min_value=0, max_value=2)

                # ----------------- update traits scat -------------------
                traits_scat = plot_traits_scatter(ax=axes[3],
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

            except Exception as e:
                # breakpoint('Error in simulation (update_func): cannot plot')
                print(f'Error in simulation (update_func): cannot plot because {e}.')
                # breakpoint()

            return scat, quiv, grass_scat, leaves_scat, agent_scat, traits_scat

        # Run simulation
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

            # Save statistics logs to json file
            self.statistics_logs.to_json(filepath=sim_config.config.STATISTICS_LOGS_JSON_FILEPATH)

            simulation_utils.copy_config_and_physical_model_to_output_folder(
                physical_model_full_path=self.physical_model_path)

    def use_brain(self, creature: Creature,
                  seek_result: dict, dt: float):
        brain_input = simulation_utils.get_brain_input(creature=creature, seek_result=seek_result)
        decision = creature.think(brain_input)
        self.physical_model.move_creature(creature=creature, decision=decision, dt=dt)

        # Collision detection
        simulation_utils.detect_collision(creature=creature, env=self.env)

    def eat_food(self,
                 creature: Creature,
                 seek_result: dict,
                 creatures_ids_to_kill: list[int],
                 step_counter: int):
        """

        :param creature:
        :param seek_result: dict of '{channel}_{eye_idx}': [distance, angle, idx]
        :param creatures_ids_to_kill
        :param step_counter:
        :return: eaten_food_type: 'grass'/'leaf'/'creature' or None if no food was eaten
        """
        # check if creature is full
        if creature.energy >= creature.max_energy:
            return None, None

        # init relevant variables
        food_energy = 0
        food_point = None
        eaten_food_type = None
        is_food_condition_met = False
        food_list, food_to_remove_list = [], []

        # Eat food if conditions are met
        for key, value in seek_result.items():
            food_type = key.split('_')[0]

            # Check if eye found something and if creature can eat it
            if value is None or creature.digest_dict[food_type] == 0:
                continue
            else:
                food_distance, food_angle, food_idx = seek_result[key]

                if food_type == 'grass':
                    food_list = self.env.grass_points
                    food_to_remove_list = self.env.grass_remove_list
                    food_energy = sim_config.config.GRASS_ENERGY
                    is_food_condition_met = True
                elif food_type == 'leaf':
                    food_list = self.env.leaf_points
                    food_to_remove_list = self.env.leaf_remove_list
                    food_energy = sim_config.config.LEAF_ENERGY
                    is_food_condition_met = creature.height >= food_list[food_idx].height
                elif food_type == 'creature':
                    food_list = [creature_id for creature_id in self.creatures.keys()]
                    food_to_remove_list = creatures_ids_to_kill
                    prey_id = food_list[food_idx]
                    prey = self.creatures[prey_id]
                    food_energy = prey.energy + self.physical_model.energy_conversion_factors['mass_energy'] * prey.mass

                    is_child = creature.creature_id == prey.parent_id
                    is_father = creature.parent_id == prey.creature_id
                    is_food_condition_met = creature.mass >= prey.mass and not is_child and not is_father

                # Check if conditions to eat food are met (if so, eat and add to food_remove_list)
                food_point = food_list[food_idx]
                is_food_available = food_point not in food_to_remove_list
                is_food_close_enough = food_distance <= sim_config.config.FOOD_DISTANCE_THRESHOLD

                if is_food_available and is_food_close_enough and is_food_condition_met:
                    # eat food and record it
                    self.physical_model.digest_food(
                        creature=creature, food_type=food_type, food_energy=food_energy)
                    creature.log.add_record(f'eat_{food_type}', step_counter)

                    # remove food
                    food_to_remove_list.append(food_point)
                    eaten_food_type = food_type
                    break

        return eaten_food_type, food_point
