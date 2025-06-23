import numpy as np

from environment import Environment
import simulation_utils
from tqdm import tqdm
from input.codes.config import config
import plot_utils as plot

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec

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
        # Create the environment. Ensure that 'map.png' exists and follows the color conventions.
        self.env = Environment(map_filename=config.ENV_PATH,
                               grass_generation_rate=config.GRASS_GENERATION_RATE,
                               leaves_generation_rate=config.LEAVES_GENERATION_RATE)

        # Initialize creatures (ensuring they are not in forbidden areas)
        brain_module = importlib.import_module(f"brain_models.{config.BRAIN_TYPE}")
        brain_obj = getattr(brain_module, 'Brain')
        self.creatures = simulation_utils.initialize_creatures(env=self.env, brain_obj=brain_obj)
        self.dead_creatures = dict()
        self.positions = []

        # Build a KDTree for creature positions.
        self.creatures_kd_tree = simulation_utils.build_creatures_kd_tree(creatures=self.creatures)
        self.children_num = 0

        # simulation control parameters
        self.abort_simulation = False
        self.num_steps_per_frame = 0
        self.step_counter = 0  # Initialize step counter
        self.id_count = config.NUM_CREATURES - 1
        self.focus_id = 0
        self.make_agent(focus_id=0)
        config.OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)

        self.do_purge = True  # flag for purge events

        # statistics logs
        self.statistics_logs = StatisticsLogs()

        if config.DEBUG_MODE:
            np.seterr(all='raise')  # Convert NumPy warnings into exceptions

        self.statistics_logs.num_frames = config.NUM_FRAMES
        self.statistics_logs.total_num_steps = \
            simulation_utils.calc_total_num_steps(config.NUM_FRAMES)
        print(f'{config.OUTPUT_FOLDER.stem}: '
              f'num frames = {self.statistics_logs.num_frames}, '
              f'Num steps = {self.statistics_logs.total_num_steps}')

    def make_agent(self, focus_id: int = 0):
        self.focus_id = focus_id
        self.creatures[self.focus_id].make_agent()

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
                             for food_type in config.INIT_HERBIVORE_DIGEST_DICT.keys()}

        for creature_id, creature in self.creatures.items():
            # death from age/fatigue/eaten by another creature
            if creature.age >= creature.max_age or creature.energy <= 0:
                creatures_ids_to_kill.append(creature_id)

            if creature_id in creatures_ids_to_kill:
                continue
            else:
                # Creature get older
                creature.age += config.DT

                # Seek in environment
                seek_result = simulation_utils.seek(creatures=self.creatures,
                                                    creatures_kd_tree=self.creatures_kd_tree,
                                                    env=self.env,
                                                    creature=creature, noise_std=noise_std,
                                                    creatures_ids_to_kill=creatures_ids_to_kill)

                # Use brain to move
                simulation_utils.use_brain(creature=creature,
                                           env=self.env,
                                           seek_result=seek_result,
                                           dt=dt)

                # Check for nearby food
                eaten_food_type = simulation_utils.eat_food(
                    creature=creature,
                    env=self.env,
                    seek_result=seek_result,
                    creatures=self.creatures,
                    creatures_ids_to_kill=creatures_ids_to_kill,
                    step_counter=self.step_counter)

                # record eaten_food_type to update kd tree afterward
                if eaten_food_type is not None:
                    to_update_kd_tree[eaten_food_type] = True

                # reproduce if able
                energy_needed_to_reproduce = creature.reproduction_energy + config.MIN_LIFE_ENERGY
                if (creature.energy > energy_needed_to_reproduce and
                        creature.can_reproduce(self.step_counter)):
                    creatures_ids_to_reproduce.append(creature_id)

        # ---------------------------- After all creatures actions ----------------------------------
        # Purge
        self.do_purge = simulation_utils.do_purge(do_purge=self.do_purge,
                                                  creatures=self.creatures,
                                                  creatures_ids_to_kill=creatures_ids_to_kill,
                                                  creatures_ids_to_reproduce=creatures_ids_to_reproduce,
                                                  step_counter=self.step_counter)

        # kill creatures
        new_dead_ids = simulation_utils.kill_creatures(creatures_ids_to_kill=creatures_ids_to_kill,
                                                       creatures=self.creatures,
                                                       dead_creatures=self.dead_creatures)

        # Reproduction
        new_child_ids, self.children_num, self.id_count = \
            simulation_utils.reporduce_creatures(creatures_ids_to_reproduce=creatures_ids_to_reproduce,
                                                 creatures=self.creatures,
                                                 id_count=self.id_count,
                                                 children_num=self.children_num,
                                                 step_counter=self.step_counter)

        # Update creatures logs (after movement, eating and reproduction)
        simulation_utils.update_creatures_logs(creatures=self.creatures)

        # Update environment and KD trees
        is_time_to_update_kd_trees = self.step_counter % config.UPDATE_KDTREE_INTERVAL == 0
        if to_update_kd_tree['grass']:
            self.env.remove_grass_points()
            if not is_time_to_update_kd_trees:
                self.env.update_grass_kd_tree()

        if to_update_kd_tree['leaf']:
            pass

        if to_update_kd_tree['creature']:
            if not is_time_to_update_kd_trees:
                self.creatures_kd_tree = \
                    simulation_utils.update_creatures_kd_tree(creatures=self.creatures)

        # generate new food points in the environment and update KD tree if condition are met
        self.creatures_kd_tree = simulation_utils.update_environment_and_kd_trees(
            env=self.env,
            creatures=self.creatures,
            creatures_kd_tree=self.creatures_kd_tree,
            is_time_to_update_kd_trees=is_time_to_update_kd_trees)

        # Update step counter
        self.step_counter += 1

        return new_child_ids, new_dead_ids

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
        global fig, axes, progress_bar

        def init_fig():
            """
            Init the simulation figure.
            :return:
            """
            global lineage_graph, traits_scat, quiv, scat, grass_scat, leaves_scat, agent_scat
            global fig, axes, progress_bar

            # init fig with the grid layout with uneven ratios
            # TODO: fig, axes = set_animation_figure()
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
            sizes = np.array([creature.mass for creature in self.creatures.values()]) * config.FOOD_SIZE / 100
            scat = axes[1].scatter(self.positions[:, 0], self.positions[:, 1], c=colors, s=sizes,
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
                                  color=colors, scale=150, width=0.005)  # 'black'

            # Scatter food points for vegetation
            grass_scat = axes[1].scatter([], [], c='lightgreen', edgecolors='black', s=10)
            leaves_scat = axes[1].scatter([], [], c='darkgreen', edgecolors='black', s=10)
            agent_scat = axes[1].scatter(
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
            lineage_graph = axes[0].scatter([], [], c=[], s=50)

            # Init traits plot
            traits_scat = axes[3].scatter([], [], c=[], s=50)

            # Initialize the progress bar to print
            if config.STATUS_EVERY_STEP:
                update_num = simulation_utils.calc_total_num_steps(up_to_frame=config.NUM_FRAMES)
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
                if config.DEBUG_MODE:
                    from matplotlib import use

                    use('TkAgg')
                    self.statistics_logs.plot_and_save_statistics_graphs(to_save=False)
                    # breakpoint()

                axes[1].set_title(f"Evolution Simulation ({frame=}, step={self.step_counter})")
                progress_bar.update(self.num_steps_per_frame)
                self.step_counter += self.num_steps_per_frame

                return scat, quiv, grass_scat, leaves_scat, agent_scat, traits_scat

            # Run steps of frame
            for step in range(self.num_steps_per_frame):
                # Do simulation step
                child_ids, dead_ids = self.do_step(dt=config.DT, noise_std=config.NOISE_STD)

                # abort simulation if there are too many creatures or no creatures left
                self.abort_simulation = simulation_utils.check_abort_simulation(creatures=self.creatures,
                                                                                step_counter=self.step_counter)

                # Update statistics logs
                self.statistics_logs.update_statistics_logs(creatures=self.creatures, env=self.env,
                                                            child_ids=child_ids, dead_ids=dead_ids,
                                                            step_counter=self.step_counter)

                # Update the progress bar every step
                if config.STATUS_EVERY_STEP:
                    progress_bar.set_description(f"Alive: {len(self.creatures):4} | "
                                                 f"Children: {self.children_num:4} | "
                                                 f"Dead: {len(self.dead_creatures):4} | "
                                                 f"leaves: {len(self.env.leaf_points):3} | "
                                                 f"grass: {len(self.env.grass_points):3} | "
                                                 f"Progress")
                    progress_bar.update(1)  # or self.num_steps_per_frame outside the for loop

            # update the progress bar every frame
            if not config.STATUS_EVERY_STEP:
                progress_bar.set_description(f"Alive: {len(self.creatures):4} | "
                                             f"Children: {self.children_num:4} | "
                                             f"Dead: {len(self.dead_creatures):4} | "
                                             f"leaves: {len(self.env.leaf_points):3} | "
                                             f"grass: {len(self.env.grass_points):3} | "
                                             f"Progress")
                progress_bar.update(1)  # or self.num_steps_per_frame outside the for loop

            # Do purge if PURGE_FRAME_FREQUENCY frames passed (to clear static agents)
            if config.DO_PURGE:
                is_time_to_purge = frame % config.PURGE_FRAME_FREQUENCY == 0
                is_too_many_creatures = \
                    len(self.creatures) > config.MAX_NUM_CREATURES * config.PURGE_POP_PERCENTAGE
                if is_time_to_purge and is_too_many_creatures:
                    self.do_purge = True

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
                        scat = axes[1].scatter(self.positions[:, 0], self.positions[:, 1],
                                               c=colors, s=sizes)
                        quiv = axes[1].quiver(self.positions[:, 0], self.positions[:, 1], U, V,
                                              color=colors, scale=150, width=0.005)
                else:
                    # plot place holder
                    scat = axes[1].scatter([1], [1])
                    quiv = axes[1].quiver([1], [1], [1], [1])

                # Update vegetation scatter data
                # num_grass_points_in_last_frame = TODO
                num_grass_points_after_step = len(self.env.grass_points)
                if num_grass_points_after_step > 0:
                    # if num_grass_points_after_step == num_grass_points_in_last_frame:
                    #     grass_scat.set_offsets(np.array(self.env.grass_points))
                    # else:
                    grass_points = np.array(self.env.grass_points)
                    grass_scat = axes[1].scatter(grass_points[:, 0], grass_points[:, 1], c='lightgreen',
                                                 edgecolors='black',
                                                 s=10)

                # num_leaf_points_in_last_frame = TODO
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

                if len(self.creatures) > 0:
                    ids = self.statistics_logs.creatures_id_history[-1]
                    if self.focus_id not in ids:
                        if self.id_count in ids:  # trying to use the youngest creature as agent
                            self.make_agent(self.id_count)
                        else:
                            self.make_agent(np.random.choice(list(self.creatures.keys())))
                    agent = self.creatures[self.focus_id]
                    agent_scat.set_offsets([agent.position, agent.position])
                    agent.brain.plot(axes[2])
                    # ax_agent_info.clear()
                    # plot.plot_rebalance(axes[4][0][0], agent, mode='force', add_title=True, ax_secondary=None)  # ax_secondary=axes[4][0][1]
                    plot.plot_rebalance(axes[4][0], agent, mode='force', add_title=True)
                    plot.plot_rebalance(axes[4][1], agent, mode='friction angle', add_x_label=False)
                    plot.plot_rebalance(axes[4][2], agent, mode='speed')
                    plot.plot_rebalance(axes[4][3], agent, mode='energy_use', add_x_label=True)
                    # plot.plot_rebalance(axes[4][3], agent, mode='power', add_x_label=True)
                    plot.plot_live_status(axes[5][2], agent, plot_horizontal=True)
                    plot.plot_live_status_power(axes[5][1], agent, plot_horizontal=True)
                    plot.plot_acc_status(axes[5][0], agent, plot_type=1, curr_step=self.step_counter)

            except Exception as e:
                # breakpoint('Error in simulation (update_func): cannot plot')
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
