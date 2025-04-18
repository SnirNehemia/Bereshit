import numpy as np
from scipy.spatial import KDTree

from creature import Creature
from environment import Environment
import simulation_utils
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
        self.creatures = simulation_utils.initialize_creatures(num_creatures=config.NUM_CREATURES,
                                                               simulation_space=self.env.size,
                                                               input_size=config.INPUT_SIZE,
                                                               output_size=config.OUTPUT_SIZE,
                                                               eyes_params=config.EYES_PARAMS,
                                                               env=self.env)
        self.dead_creatures = dict()
        self.positions = []

        # Build a KDTree for creature positions.
        self.creatures_kd_tree = simulation_utils.build_creatures_kd_tree(creatures=self.creatures)
        self.children_num = 0

        # simulation control parameters
        self.abort_simulation = False
        self.kdtree_update_interval = config.UPDATE_KDTREE_INTERVAL
        self.num_steps_per_frame = 0
        self.step_counter = 0  # Initialize step counter
        self.id_count = config.NUM_CREATURES - 1
        self.focus_id = 0
        self.make_agent(focus_id=0)

        self.purge = True  # flag for purge events

        # statistics logs
        self.statistics_logs = StatisticsLogs()

        if config.DEBUG_MODE:
            np.seterr(all='raise')  # Convert NumPy warnings into exceptions

        print(f'Num frames = {config.NUM_FRAMES},\n'
              f'Num steps = {simulation_utils.calc_total_num_steps(config.NUM_FRAMES)}')

    def make_agent(self, focus_id: int = 0):
        self.focus_id = focus_id
        self.creatures[self.focus_id].make_agent()

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

        # ------------------ seek creatures' targets and use brain ------------------------------
        seek_results = {}
        for creature_id, creature in self.creatures.items():
            seek_results[creature_id] = simulation_utils.seek(creatures=self.creatures,
                                                              creatures_kd_tree=self.creatures_kd_tree,
                                                              env=self.env,
                                                              creature=creature, noise_std=noise_std)
            # -------------------- Use brain and update creature velocities --------------------------

        for creature_id, creature in self.creatures.items():
            simulation_utils.use_brain(creature=creature,
                                       env=self.env,
                                       seek_results=seek_results[creature_id],
                                       dt=dt)

        # ----------------- die / eat / reproduce (+create list of creatures to die or reproduce) -----------------

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
                is_eat = simulation_utils.eat_food(creature=creature, env=self.env,
                                                   seek_result=seek_results[creature_id],
                                                   food_type='grass', step_counter=self.step_counter)
                if not is_eat and creature.height >= config.LEAF_HEIGHT:
                    _ = simulation_utils.eat_food(creature=creature, env=self.env,
                                                  seek_result=seek_results[creature_id],
                                                  food_type='leaf', step_counter=self.step_counter)

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
            self.creatures_kd_tree = simulation_utils.update_creatures_kd_tree(creatures=self.creatures)
            self.env.update_grass_kd_tree()

        self.step_counter += 1

        return child_ids, dead_ids

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

            # check num steps per frame
            self.num_steps_per_frame = simulation_utils.calc_num_steps_per_frame(frame=frame)

            # abort simulation if no creatures left or there are too many creatures
            if self.abort_simulation:
                if config.DEBUG_MODE:
                    from matplotlib import use

                    use('TkAgg')
                    self.statistics_logs.plot_and_save_statistics_graphs(to_save=False)
                    # breakpoint()

                ax_env.set_title(f"Evolution Simulation ({frame=}, step={self.step_counter})")
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
                                                            child_ids=child_ids, dead_ids=dead_ids)

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
