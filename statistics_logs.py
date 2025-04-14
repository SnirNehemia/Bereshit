import numpy as np
from matplotlib import pyplot as plt
from config import Config as config
from creature import Creature
from environment import Environment


class StatisticsLogs:
    def __init__(self):
        self.num_creatures_per_step = []
        self.num_new_creatures_per_step = []
        self.num_dead_creatures_per_step = []
        self.creatures_id_history = []
        self.num_grass_history = []
        self.num_leaves_history = []
        self.log_num_eats = []

        # statistics log parameters
        self.stat_dict = {'min': np.min, 'max': np.max, 'mean': np.mean, 'std': np.std}
        self.traits_stat_names = ['energy', 'speed']
        for stat_name in self.stat_dict.keys():
            for trait_stat_name in self.traits_stat_names:
                setattr(self, f'{stat_name}_creature_{trait_stat_name}_per_step', [])

    def update_statistics_logs(self, creatures: dict[int, Creature], env: Environment,
                               child_ids, dead_ids):

        if len(creatures) > 0:
            # update environment log
            self.num_grass_history.append(len(env.grass_points))
            self.num_leaves_history.append(len(env.leaf_points))

            # update number of alive/new/dead creatures (and also id of all alive)
            self.num_creatures_per_step.append(len(creatures))
            self.num_new_creatures_per_step.append(len(child_ids))
            self.num_dead_creatures_per_step.append(len(dead_ids))
            self.creatures_id_history.append(
                [getattr(creature, 'creature_id') for creature in creatures.values()])

            # update energy and speed statistics
            self.update_trait_statistics_logs(creatures=creatures, trait='energy')
            self.update_trait_statistics_logs(creatures=creatures, trait='speed')

            # Update eating logs
            self.log_num_eats.append(np.sum([len(creature.log.record['eat']) for creature in creatures.values()]))

    def update_trait_statistics_logs(self, creatures: dict[int, Creature], trait: str):
        creatures_trait = [getattr(creature, trait) for creature in creatures.values()]

        for stat_name, stat_func in self.stat_dict.items():
            getattr(self, f'{stat_name}_creature_{trait}_per_step').append(stat_func(creatures_trait))

    def plot_creatures_statistics_graph(self, stat_trait: str):
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
        statistics_ax2.plot(num_dead_creatures_per_step, 'r.-', label='dead')
        statistics_ax2.plot(num_new_creatures_per_step, 'g.-', label='new')
        statistics_ax2.legend()

        # plot min/max/mean/std trait (speed or energy) of all creatures alive in every step
        statistics_ax[1].plot(getattr(self, f'min_creature_{stat_trait}_per_step'), '.-',
                              label=f'min {stat_trait}')
        statistics_ax[1].plot(getattr(self, f'max_creature_{stat_trait}_per_step'), '.-',
                              label=f'max {stat_trait}')
        statistics_ax[1].errorbar(x=np.arange(len(getattr(self, f'mean_creature_{stat_trait}_per_step'))),
                                  y=getattr(self, f'mean_creature_{stat_trait}_per_step'),
                                  yerr=getattr(self, f'std_creature_{stat_trait}_per_step'), linestyle='-',
                                  marker='.',
                                  label=f'mean and std {stat_trait}')
        if stat_trait == 'energy':
            statistics_ax[1].axhline(y=config.REPRODUCTION_ENERGY + config.MIN_LIFE_ENERGY,
                                     linestyle='--', color='r', label='reproduction threshold')
        statistics_ax[1].set_title(f'{stat_trait} statistics per step')
        statistics_ax[1].set_xlabel('step number')
        statistics_ax[1].legend()

        return statistics_fig

    def plot_and_save_statistics_graphs(self, to_save: bool = False):
        # Plot and save creature traits statistics summary graphs
        for stat_trait in self.traits_stat_names:
            statistics_fig = self.plot_creatures_statistics_graph(stat_trait=stat_trait)
            statistics_fig.show()

            if to_save:
                statistics_fig_filepath = config.OUTPUT_FOLDER.joinpath(
                    config.STATISTICS_FIG_FILEPATH.stem + f'_{stat_trait}.png')
                statistics_fig.savefig(fname=statistics_fig_filepath)
                print(f'statistics fig saved as {statistics_fig_filepath.stem}.')

        # Plot and save env statistics summary graphs
        env_fig, ax = plt.subplots(1, 1)
        ax.plot(self.num_grass_history, 'g.-', label='num grass')
        ax.plot(self.num_leaves_history, 'k.-', label='num leaves')
        ax.legend()
        env_fig.show()

        if to_save:
            env_fig.savefig(fname=config.ENV_FIG_FILE_PATH)
