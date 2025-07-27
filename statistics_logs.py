import numpy as np
from matplotlib import pyplot as plt
from input.codes.config import config
from creature import Creature
from environment import Environment
from json_utils import Serializable


class StatisticsLogs(Serializable):
    def __init__(self):
        self.num_frames = 0
        self.total_num_steps = 0
        self.num_creatures_per_step = []
        self.num_new_creatures_per_step = []
        self.num_dead_creatures_per_step = []
        self.num_herbivores_per_step = []
        self.num_carnivores_per_step = []
        self.creatures_id_history = []
        self.num_grass_history = []
        self.num_leaves_history = []
        self.log_eats_dict = {food_type: list()
                              for food_type in config.INIT_HERBIVORE_DIGEST_DICT.keys()}
        self.death_causes_dict = {'age': [], 'fatigue': [], 'eaten': [], 'purge': []}

        # statistics log parameters
        self.traits_stat_names = ['energy', 'speed']
        for stat_name in self.stat_dict.keys():
            for trait_stat_name in self.traits_stat_names:
                setattr(self, f'{stat_name}_creature_{trait_stat_name}_per_step', [])

    @property
    def stat_dict(self):
        return {
            'min': np.min,
            'max': np.max,
            'mean': np.mean,
            'std': np.std
        }

    def update_statistics_logs(self, creatures: dict[int, Creature], env: Environment,
                               child_ids: list, dead_ids: list, step_counter: int):

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

            self.num_herbivores_per_step.append(np.sum([1 for creature in creatures.values()
                                                        if creature.digest_dict['grass'] > 0
                                                        or creature.digest_dict['leaf'] > 0]))
            self.num_carnivores_per_step.append(np.sum([1 for creature in creatures.values()
                                                        if creature.digest_dict['creature'] > 0]))

            # update energy and speed statistics
            self.update_trait_statistics_logs(creatures=creatures, trait='energy')
            self.update_trait_statistics_logs(creatures=creatures, trait='speed')

            # Update eating logs
            for food_type in config.INIT_HERBIVORE_DIGEST_DICT.keys():
                food_type_key = f'eat_{food_type}'
                creature_ids_that_ate = [creature_id
                                         for creature_id, creature in creatures.items()
                                         if food_type_key in creature.log.record.keys() and
                                         creature.log.record[food_type_key][-1] == step_counter - 1]
                self.log_eats_dict[food_type].append(creature_ids_that_ate)

                # Print who ate
                # if len(creature_ids_that_ate) > 0:
                #     print(f'Creatures {creature_ids_that_ate} ate {food_type}')

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
        self.plot_env_graph(to_save=to_save)

    def plot_env_graph(self, to_save: bool = False):
        """
        Plot and save env statistics summary graphs
        :param to_save:
        :return:
        """
        env_fig, ax = plt.subplots(1, 1)
        ax.plot(self.num_grass_history, 'g.-', label='num grass')
        ax.plot(self.num_leaves_history, 'k.-', label='num leaves')
        ax.legend()
        env_fig.show()

        if to_save:
            env_fig.savefig(fname=config.ENV_FIG_FILE_PATH)

    def get_creatures_lifespan_vs_step_matrix(self):
        num_steps = len(self.creatures_id_history)
        max_id = np.max([np.max(creatures_id_in_step_i) + 1
                         for creatures_id_in_step_i in self.creatures_id_history])

        creatures_lifespan_vs_step = np.zeros((num_steps, max_id))
        for i_step, creatures_id_in_step_i in enumerate(self.creatures_id_history):
            for creature_id in creatures_id_in_step_i:
                creatures_lifespan_vs_step[i_step][creature_id] = 1

        return creatures_lifespan_vs_step

    def plot_creatures_lifespan_vs_step_matrix(self, to_show: bool = False):
        """
        Plot an image of [num_steps, ids] showing creatures lifespan vs step number.
        :return:
        """
        # get_creatures_lifespan_vs_step_matrix
        creatures_lifespan_vs_step_matrix = self.get_creatures_lifespan_vs_step_matrix()

        # plot_creatures_lifespan_vs_step_matrix
        plt.figure()
        plt.imshow(creatures_lifespan_vs_step_matrix.T, cmap='gray_r',
                   aspect='auto', origin='lower', interpolation='none')
        plt.title(f'alive creature ids vs. step number')
        plt.xlabel('step number')
        plt.ylabel('creature ids')
        plt.grid(which='both')

        if to_show:
            plt.show()

    def get_creatures_nutrition_vs_step_matrix(self):
        """

        :return: creatures_nutrition_vs_step_matrix: where 1 is grass, 2 is leaf and 3 is creature
        """
        num_steps = len(self.creatures_id_history)
        max_id = np.max([np.max(creatures_id_in_step_i) + 1
                         for creatures_id_in_step_i in self.creatures_id_history])
        creatures_nutrition_vs_step_matrix = np.zeros((num_steps, max_id))
        for food_idx, food_type in enumerate(self.log_eats_dict.keys()):
            for i_step, creatures_id_in_step_i in enumerate(self.log_eats_dict[food_type]):
                for creature_id in creatures_id_in_step_i:
                    creatures_nutrition_vs_step_matrix[i_step][creature_id] = food_idx + 1

        return creatures_nutrition_vs_step_matrix

    def plot_creatures_nutrition_vs_step_matrix(self, to_show: bool = False):
        """
        Plot an image of [num_steps, ids] showing which creatures ate in each step.
        :return:
        """

        # get_creatures_nutrition_vs_step_matrix
        creatures_nutrition_vs_step_matrix = self.get_creatures_nutrition_vs_step_matrix()

        # plot_creatures_nutrition_vs_step_matrix
        # plt.figure()
        food_type_colors = {'grass': 'lightgreen', 'leaf': 'darkgreen', 'creature': 'red'}
        food_type_markers = {'grass': 'o', 'leaf': 's', 'creature': 'D'}
        for food_idx, food_type in enumerate(self.log_eats_dict.keys()):
            rows, cols = np.where(creatures_nutrition_vs_step_matrix == food_idx + 1)
            plt.scatter(rows, cols, label=food_type,
                        s=10, marker=food_type_markers[food_type],
                        c=food_type_colors[food_type], edgecolors='white', linewidths=0.5)
        plt.title(f'Eating of creature ids vs. step number')
        plt.xlabel('step number')
        plt.ylabel('creature ids')
        plt.grid(which='both')
        plt.legend()
        if to_show:
            plt.show()

    def plot_creatures_lifespan_graphs(self, to_show: bool = False):
        # get_creatures_lifespan_vs_step_matrix
        creatures_lifespan_vs_step_matrix = self.get_creatures_lifespan_vs_step_matrix()
        creatures_lifespan_per_creature = np.sum(creatures_lifespan_vs_step_matrix >= 1, axis=0)  # sum steps
        creatures_lifespan_per_step = np.sum(creatures_lifespan_vs_step_matrix >= 1, axis=1)  # sum creatures

        # lifespan statistics graphs
        fig, ax = plt.subplots(2, 1)
        # ax[0].plot(creatures_lifespan_per_creature)
        # ax[0].set_title('How many steps each creature live')
        # ax[0].set_xlabel('creature id')
        ax[0].hist(creatures_lifespan_per_creature)
        ax[0].set_title('How many steps each creature live - histogram')
        ax[0].set_xlabel('step number')
        ax[1].plot(creatures_lifespan_per_step)
        ax[1].set_title('How many creatures live in each step')
        ax[1].set_xlabel('step number')
        ax[1].minorticks_on()
        ax[1].grid(which='major', color='gray', linestyle='-', linewidth=0.8)
        ax[1].grid(which='minor', color='lightgray', linestyle=':', linewidth=0.8)
        if to_show:
            plt.show()

    def plot_creatures_nutrition_graphs(self, to_show: bool = False):
        # get_creatures_nutrition_vs_step_matrix
        creatures_nutrition_vs_step_matrix = self.get_creatures_nutrition_vs_step_matrix()
        food_types = list(self.log_eats_dict.keys())

        num_eats_per_food_type = [np.sum(creatures_nutrition_vs_step_matrix == food_idx + 1)
                                  for food_idx in range(len(food_types))]

        creatures_diet = []
        num_creatures_that_eat_food_type = np.zeros(4)
        max_id = creatures_nutrition_vs_step_matrix.shape[1]
        for creature_id in range(max_id):
            creature_nutrition_vs_step = creatures_nutrition_vs_step_matrix[:, creature_id]

            has_eat = False
            for food_idx, food_type in enumerate(food_types):
                if food_idx + 1 in creature_nutrition_vs_step:
                    num_creatures_that_eat_food_type[food_idx] += 1
                    is_carnivore = int(food_idx + 1 == 3)
                    creatures_diet.append(is_carnivore)  # Herbivore: 0, carnivore: 1
                    has_eat = True

            if not has_eat:
                num_creatures_that_eat_food_type[3] += 1
                creatures_diet.append(-1)

        creatures_diet = np.array(creatures_diet)
        # num eats/creatures per food type (bar plots)
        fig, ax = plt.subplots(2, 1)
        ax[0].bar(x=food_types, height=num_eats_per_food_type)
        ax[0].set_title('num_eats_per_food_type')
        ax[1].bar(x=food_types + ['didnt eat'], height=num_creatures_that_eat_food_type)
        ax[1].set_title('num_creatures_that_eat_food_type')

        # Eat statistics graphs
        num_eats_per_creature = np.sum(creatures_nutrition_vs_step_matrix >= 1, axis=0)  # sum steps
        num_eats_per_step = np.sum(creatures_nutrition_vs_step_matrix >= 1, axis=1)  # sum creatures

        fig, ax = plt.subplots(2, 1)
        creatures_ids = np.arange(0, max_id)
        is_didnt_eat = creatures_diet == -1
        is_herbivore = creatures_diet == 0
        is_carnivore = creatures_diet == 1
        ax[0].plot(creatures_ids[is_didnt_eat], num_eats_per_creature[is_didnt_eat],
                   'ko', label='didnt eat')  # , markersize=5)
        ax[0].plot(creatures_ids[is_herbivore], num_eats_per_creature[is_herbivore],
                   'go', label='herbivore')  # , markersize=5)
        ax[0].plot(creatures_ids[is_carnivore], num_eats_per_creature[is_carnivore],
                   'ro', label='carnivore')  # , markersize=5)
        ax[0].set_title('How many each creature ate in its life')
        ax[0].set_xlabel('creature id')
        ax[0].minorticks_on()
        ax[0].grid(which='major', color='gray', linestyle='-', linewidth=0.8)
        ax[0].grid(which='minor', color='lightgray', linestyle=':', linewidth=0.8)
        ax[0].legend()

        ax[1].plot(num_eats_per_step, 'o')  # , markersize=5)
        ax[1].set_title('How many creatures ate in each step')
        ax[1].set_xlabel('step number')
        ax[1].minorticks_on()
        ax[1].grid(which='major', color='gray', linestyle='-', linewidth=0.8)
        ax[1].grid(which='minor', color='lightgray', linestyle=':', linewidth=0.8)
        if to_show:
            plt.show()

    def plot_num_herbivores_vs_num_carnivores_per_step(self, to_show: bool = False):
        plt.figure()
        plt.plot(self.num_herbivores_per_step, label='num herbivores')
        plt.plot(self.num_carnivores_per_step, label='num carnivores')
        # plt.plot(statistics_logs.num_creatures_per_step, label='num creatures')
        # plt.plot(statistics_logs.num_new_creatures_per_step, label='num new childs')
        # plt.plot(statistics_logs.num_grass_history, label='num grass points')
        plt.xlabel('step number')
        plt.grid(which='both')
        plt.legend()
        if to_show:
            plt.show()
