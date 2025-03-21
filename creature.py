import copy

import numpy as np
from static_traits import StaticTraits
from brain_models.fully_connected_brain import Brain
from config import Config as config


class Creature(StaticTraits):
    """
    A dynamic creature in the simulation.
    Inherits static traits and adds dynamic properties such as position, speed, hunger, etc.
    """

    def __init__(self, creature_id: int, gen: int, parent_id: str | None, birth_frame: int,
                 max_age: int, max_weight: float, max_height: float,
                 max_speed: float, max_energy: float, color: np.ndarray,
                 energy_efficiency: float, motion_efficiency: float,
                 food_efficiency: float, reproduction_energy: float,
                 eyes_params: list[tuple], vision_limit: float, brain: Brain,
                 position: np.ndarray):
        super().__init__(creature_id=creature_id, gen=gen, parent_id=parent_id, birth_frame=birth_frame,
                         max_age=max_age, max_weight=max_weight, max_height=max_height,
                         max_speed=max_speed, max_energy=max_energy, color=color,
                         energy_efficiency=energy_efficiency, motion_efficiency=motion_efficiency,
                         food_efficiency=food_efficiency, reproduction_energy=reproduction_energy,
                         eyes_params=eyes_params, vision_limit=vision_limit, brain=brain)

        self.age = 0
        self.position = position
        self.max_speed_exp = 0
        self.init_state()  # init height, weight, velocity, energy, hunger and thirst
        self.calc_speed()
        self.ancestors = []  # list of creature ids that are ancestors

        # logs for debugging
        self.log_eat = []
        self.log_reproduce = []
        self.log_energy = []

    def init_state(self):
        """
        :return:
        """
        # dynamic traits
        self.weight = np.random.randint(low=self.max_weight * 0.01, high=self.max_weight * 0.1)
        self.height = np.random.randint(low=self.max_height * 0.01, high=self.max_height * 0.1)
        self.velocity = (np.random.rand(2) - 0.5) * self.max_speed
        self.energy = int(self.reproduction_energy * 0.95)
        self.hunger = 100
        self.thirst = 100

    def plot_live_status(self, ax, debug=False):
        """
        Plots the agent's status (energy, hunger, thirst) on the given axes.
        """
        if debug:
            import matplotlib.pyplot as plt
            plt.ion()
            fig, ax = plt.subplots(1, 1)
        # Define attributes dynamically
        ls = ['energy', 'hunger', 'thirst', 'age']
        colors = ['green', 'red', 'blue', 'grey']
        values = [getattr(self, attr) for attr in ls]  # Dynamically get values
        ax.clear()
        ax.set_title(f'Agent # {self.creature_id} Status | ancestors = {self.ancestors}')
        ax.barh(ls, values, color=colors)
        # ax.barh(['Energy', 'Hunger', 'Thirst'], [self.energy, self.hunger, self.thirst], color=['green', 'red', 'blue'])
        ax.set_xlim(0, max(config.REPRODUCTION_ENERGY, self.max_age))
        # ax.set_xticks([0,self.max_energy/2, self.max_energy])
        ax.set_yticks(ls)
        if 'energy' in ls:
            ax.scatter([config.REPRODUCTION_ENERGY], ['energy'], color='black', s=20)
        if 'age' in ls:
            ax.scatter([self.max_age], ['age'], color='black', s=20)

    def plot_acc_status(self, ax, debug=False, plot_type=1, curr_frame=-1):
        """
        Plots the agent's accumulated status (logs) on the given axes.
        """
        if debug:
            print('debug_mode')
            import matplotlib.pyplot as plt
            plt.ion()
            fig, ax = plt.subplots(1, 1)
        # Define attributes dynamically
        ls = ['log_eat', 'log_reproduce']
        colors = ['green', 'pink']
        ax.clear()
        if max(self.color) > 1 or min(self.color) < 0:
            raise ('color exceed [0, 1] range')
        ax.set_facecolor(list(self.color) + [0.3])
        if plot_type == 0:
            # option 1
            values = [len(getattr(self, attr)) for attr in ls]  # Dynamically get values
            ax.set_title(f'Agent # {self.id} Accumulated Status')
            ax.bar(ls, values, color=colors, width=0.2)
            ax.set_ylim(0, 10)
            ax.set_yticks([0, 5, 10, 100])
            ax.set_xticks(ls)
        if plot_type == 1:
            # option 2
            if curr_frame == -1: curr_frame = self.max_age + self.birth_frame
            # values = [getattr(self, attr) for attr in ls]  # Dynamically get values
            eating_frames = self.log_eat
            reproducing_frames = self.log_reproduce
            ax.scatter(eating_frames, [1] * len(eating_frames), color='green', marker='o', s=100, label='Eating')
            ax.scatter(reproducing_frames, [2] * len(reproducing_frames), color='red', marker='D', s=100,
                       label='Reproducing')
            ax.set_yticks([1, 2])
            ax.set_yticklabels(['Eating', 'Reproducing'])
            # Label x-axis and add a title
            ax.set_xlabel('Frame Number')
            ax.set_title('Event Timeline: Eating & Reproducing')
            ax.set_xlim([self.birth_frame - 1, curr_frame + 1])
            ax.set_ylim([0.5, 2.5])
            ax.legend()

    def get_heading(self):
        """
        Returns the creature's current heading (unit vector).
        If the creature is stationary, defaults to (1, 0).
        """
        if hasattr(self, 'velocity') and self.speed > 1e-3:
            return self.velocity / self.speed
        else:
            return np.array([1.0, 0.0])

    def calc_speed(self):
        """
        Returns the creature's current speed vector.
        """
        self.speed = np.linalg.norm(self.velocity)
        # self.max_speed_exp = max(self.max_speed_exp, self.speed)
        self.max_speed_exp = (self.max_speed_exp + self.speed) / 2

    def reproduce(self):
        """
        Returns a new creature with mutated traits.
        """
        # Mutate traits
        child = copy.deepcopy(self)
        child.mutate(config.MAX_MUTATION_FACTORS)
        child.brain.mutate(config.MUTATION_BRAIN)
        child.reset()
        child.ancestors.append(self.creature_id)
        # Reduce energy
        self.energy -= self.reproduction_energy
        return child

    def reset(self):
        """
        Reset the creature to initial state and flip velocity.
        """
        self.gen += 1
        self.parent_id = self.creature_id
        self.age = 0
        self.velocity = -self.velocity
        self.creature_id = 0
        self.log_energy = []
        self.log_eat = []
        self.log_reproduce = []
        self.max_speed_exp = 0
        self.init_state()

    def mutate(self, max_mutation_factors, mutation_chance=config.MUTATION_CHANCE):
        """
        mutate the desired traits.
        """
        for key in max_mutation_factors:
            if np.random.rand(1) < config.MUTATION_CHANCE:
                if key == 'eyes_params':
                    # Mutate eyes_params
                    for i in range(len(self.eyes_params)):
                        for j in range(2):
                            mutation_factor = np.random.uniform(-max_mutation_factors[key], max_mutation_factors[key])
                            self.eyes_params[i][j] = self.eyes_params[i][j] + mutation_factor
                else:
                    # Mutate trait
                    if np.random.rand(1) < config.MUTATION_CHANCE:
                        mutation_factor = np.random.uniform(-max_mutation_factors[key], max_mutation_factors[key])
                        setattr(self, key, getattr(self, key) + mutation_factor)
        self.color = np.clip(self.color, 0, 1)

    def eat(self, food_energy):
        self.energy += self.food_efficiency * food_energy
        self.height += (self.max_height - self.height) * food_energy / config.GROWTH_RATE
        self.weight += (self.max_weight - self.weight) * food_energy / config.GROWTH_RATE

    def consume_energy(self):
        energy_consumption = 0
        energy_consumption += self.energy_efficiency  # idle energy
        energy_consumption += self.motion_efficiency * self.speed  # movement energy
        self.energy -= energy_consumption
