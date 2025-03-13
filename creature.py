# creature.py
import numpy as np
from static_traits import StaticTraits
from brain import Brain
import config as config

class Creature(StaticTraits):
    """
    A dynamic creature in the simulation.
    Inherits static traits and adds dynamic properties such as position, speed, hunger, etc.
    """

    def __init__(self,id: int, max_age: int, max_weight: float, max_height: float, max_speed: list[float],
                 color: np.ndarray,
                 energy_efficiency: float, motion_efficiency: float,
                 food_efficiency: float, reproduction_energy: float,
                    max_energy: float,
                 eyes_params: list[tuple], vision_limit: float, brain: Brain,
                 weight: float, height: float,
                 position: np.ndarray, velocity: np.ndarray,
                 energy: float, hunger: float, thirst: float):
        super().__init__(max_age=max_age, max_weight=max_weight, max_height=max_height, max_speed=max_speed,
                         color=color,
                         energy_efficiency=energy_efficiency, motion_efficiency=motion_efficiency,
                         food_efficiency=food_efficiency, reproduction_energy=reproduction_energy,
                         max_energy=max_energy,
                         eyes_params=eyes_params, vision_limit=vision_limit, brain=brain)
        self.id = id
        self.age = 0
        self.weight = weight
        self.height = height
        self.position = position
        self.velocity = velocity  # 2D velocity vector.
        self.calc_speed()
        self.energy = energy
        self.hunger = hunger
        self.thirst = thirst

        # logs for debugging
        self.log_eat = []
        self.log_reproduce = []
        self.log_energy = []


    def plot_live_status(self, ax, debug=False):
        """
        Plots the agent's status (energy, hunger, thirst) on the given axes.
        """
        if debug:
            import matplotlib.pyplot as plt
            plt.ion()
            fig, ax = plt.subplots(1,1)
        # Define attributes dynamically
        ls = ['energy', 'hunger', 'thirst', 'age']
        colors = ['green', 'red', 'blue', 'grey']
        values = [getattr(self, attr) for attr in ls]  # Dynamically get values
        ax.clear()
        ax.set_title(f'Agent # {self.id} Status')
        ax.barh(ls, values, color=colors)
        # ax.barh(['Energy', 'Hunger', 'Thirst'], [self.energy, self.hunger, self.thirst], color=['green', 'red', 'blue'])
        ax.set_xlim(0, max(config.REPRODUCTION_ENERGY, self.max_age))
        # ax.set_xticks([0,self.max_energy/2, self.max_energy])
        ax.set_yticks(ls)
        if 'energy' in ls:
            ax.scatter([config.REPRODUCTION_ENERGY], ['energy'], color='black', s=20)
        if 'age' in ls:
            ax.scatter([self.max_age], ['age'], color='black', s=20)


    def plot_acc_status(self, ax, debug=False):
        """
        Plots the agent's accumulated status (logs) on the given axes.
        """
        if debug:
            import matplotlib.pyplot as plt
            plt.ion()
            fig, ax = plt.subplots(1,1)
        # Define attributes dynamically
        ls = ['log_eat', 'log_reproduce']
        colors = ['green', 'pink']
        values = [sum(getattr(self, attr)) for attr in ls]  # Dynamically get values
        ax.clear()
        ax.set_title(f'Agent # {self.id} Accumulated Status')
        ax.bar(ls, values, color=colors)
        ax.set_ylim(0, 10)
        ax.set_yticks([0, 5, 10])
        ax.set_xticks(ls)


    def get_heading(self):
        """
        Returns the creature's current heading (unit vector).
        If the creature is stationary, defaults to (1, 0).
        """
        if hasattr(self, 'velocity') and self.speed > 0:
            return self.velocity / self.speed
        else:
            return np.array([1.0, 0.0])


    def calc_speed(self):
        """
        Returns the creature's current speed vector.
        """
        self.speed = np.linalg.norm(self.velocity)