# creature.py
import numpy as np
from static_traits import StaticTraits
from brain import Brain


class Creature(StaticTraits):
    """
    A dynamic creature in the simulation.
    Inherits static traits and adds dynamic properties such as position, speed, hunger, etc.
    """

    def __init__(self, max_age: int, max_weight: float, max_height: float, max_speed: list[float], color: np.ndarray,
                 energy_efficiency: float, speed_efficiency: float,
                 food_efficiency: float, reproduction_energy: float,
                    max_energy: float,
                 eyes_params: list[tuple], vision_limit: float, brain: Brain,
                 weight: float, height: float,
                 position: np.ndarray, speed: np.ndarray,
                 energy: float, hunger: float, thirst: float):
        super().__init__(max_age=max_age, max_weight=max_weight, max_height=max_height, max_speed=max_speed,
                         color=color,
                         energy_efficiency=energy_efficiency, speed_efficiency=speed_efficiency,
                         food_efficiency=food_efficiency, reproduction_energy=reproduction_energy,
                         max_energy=max_energy,
                         eyes_params=eyes_params, vision_limit=vision_limit, brain=brain)

        self.age = 0
        self.weight = weight
        self.height = height
        self.position = position
        self.speed = speed  # 2D velocity vector.
        self.energy = energy
        self.hunger = hunger
        self.thirst = thirst

        # logs for debugging
        self.log_eat = []
        self.log_reproduce = []
        self.log_energy = []

