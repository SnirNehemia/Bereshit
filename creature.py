# creature.py
import numpy as np
from static_traits import StaticTraits
from brain import Brain

class Creature(StaticTraits):
    """
    A dynamic creature in the simulation.
    Inherits static traits and adds dynamic properties such as position, speed, hunger, etc.
    """
    def __init__(self, position: np.ndarray, max_size: float, max_speed: float,
                 eyes_params: list,
                 vision_limit: float, brain: Brain, speed: np.ndarray,
                 hunger: float, thirst: float, color: np.ndarray):
        super().__init__(position, max_size, max_speed,
                         eyes_params, vision_limit, brain)
        self.speed = speed      # 2D velocity vector.
        self.hunger = hunger
        self.thirst = thirst
        self.color = color      # Creature's color as an RGB vector.
