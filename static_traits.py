# static_traits.py
import numpy as np
from brain import Brain


class StaticTraits:
    """
    Contains constant traits for a creature (e.g., maximum size, vision parameters).
    """

    def __init__(self, creature_id: int, gen: int, parent_id: str | None, birth_frame: int,
                 max_age: int, max_weight: float, max_height: float,
                 max_speed: list[float], max_energy: float, color: np.ndarray,
                 energy_efficiency: float, motion_efficiency: float,
                 food_efficiency: float, reproduction_energy: float,
                 eyes_params: list[tuple], vision_limit: float, brain: Brain):
        """
        eyes_params is a list of tuples: (angle_offset, aperture)
        where angle_offset (in radians) is relative to the creature's heading.
        """
        self.creature_id = creature_id
        self.gen = gen
        self.parent_id = parent_id
        self.birth_frame = birth_frame
        self.max_age = max_age
        self.max_weight = max_weight
        self.max_height = max_height
        self.max_speed = max_speed
        self.max_energy = max_energy
        self.color = color      # Creature's color as an RGB vector.

        self.energy_efficiency = energy_efficiency
        self.motion_efficiency = motion_efficiency  # number between 0 and 1
        self.food_efficiency = food_efficiency  # number between 0 and 1
        self.reproduction_energy = reproduction_energy

        self.eyes_params = eyes_params
        self.vision_limit = vision_limit
        self.brain = brain


    def think(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Processes the input vector through the brain to generate a decision.
        """
        return self.brain.forward(input_vector)
