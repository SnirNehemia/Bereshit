import numpy as np
from input.codes.config import config, load_config

import importlib

brain_module = importlib.import_module(f"brain_models.{config.BRAIN_TYPE}")
Brain = getattr(brain_module, 'Brain')

class StaticTraits:
    """
    Contains constant traits for a creature (e.g., maximum size, vision parameters).
    """

    def __init__(self, creature_id: int,
                 gen: int, parent_id: str | None, birth_step: int, color: np.ndarray,
                 max_age: int, max_mass: float, max_height: float, max_strength: float,
                 max_speed: float, max_energy: float,
                 digest_dict: dict, reproduction_energy: float,
                 eyes_params: list[tuple], vision_limit: float, brain: Brain):
        """
        eyes_params is a list of tuples: (angle_offset, aperture)
        where angle_offset (in radians) is relative to the creature's heading.
        """
        # for lineage tracking
        self.creature_id = creature_id
        self.gen = gen
        self.parent_id = parent_id
        self.birth_step = birth_step
        self.ancestors = []  # list of creature ids that are ancestors
        self.color = color  # Creature's color as an RGB vector.

        # for constraining dynamic traits
        self.max_age = max_age
        self.adulescence = self.max_age / 4
        self.max_mass = max_mass
        self.max_height = max_height
        self.max_strength = max_strength

        self.max_speed = max_speed  # currently used only at initialization
        self.max_energy = max_energy

        # eating food parameters
        self.digest_dict = digest_dict  # numbers between 0 and 1

        # reproduction energy
        self.reproduction_energy = reproduction_energy

        self.eyes_params = eyes_params
        self.vision_limit = vision_limit
        self.brain = brain

    def think(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Processes the input vector through the brain to generate a decision.
        """
        return self.brain.forward(input_vector)
