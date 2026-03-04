import numpy as np


class StaticTraits:
    """
    Contains constant traits for a creature (e.g., maximum size, vision parameters).
    """

    def __init__(self, creature_id: int,
                 gen: int, parent_id: str | None, birth_step: int, color: np.ndarray,
                 max_age: int, max_mass: float, max_height: float, max_strength: float,
                 max_speed: float, max_energy: float,
                 digest_dict: dict,
                 reproduction_cooldown: float, reproduction_energy: float,
                 eyes: list[list], vision_limit: float, brain):
        """
        eyes is a list of tuples: (angle_offset, aperture)
        where angle_offset (in radians) is relative to the creature's heading.
        """
        # for lineage tracking
        self.creature_id = creature_id
        self.gen = gen
        self.parent_id = parent_id
        self.birth_step = birth_step
        self.color = color  # Creature's color as an RGB vector.

        # for constraining dynamic traits
        self.max_age = max_age
        self.adolescence_age = 0
        self.max_mass = max_mass
        self.max_height = max_height
        self.max_strength = max_strength

        self.max_speed = max_speed  # currently used only at initialization
        self.max_energy = max_energy

        # eating food parameters
        self.digest_dict = digest_dict  # numbers between 0 and 1

        # reproduction energy
        self.reproduction_cooldown = reproduction_cooldown
        self.reproduction_energy = reproduction_energy

        # Eyes
        self.eyes = eyes
        self.eye_cos_offset = []  # used for faster calc
        self.eye_sin_offset = []  # used for faster calc
        self.eye_cos_half_aperture = []  # used for faster calc
        for i_eye in range(len(self.eyes)):
            angle_offset, eye_aperture = self.eyes[i_eye]
            self.eye_cos_offset.append(np.cos(np.radians(angle_offset)))
            self.eye_sin_offset.append(np.sin(np.radians(angle_offset)))
            self.eye_cos_half_aperture.append(np.cos(np.radians(eye_aperture) / 2.0))

        self.vision_limit = vision_limit
        self.vision_limit_sq = self.vision_limit ** 2  # for faster calc

        # Brain
        self.brain = brain

    def think(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Processes the input vector through the brain to generate a decision.
        decision = [propulsion_force_mag, relative_propulsion_force_angle].
        angle is relative to heading (== velocity direction)
        """
        return self.brain.forward(input_vector)
