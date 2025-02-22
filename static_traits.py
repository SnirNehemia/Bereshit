# static_traits.py
import numpy as np
from brain import Brain


class StaticTraits:
    """
    Contains constant traits for a creature (e.g., maximum size, vision parameters).
    """

    def __init__(self, position: np.ndarray, max_size: float, max_speed: float,
                 left_eye_params: tuple, right_eye_params: tuple,
                 vision_limit: float, brain: Brain):
        """
        left_eye_params and right_eye_params are tuples: (angle_offset, aperture)
        where angle_offset (in radians) is relative to the creature's heading.
        """
        self.position = position
        self.max_size = max_size
        self.max_speed = max_speed
        self.left_eye_params = left_eye_params
        self.right_eye_params = right_eye_params
        self.vision_limit = vision_limit
        self.brain = brain

    def get_heading(self):
        """
        Returns the creature's current heading (unit vector).
        If the creature is stationary, defaults to (1, 0).
        """
        if hasattr(self, 'speed') and np.linalg.norm(self.speed) > 0:
            return self.speed / np.linalg.norm(self.speed)
        else:
            return np.array([1.0, 0.0])

    def think(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Processes the input vector through the brain to generate a decision.
        """
        return self.brain.forward(input_vector)
