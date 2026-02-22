from abc import ABC, abstractmethod


class PhysicalModel(ABC):
    """Abstract base class for all physical models."""

    def __init__(self):
        pass

    @abstractmethod
    def move_creature(self, creature, decision, dt, **kwargs):
        """
        Update creature position, velocity and energy given decision (brain output).
        :param creature: Creature
        :param decision: brain output, 2 X 1 vector (magnitude, direction)
        :param dt: float
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def digest_food(self, creature, food_type, food_energy, **kwargs):
        """
        Update creature energy based on given food.
        :param creature: Creature
        :param food_type: str
        :param food_energy: float
        :param kwargs:
        :return:
        """
        pass
