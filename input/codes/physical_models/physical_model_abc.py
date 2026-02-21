from abc import ABC, abstractmethod

from creature import Creature


class PhysicalModel(ABC):
    """Abstract base class for all physical models."""

    def __init__(self):
        pass

    @abstractmethod
    def move_creature(self, creature: Creature, decision, dt, **kwargs):
        """
        Update creature position, velocity and energy given decision (brain output).
        :param creature: Creature
        :param decision: brain output, 2 X 1 vector (magnitude, direction)
        :param dt: float
        :param kwargs:
        :return:
        """
        # update position, velocity and speed
        creature.velocity += decision
        creature.position += creature.velocity * dt
        creature.calc_speed()

        # update energy
        propulsion_energy = self.energy_conversion_factors['activity_efficiency'] * decision[0]
        creature.energy -= propulsion_energy

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
        creature.energy += creature.digest_dict[food_type] * food_energy

