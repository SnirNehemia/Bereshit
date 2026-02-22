import numpy as np

from input.codes.physical_models.physical_model_abc import PhysicalModel


class PhysicalModel1(PhysicalModel):
    def __init__(self, **params):
        # init config based on data from yaml
        super().__init__()

        for key, value in params.items():
            setattr(self, key, value)

    def move_creature(self, creature, decision, dt, **kwargs):
        """
        Update creature position, velocity and energy given decision (brain output).
        :param creature: Creature
        :param decision: brain output, 2 X 1 vector (magnitude, direction)
        :param dt: float
        :param kwargs:
        :return:
        """

        # update position, velocity and speed
        force = self._transform_propulsion_force(creature=creature, decision=decision)
        self._update_position_and_velocity(creature=creature, total_force=force, dt=dt)

        # update energy
        force_mag = decision[0]
        propulsion_energy = self.energy_conversion_factors['activity_efficiency'] * force_mag
        creature.energy -= propulsion_energy

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

    def _transform_propulsion_force(self, creature, decision):
        # Clip propulsion force based on strength
        propulsion_force_mag, relative_propulsion_force_angle = decision
        propulsion_force_mag = np.clip(propulsion_force_mag, 0, 1)  # * creature.strength

        # transform relative propulsion_force to global cartesian coordinates (x,y)
        current_direction = creature.get_heading()
        cos_angle = np.cos(relative_propulsion_force_angle)
        sin_angle = np.sin(relative_propulsion_force_angle)
        global_propulsion_force_direction = np.array([
            current_direction[0] * cos_angle - current_direction[1] * sin_angle,
            current_direction[0] * sin_angle + current_direction[1] * cos_angle
        ])
        global_propulsion_force = global_propulsion_force_direction * propulsion_force_mag

        return global_propulsion_force

    @staticmethod
    def _update_position_and_velocity(creature, total_force, dt,
                                      debug_position: bool = False):
        acceleration = total_force / creature.mass
        new_velocity = creature.velocity + acceleration * dt
        new_position = creature.position + new_velocity * dt

        if debug_position:
            print(f'\t\t{acceleration=}\n'
                  f'\t\t{creature.velocity=} --> {new_velocity=}\n'
                  f'\t\t{creature.position} --> {new_position}')

        # update position, velocity and speed
        creature.position = new_position
        creature.velocity = new_velocity
        creature.calc_speed()
