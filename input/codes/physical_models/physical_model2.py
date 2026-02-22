import numpy as np

from input.codes.physical_models.physical_model_abc import PhysicalModel


class PhysicalModel2(PhysicalModel):
    def __init__(self, **params):
        # init config based on data from yaml
        super().__init__()

        for key, value in params.items():
            setattr(self, key, value)

        # make needed adjustments
        self.trait_energy_func = lambda factor, rate, age: factor * np.exp(-rate * age)

    def move_creature(self, creature, decision, dt,
                      debug_position: bool = False, debug_energy: bool = False,
                      debug_force: bool = False):
        # Calc total force and propulsion energy
        total_force, propulsion_energy = \
            self._calc_total_force_and_propulsion_energy(creature=creature, decision=decision,
                                                         debug_force=debug_force)

        # Update position and velocity
        self._update_position_and_velocity(creature=creature, total_force=total_force, dt=dt,
                                           debug_position=debug_position)

        # update energy
        self._update_energy(creature=creature, propulsion_energy=propulsion_energy, debug_energy=debug_energy)

    def digest_food(self, creature, food_type, food_energy,
                    rebalance: bool = False):
        gained_energy = creature.digest_dict[food_type] * food_energy

        if creature.age < creature.adolescence:
            creature.height, height_energy = self._convert_gained_energy_to_trait(
                trait_type='height_energy',
                old_trait=creature.height,
                gained_energy=gained_energy,
                age=creature.age,
            )
            creature.mass, mass_energy = self._convert_gained_energy_to_trait(
                trait_type='mass_energy',
                old_trait=creature.mass,
                gained_energy=gained_energy,
                age=creature.age,
            )
        else:
            mass_energy, height_energy = 0, 0

        excess_energy = gained_energy - height_energy - mass_energy
        creature.energy += excess_energy

        if rebalance:
            creature.log.add_record('energy_excess', excess_energy)
            creature.log.add_record('energy_gain', gained_energy)
            creature.log.add_record('energy_height', height_energy)
            creature.log.add_record('energy_mass', mass_energy)
        else:
            creature.log.add_record('energy_excess', excess_energy)

    def _transform_propulsion_force(self, creature, decision):
        # Clip propulsion force based on strength
        propulsion_force_mag, relative_propulsion_force_angle = decision
        propulsion_force_mag = np.clip(propulsion_force_mag, 0, 1) * creature.strength

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

    def _calc_gravity_and_normal_forces(self, creature):
        """
        # Calculate gravity and normal force.
        Right now (2D movement) normal force is only used for friction
        and not directly in equation of motion.
        :param mass: creature mass
        :return:
        """
        gravity_force = creature.mass * self.g
        normal_force = - gravity_force
        return gravity_force, normal_force

    def _calc_reaction_friction_force(self, normal_force, propulsion_force):
        normal_force_mag = np.linalg.norm(normal_force)
        propulsion_force_mag = np.linalg.norm(propulsion_force)
        if propulsion_force_mag > self.mu_static * normal_force_mag:
            propulsion_force_direction = propulsion_force / propulsion_force_mag
            reaction_friction_force = - self.mu_kinetic * normal_force_mag * \
                                      propulsion_force_direction
        else:
            reaction_friction_force = - propulsion_force
        return reaction_friction_force

    def _calc_drag_force(self, creature):
        drag_force = [0, 0]

        if creature.speed > 1e-3:
            current_direction = creature.velocity / creature.speed
            linear_drag_force = - self.gamma * creature.height ** 2 * creature.velocity
            quadratic_drag_force = - self.c_drag * creature.height ** 2 * creature.speed ** 2 * current_direction
            drag_force = linear_drag_force + quadratic_drag_force

        return drag_force

    def _calc_propulsion_energy(self, propulsion_force):
        eta = self.energy_conversion_factors['activity_efficiency']
        c_heat = self.energy_conversion_factors['heat_loss']
        propulsion_energy = (1 / eta + c_heat) * np.linalg.norm(propulsion_force)
        return propulsion_energy

    def _calc_inner_energy(self, creature):
        c_d = self.energy_conversion_factors['digest']
        c_h = self.energy_conversion_factors['height_energy']
        rest_energy = self.energy_conversion_factors['rest'] * creature.mass ** 0.75  # adds mass (BMR) energy
        inner_energy = rest_energy + c_d * np.sum(
            list(creature.digest_dict.values())) + c_h * creature.height  # adds height energy
        inner_energy = inner_energy + creature.brain.size * self.energy_conversion_factors['brain_consumption']
        return inner_energy

    def _calc_trait_energy(self, trait_type, gained_energy, age):
        trait_energy_params = self.trait_energy_params_dict[trait_type]
        factor = trait_energy_params['factor']
        rate = trait_energy_params['rate']
        trait_energy_func = self.trait_energy_func(factor=factor, rate=rate, age=age)
        trait_energy = trait_energy_func * gained_energy
        return trait_energy

    def _convert_gained_energy_to_trait(self, trait_type: str, old_trait: float,
                                        gained_energy: float, age: float,
                                        ):
        trait_energy = self._calc_trait_energy(trait_type=trait_type, gained_energy=gained_energy, age=age)
        c_trait = self.energy_conversion_factors[trait_type]
        new_trait = old_trait + trait_energy / c_trait
        return new_trait, trait_energy

    def _calc_total_force_and_propulsion_energy(self, creature, decision,
                                                debug_force: bool = False):
        # Propulsion force
        global_propulsion_force = self._transform_propulsion_force(creature=creature, decision=decision)
        propulsion_energy = self._calc_propulsion_energy(global_propulsion_force)

        # Gravity and normal force
        gravity_force, normal_force = self._calc_gravity_and_normal_forces(creature=creature)

        # Reaction friction force
        reaction_friction_force = self._calc_reaction_friction_force(
            normal_force=normal_force, propulsion_force=global_propulsion_force)

        # Drag force (air resistence)
        drag_force = self._calc_drag_force(creature=creature)

        total_force = reaction_friction_force + drag_force

        if debug_force:
            print(f'\t\t{reaction_friction_force=}\n'
                  f'\t\t{drag_force=}')

        # if self.is_agent:
        creature.log.add_record('gravity_force', gravity_force)
        creature.log.add_record('normal_force', normal_force)
        creature.log.add_record('reaction_friction_force', reaction_friction_force)
        creature.log.add_record('drag_force', drag_force)

        return total_force, propulsion_energy

    def _update_energy(self, creature, propulsion_energy,
                       debug_energy: bool = False):
        inner_energy = self._calc_inner_energy(creature)
        creature.energy -= propulsion_energy + inner_energy

        # if self.is_agent:
        creature.log.add_record('energy_propulsion', propulsion_energy)
        creature.log.add_record('energy_inner', inner_energy)
        creature.log.add_record('energy_consumption', inner_energy + propulsion_energy)
        if creature.log.record['energy_consumption'][-1] < 0:
            breakpoint('energy consumption < 0')
            raise ValueError('Energy consumption cannot be negative')
        if debug_energy: print(f'\t\t{propulsion_energy=:.1f} | {inner_energy=:.1f}')

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
