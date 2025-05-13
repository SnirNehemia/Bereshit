import copy

import numpy as np

from static_traits import StaticTraits
from creatures_log import CreaturesLogs
from input.codes.config import config
from input.codes.physical_model import physical_model

import importlib

brain_module = importlib.import_module(f"brain_models.{config.BRAIN_TYPE}")
Brain = getattr(brain_module, 'Brain')


class Creature(StaticTraits):
    """
    A dynamic creature in the simulation.
    Inherits static traits and adds dynamic properties such as position, speed, hunger, etc.
    """

    def __init__(self, creature_id: int, gen: int, parent_id: str | None, birth_step: int, color: np.ndarray,
                 max_age: int, max_mass: float, max_height: float, max_strength: float,
                 max_speed: float, max_energy: float,
                 digest_dict: dict, reproduction_energy: float,
                 eyes_params: list[tuple], vision_limit: float, brain: Brain,
                 position: np.ndarray):
        super().__init__(creature_id=creature_id,
                         gen=gen, parent_id=parent_id, birth_step=birth_step, color=color,
                         max_age=max_age, max_mass=max_mass, max_height=max_height, max_strength=max_strength,
                         max_speed=max_speed, max_energy=max_energy,
                         digest_dict=digest_dict, reproduction_energy=reproduction_energy,
                         eyes_params=eyes_params, vision_limit=vision_limit, brain=brain)

        self.age = None
        self.mass = None
        self.height = None
        self.strength = None

        self.position = position
        self.velocity = None
        self.speed = None
        self.max_speed_exp = None
        self.energy = None

        self.hunger = None
        self.thirst = None

        self.init_state()
        # self.log.record['speed'] = [self.speed]  # fixes an issue with the first generation

    def init_state(self, rebalance=config.REBALANCE):
        """
        :return:
        """
        self.is_agent = False
        # static trait - update for current max_age
        self.adolescence = self.max_age * config.ADOLESCENCE_AGE_FRACTION
        self.reproduction_cooldown = config.REPRODUCTION_COOLDOWN

        # dynamic traits
        self.age = 0
        self.mass = np.random.uniform(low=0.01, high=0.1) * self.max_mass
        self.height = np.random.uniform(low=0.01, high=0.1) * self.max_height
        self.strength = np.random.uniform(low=0.01, high=0.1) * self.max_strength

        self.energy = 0.8 * (
                    config.REPRODUCTION_ENERGY + config.MIN_LIFE_ENERGY)  # TODO: patch for runability | was self.max_energy
        self.velocity = (np.random.rand(2) - 0.5) * self.max_speed
        self.max_speed_exp = np.linalg.norm(self.velocity)
        self.calc_speed()

        self.hunger = 100
        self.thirst = 100

        # TODO: logs are in step units for now
        self.log = CreaturesLogs(self.creature_id)  # pay attention! the id is None for children and it is defines only later, in the simulation


    def make_agent(self):
        self.is_agent = True

    def get_heading(self):
        """
        Returns the creature's current heading (unit vector).
        If the creature is stationary, defaults to (1, 0).
        """
        if hasattr(self, 'velocity') and self.speed > 1e-3:
            return self.velocity / self.speed
        else:
            theta = np.random.uniform(0, 2 * np.pi)
            return np.array([np.cos(theta), np.sin(theta)])

    def calc_speed(self):
        """
        Returns the creature's current speed vector
        and updated max speed experienced (with recursive averaging)
        """
        self.speed = np.linalg.norm(self.velocity)
        # self.max_speed_exp = max(self.max_speed_exp, self.speed)
        self.max_speed_exp = (self.max_speed_exp + self.speed) / 2

    def can_reproduce(self, step_num):
        """
        Returns 1 if the creature can reproduce (i.e., it is old enough and has not reproduced recently),
        else returns 0.
        """
        if self.age > self.adolescence:
            if len(self.log.record['reproduce']) > 0:
                if (step_num - self.log.record['reproduce'][-1]) * config.DT > self.reproduction_cooldown:
                    return 1
                else: return 0
            else: return 1  # if it's his first time reproducing
        else:
            return 0


    def reproduce(self):
        """
        Returns a new creature with mutated traits.
        """
        # Reproduce child
        child = copy.deepcopy(self)
        child.reset()

        # Reduce energy of reproduction
        self.energy -= self.reproduction_energy
        return child

    def reset(self):
        """
        Reset the creature to initial state and flip velocity.
        """
        self.gen += 1
        self.parent_id = self.creature_id
        self.ancestors.append(self.creature_id)
        self.creature_id = None  # updated in simulation

        self.mutate()
        self.brain.mutate(brain_mutation_rate=config.MUTATION_BRAIN)
        self.init_state()
        self.age = config.DT  # fix a delay in the logs
        self.velocity = -self.velocity  # go opposite to father

    def mutate(self):
        """
        mutate the desired traits.
        """
        std_mutation_factors = config.STD_MUTATION_FACTORS
        for trait_key in std_mutation_factors:
            if np.random.rand(1) < config.MUTATION_CHANCE:
                if trait_key == 'eyes_params':
                    # Mutate eyes_params
                    for eye_idx in range(len(self.eyes_params)):
                        for j in range(2):  # 2 for: angle offset, aperture
                            std_mutation_factor = std_mutation_factors[trait_key]
                            mutation_factor = np.random.normal(scale=std_mutation_factor)
                            self.eyes_params[eye_idx][j] += mutation_factor
                # mutate digest dict
                elif trait_key == 'digest_dict':
                    std_mutation_factor = np.array(list(std_mutation_factors[trait_key].values()))
                    mutation_factor = np.random.normal(scale=std_mutation_factor)
                    for i, food_type in enumerate(self.digest_dict.keys()):
                        self.digest_dict[food_type] += mutation_factor[i]
                        self.digest_dict[food_type] = np.clip(self.digest_dict[food_type], 0, 1)
                else:
                    std_mutation_factor = std_mutation_factors[trait_key]
                    mutation_factor = np.random.normal(scale=std_mutation_factor)
                    setattr(self, trait_key, getattr(self, trait_key) + mutation_factor)

        # clip relevant traits
        self.color = np.clip(self.color, 0, 1)

    def move(self, decision: np.array([float]), dt: float = config.DT):

        # constrain propulsion force based on strength
        propulsion_force_mag, relative_propulsion_force_angle = decision
        propulsion_force_mag = np.clip(propulsion_force_mag, 0, self.strength)

        # transform relative propulsion_force to global cartesian coordinates (x,y)
        current_direction = self.get_heading()
        cos_angle = np.cos(relative_propulsion_force_angle)
        sin_angle = np.sin(relative_propulsion_force_angle)
        global_propulsion_force_direction = np.array([
            current_direction[0] * cos_angle - current_direction[1] * sin_angle,
            current_direction[0] * sin_angle + current_direction[1] * cos_angle
        ])
        global_propulsion_force = global_propulsion_force_direction * propulsion_force_mag

        # gravity and normal force (right now used only for friction and not in equation of motion because 2D movement)
        gravity_force = self.mass * physical_model.g
        normal_force = - gravity_force

        # reaction friction force
        if propulsion_force_mag > physical_model.mu_static * np.linalg.norm(normal_force):
            reaction_friction_force = - physical_model.mu_kinetic * np.linalg.norm(normal_force) * global_propulsion_force_direction
        else:
            reaction_friction_force = - global_propulsion_force

        # drag force (air resistence)
        linear_drag_force = - physical_model.gamma * self.height ** 2 * self.velocity
        quadratic_drag_force = - physical_model.c_drag * self.height ** 2 * self.speed ** 2 * current_direction
        drag_force = linear_drag_force + quadratic_drag_force

        # if self.is_agent:
        self.log.add_record('linear_drag_force', np.linalg.norm(linear_drag_force))
        self.log.add_record('quadratic_drag_force', np.linalg.norm(quadratic_drag_force))
        self.log.add_record('reaction_friction_force_mag',
                            np.linalg.norm(reaction_friction_force))
        self.log.add_record('reaction_friction_force_angle',
                            relative_propulsion_force_angle)

        # calc new velocity and position
        acceleration = (reaction_friction_force + drag_force) / self.mass
        new_velocity = self.velocity + acceleration * dt
        new_position = self.position + new_velocity * dt

        # print(f'{acceleration=}\n'
        #       f'{self.velocity=} --> {new_velocity=}\n'
        #       f'{self.position} --> {new_position}')

        # update position, velocity and speed
        self.velocity = new_velocity
        self.calc_speed()
        self.position = new_position

        # update energy
        propulsion_energy = self.calc_propulsion_energy(global_propulsion_force)
        inner_energy = self.calc_inner_energy()
        # if self.is_agent:
        self.log.add_record('energy_propulsion', propulsion_energy)
        self.log.add_record('energy_inner', inner_energy)
        self.log.add_record('energy_consumption', inner_energy + propulsion_energy)
        if self.log.record['energy_consumption'][-1] < 0:
            raise ValueError('Energy consumption cannot be negative')
        self.energy -= propulsion_energy + inner_energy

    @staticmethod
    def calc_propulsion_energy(propulsion_force):
        eta = physical_model.energy_conversion_factors['activity_efficiency']
        c_heat = physical_model.energy_conversion_factors['heat_loss']
        propulsion_energy = (1 / eta + c_heat) * np.linalg.norm(propulsion_force)
        return propulsion_energy

    def calc_inner_energy(self):
        c_d = physical_model.energy_conversion_factors['digest']
        c_h = physical_model.energy_conversion_factors['height']
        rest_energy = physical_model.energy_conversion_factors['rest'] * self.mass ** 0.75  # adds mass (BMR) energy
        inner_energy = rest_energy + c_d * np.sum(list(self.digest_dict.values())) + c_h * self.height  # adds height energy
        inner_energy = inner_energy + self.brain.size * physical_model.energy_conversion_factors['brain_consumption']
        return inner_energy

    @staticmethod
    def calc_trait_energy(trait_type, gained_energy, age):
        trait_energy_params = physical_model.trait_energy_params_dict[trait_type]
        factor = trait_energy_params['factor']
        rate = trait_energy_params['rate']
        trait_energy_func = physical_model.trait_energy_func(factor=factor, rate=rate, age=age)
        trait_energy = trait_energy_func * gained_energy
        return trait_energy

    def convert_gained_energy_to_trait(self, trait_type: str, old_trait: float,
                                       gained_energy: float, age: float,
                                       ):
        trait_energy = self.calc_trait_energy(trait_type=trait_type, gained_energy=gained_energy, age=age)
        c_trait = physical_model.energy_conversion_factors[trait_type]
        new_trait = old_trait + trait_energy / c_trait
        return new_trait, trait_energy

    def eat(self, food_type, food_energy, rebalance=config.REBALANCE):
        gained_energy = self.digest_dict[food_type] * food_energy

        if self.age < self.adolescence:
            self.height, height_energy = self.convert_gained_energy_to_trait(
                trait_type='height',
                old_trait=self.height,
                gained_energy=gained_energy,
                age=self.age,
            )
            self.mass, mass_energy = self.convert_gained_energy_to_trait(
                trait_type='mass',
                old_trait=self.mass,
                gained_energy=gained_energy,
                age=self.age,
            )
        else:
            mass_energy, height_energy = 0, 0

        excess_energy = gained_energy - height_energy - mass_energy
        if rebalance:
            self.log.add_record('energy_excess',excess_energy)
            self.log.add_record('energy_gain',gained_energy)
            self.log.add_record('energy_height',height_energy)
            self.log.add_record('energy_mass',mass_energy)
        else:
            self.log.add_record('energy_excess', excess_energy)
        self.energy += excess_energy


if __name__ == '__main__':
    creature = Creature(creature_id=0, gen=0, parent_id="0", birth_step=0,
                        max_age=100, max_mass=20, max_height=2, max_strength=config.INIT_MAX_STRENGTH,
                        max_speed=config.MAX_SPEED, max_energy=config.INIT_MAX_ENERGY, color=np.random.rand(3),
                        digest_dict=config.INIT_DIGEST_DICT, reproduction_energy=config.REPRODUCTION_ENERGY,
                        eyes_params=config.EYES_PARAMS, vision_limit=config.VISION_LIMIT,
                        brain=Brain([config.INPUT_SIZE, config.OUTPUT_SIZE]),
                        position=np.array([10, 10]))

    energies, velocities, positions = [], [], []

    time_steps = 10
    for _ in range(time_steps):
        creature.move(decision=[1, np.radians(5)])
        energies.append(creature.energy)
        velocities.append(creature.velocity)
        positions.append(creature.position)

    import matplotlib.pyplot as plt
    from matplotlib import use

    use('TkAgg')

    vx, vy = np.zeros_like(velocities), np.zeros_like(velocities)
    pos_x, pos_y = np.zeros_like(positions), np.zeros_like(positions)
    for i in range(time_steps):
        vx[i], vy[i] = velocities[i]
        pos_x[i], pos_y[i] = positions[i]

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(energies)
    ax[1].plot(vx, vy, marker="o", markersize=2, linestyle="-", color="b", label="Path")
    ax[1].scatter(vx[0], vy[0], color="green", label="Start", s=100)  # Mark start position
    ax[1].scatter(vx[-1], vy[-1], color="red", label="End", s=100)  # Mark end position
    ax[1].legend()
    ax[2].plot(pos_x, pos_y, marker="o", markersize=2, linestyle="-", color="b", label="Path")
    ax[2].scatter(pos_x[0], pos_y[0], color="green", label="Start", s=100)  # Mark start position
    ax[2].scatter(pos_x[-1], pos_y[-1], color="red", label="End", s=100)  # Mark end position
    ax[2].legend()
    plt.show()
