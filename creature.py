import copy

import numpy as np

from static_traits import StaticTraits
from config import Config as config
from physical_model import PhysicalModel as physical_model

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

        self.log_eat = None
        self.log_reproduce = None
        self.log_energy = None
        self.log_speed = None
        self.init_state()

    def init_state(self, rebalance=config.REBALANCE):
        """
        :return:
        """
        # dynamic traits
        self.age = 0
        self.mass = np.random.uniform(low=0.01, high=0.1) * self.max_mass
        self.height = np.random.uniform(low=0.01, high=0.1) * self.max_height
        self.strength = np.random.uniform(low=0.01, high=0.1) * self.max_strength

        self.energy = 0.95 * (config.REPRODUCTION_ENERGY + config.MIN_LIFE_ENERGY) # TODO: patch for runability | was self.max_energy
        self.velocity = (np.random.rand(2) - 0.5) * self.max_speed
        self.max_speed_exp = np.linalg.norm(self.velocity)
        self.calc_speed()

        self.hunger = 100
        self.thirst = 100

        self.log_eat = []
        self.log_reproduce = []
        self.log_energy = []
        self.log_speed = []
        if rebalance:
            self.gained_energy = []
            self.height_energy = []
            self.mass_energy = []

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
        self.velocity = -self.velocity  # go opposite to father

    def mutate(self):
        """
        mutate the desired traits.
        """
        max_mutation_factors = config.MAX_MUTATION_FACTORS
        for trait_key in max_mutation_factors:
            if np.random.rand(1) < config.MUTATION_CHANCE:
                if trait_key == 'eyes_params':
                    # Mutate eyes_params
                    for eye_idx in range(len(self.eyes_params)):
                        for j in range(2):  # 2 for: angle offset, aperture
                            max_mutation_factor = max_mutation_factors[trait_key]
                            mutation_factor = np.random.uniform(-max_mutation_factor, max_mutation_factor)
                            self.eyes_params[eye_idx][j] += mutation_factor
                # mutate digest dict
                elif trait_key == 'digest_dict':
                    max_mutation_factor = np.array(list(max_mutation_factors[trait_key].values()))
                    mutation_factor = np.random.uniform(-max_mutation_factor, max_mutation_factor)
                    for i, food_type in enumerate(self.digest_dict.keys()):
                        self.digest_dict[food_type] += mutation_factor[i]
                        self.digest_dict[food_type] = np.clip(self.digest_dict[food_type], 0, 1)
                else:
                    max_mutation_factor = max_mutation_factors[trait_key]
                    mutation_factor = np.random.uniform(-max_mutation_factor, max_mutation_factor)
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

        # static friction force
        mu_static = physical_model.mu_static
        static_friction_force_mag = mu_static * np.linalg.norm(normal_force)

        # kinetic friction force
        mu_kinetic = physical_model.mu_kinetic
        alpha_mu = physical_model.alpha_mu
        mu_total = mu_kinetic + (mu_static - mu_kinetic) * np.exp(-alpha_mu * self.speed)
        kinetic_friction_force = - mu_total * np.linalg.norm(normal_force) * global_propulsion_force_direction

        # reaction friction force used for movement:
        # when propulsion force is within the static friction force limit
        # there is reaction force opposite to propulsion force else kinetic friction opposite to propulsion force
        if np.linalg.norm(global_propulsion_force) <= static_friction_force_mag:
            reaction_friction_force = - global_propulsion_force
        else:
            reaction_friction_force = kinetic_friction_force

        # drag force (air resistence)
        linear_drag_force = - physical_model.gamma * self.velocity
        quadratic_drag_force = - physical_model.c_drag * self.speed ** 2 * current_direction
        drag_force = linear_drag_force + quadratic_drag_force

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
        rest_energy = physical_model.energy_conversion_factors['rest'] * self.mass ** 0.75  # called BMR energy
        inner_energy = rest_energy + c_d * np.sum(list(self.digest_dict.values())) + c_h * self.height
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

        excess_energy = gained_energy - height_energy - mass_energy
        if rebalance:
            self.gained_energy.append(gained_energy)
            self.height_energy.append(height_energy)
            self.mass_energy.append(mass_energy)
        self.energy += excess_energy

    def plot_rebalance(self, ax, debug=False, mode='energy'):
        if debug:
            import matplotlib.pyplot as plt
            plt.ion()
            fig, ax = plt.subplots(1, 1)
        ax.clear()
        if mode == 'speed':
            ax.plot(self.log_speed, color='teal', alpha=0.5)
            ax.set_ylim(0, max(self.log_speed) * 1.1)
            ax.tick_params(axis='y', colors='teal')
            ax.spines['left'].set_color('maroon')
            ax.spines['right'].set_color('teal')
        elif mode == 'energy':
            ax.plot(self.log_energy, color='maroon', alpha=0.5)
            ax.set_ylim(0, (config.REPRODUCTION_ENERGY + config.MIN_LIFE_ENERGY) * 1.1)
            ax.tick_params(axis='y', colors='maroon')
            ax.spines['left'].set_color('maroon')
            ax.spines['right'].set_color('teal')

        # ax2 = ax.twinx()
        # ax2.clear()
        # ax2.plot(self.log_speed)
        # ax.plot(self.height_energy)
        # ax.plot(self.mass_energy)

    def plot_live_status(self, ax, debug=False, plot_horizontal=True):
        """
        Plots the agent's status (energy, hunger, thirst) on the given axes.
        """
        if debug:
            import matplotlib.pyplot as plt
            plt.ion()
            fig, ax = plt.subplots(1, 1)
        # Define attributes dynamically
        ls = ['energy', 'age']  # , 'hunger', 'thirst'
        colors = ['green', 'grey']  # , 'red', 'blue'
        values = [getattr(self, attr) for attr in ls]  # Dynamically get values
        ax.clear()
        ax.set_title(f'Agent # {self.creature_id} Status | ancestors num = len({self.ancestors})')
        if plot_horizontal:
            ax.barh(ls, values, color=colors)
            if 'energy' in ls:
                ax.scatter([config.REPRODUCTION_ENERGY + config.MIN_LIFE_ENERGY], ['energy'], color='black', s=20)
            if 'age' in ls:
                ax.scatter([self.max_age], ['age'], color='black', s=20)
                # ax.barh(['Energy', 'Hunger', 'Thirst'], [self.energy, self.hunger, self.thirst], color=['green', 'red', 'blue'])
                ax.set_xlim(0, max(config.REPRODUCTION_ENERGY + config.MIN_LIFE_ENERGY, self.max_age))
                # ax.set_xticks([0,self.max_energy/2, self.max_energy])
                ax.set_yticks(ls)
        else:
            ax.bar(ls, values, color=colors)
            if 'energy' in ls:
                ax.scatter( ['energy'], [config.REPRODUCTION_ENERGY + config.MIN_LIFE_ENERGY], color='black', s=20)
            if 'age' in ls:
                ax.scatter( ['age'], [self.max_age], color='black', s=20)
                ax.set_ylim(0, max(config.REPRODUCTION_ENERGY + config.MIN_LIFE_ENERGY, self.max_age))
                ax.set_xticks(ls)
                ax.set_xticklabels(ls, rotation=90, ha='right')
                ax.set_yticks([])
                ax.yaxis.set_tick_params(labelleft=False)




    def plot_acc_status(self, ax, debug=False, plot_type=1, curr_step=-1):
        """
        Plots the agent's accumulated status (logs) on the given axes.
        """
        if debug:
            print('debug_mode')
            import matplotlib.pyplot as plt
            plt.ion()
            fig, ax = plt.subplots(1, 1)
        # Define attributes dynamically
        ls = ['log_eat', 'log_reproduce']
        colors = ['green', 'pink']
        ax.clear()
        if max(self.color) > 1 or min(self.color) < 0:
            raise ('color exceed [0, 1] range')
        ax.set_facecolor(list(self.color) + [0.3])
        if plot_type == 0:
            # option 1
            values = [len(getattr(self, attr)) for attr in ls]  # Dynamically get values
            ax.set_title(f'Agent # {self.id} Accumulated Status')
            ax.bar(ls, values, color=colors, width=0.2)
            ax.set_ylim(0, 10)
            ax.set_yticks([0, 5, 10, 100])
            ax.set_xticks(ls)
        if plot_type == 1:
            # option 2
            if curr_step == -1: curr_step = self.max_age / config.DT + self.birth_step
            # values = [getattr(self, attr) for attr in ls]  # Dynamically get values
            eating_frames = self.log_eat
            reproducing_frames = self.log_reproduce
            ax.scatter(eating_frames, [1] * len(eating_frames), color='green', marker='o', s=100, label='Eating')
            ax.scatter(reproducing_frames, [2] * len(reproducing_frames), color='red', marker='D', s=100,
                       label='Reproducing')
            ax.set_yticks([1, 2])
            # ax.set_yticklabels(['Eating', 'Reproducing'])
            # Label x-axis and add a title
            ax.set_xlabel('Frame Number')
            ax.set_title('Event Timeline: Eating & Reproducing')
            ax.set_xlim([self.birth_step - 1, curr_step + 1])
            ax.set_ylim([0.5, 2.5])
            ax.legend()

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
