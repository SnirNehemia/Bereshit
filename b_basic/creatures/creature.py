import copy

import numpy as np

from b_basic.creatures.static_traits import StaticTraits
from e_logs.creatures_log import CreaturesLogs
from b_basic.sim_config import sim_config

import importlib


class Creature(StaticTraits):
    """
    A dynamic creature in the simulation.
    Inherits static traits and adds dynamic properties such as position, velocity, etc.
    """

    def __init__(self, creature_id: int, gen: int, parent_id: str | None, birth_step: int, color: np.ndarray,
                 max_age: int, max_mass: float, max_height: float, max_strength: float,
                 max_speed: float, max_energy: float,
                 digest_dict: dict,
                 reproduction_cooldown: float, reproduction_energy: float,
                 eyes: list[list], vision_limit: float,
                 brain,
                 position: np.ndarray):
        super().__init__(creature_id=creature_id,
                         gen=gen, parent_id=parent_id, birth_step=birth_step, color=color,
                         max_age=max_age, max_mass=max_mass, max_height=max_height, max_strength=max_strength,
                         max_speed=max_speed, max_energy=max_energy,
                         digest_dict=digest_dict,
                         reproduction_cooldown=reproduction_cooldown, reproduction_energy=reproduction_energy,
                         eyes=eyes, vision_limit=vision_limit, brain=brain)

        self.age = None
        self.mass = None
        self.height = None
        self.strength = None

        self.position = position
        self.velocity = None
        self.speed = None
        self.max_speed_exp = None
        self.energy = None
        self.is_agent = False

        self.init_state()
        # self.log.record['speed'] = [self.speed]  # fixes an issue with the first generation

    def init_state(self):
        """
        :return:
        """
        self.is_agent = False
        # static trait - update for current max_age
        self.adolescence_age = self.max_age * sim_config.config.ADOLESCENCE_AGE_FRACTION

        # dynamic traits
        self.age = 0
        self.mass = sim_config.config.INIT_FROM_MAX_FRACTION * self.max_mass
        self.height = sim_config.config.INIT_FROM_MAX_FRACTION * self.max_height
        self.strength = sim_config.config.INIT_FROM_MAX_FRACTION * self.max_strength

        self.energy = sim_config.config.INIT_ENERGY_FRACTION * (
                sim_config.config.REPRODUCTION_ENERGY + sim_config.config.MIN_LIFE_ENERGY)
        self.velocity = (np.random.rand(2) - 0.5) * 2 / sim_config.config.SQRT2 * self.max_speed
        self.max_speed_exp = np.linalg.norm(self.velocity)
        self.calc_speed()

        self.log = CreaturesLogs(
            self.creature_id)  # pay attention! the id is None for children and it is defines only later, in the simulation

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
        self.max_speed_exp = max([self.max_speed_exp, self.speed])

    def can_reproduce(self, step_num):
        """
        Returns 1 if the creature can reproduce (i.e., it is old enough and has not reproduced recently),
        else returns 0.
        """
        if self.age > self.adolescence_age:
            if len(self.log.record['reproduce']) > 0:
                if (step_num - self.log.record['reproduce'][-1]) * sim_config.config.DT > self.reproduction_cooldown:
                    return 1
                else:
                    return 0
            else:
                return 1  # if it's his first time reproducing
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
        self.creature_id = None  # updated in simulation

        self.mutate()
        self.brain.mutate(brain_mutation_rate=sim_config.config.MUTATION_BRAIN)
        self.init_state()
        self.age = sim_config.config.DT  # fix a delay in the logs
        self.velocity = -self.velocity  # go opposite to father
        self.calc_speed()
        self.log.add_record('speed', self.speed)

    def mutate(self):
        """
        mutate the desired traits.
        """
        mutation_traits_std = sim_config.config.MUTATION_TRAITS_STD
        for trait_key, mutation_trait_std in mutation_traits_std.items():
            if np.random.rand(1) < sim_config.config.MUTATION_CHANCE:
                if trait_key == 'eyes':
                    for eye_idx in range(len(self.eyes)):
                        mutation_factor = np.random.normal(scale=mutation_trait_std,
                                                           size=2)  # 2 for angle offset and aperture
                        self.eyes[eye_idx] += mutation_factor
                elif trait_key == 'digest_dict':
                    mutation_trait_std = np.array(list(mutation_trait_std.values()))
                    mutation_factor = np.random.normal(scale=mutation_trait_std)
                    for i, food_type in enumerate(self.digest_dict.keys()):
                        if self.digest_dict[food_type] > 0:  # verify creature can eat this food type
                            self.digest_dict[food_type] += mutation_factor[i]
                            self.digest_dict[food_type] = np.clip(self.digest_dict[food_type], 0, 1)
                else:
                    mutation_factor = np.random.normal(scale=mutation_trait_std)
                    new_trait = getattr(self, trait_key) + mutation_factor
                    if trait_key == 'color':
                        new_trait = np.clip(new_trait, 0, 1)
                    else:
                        new_trait = max(0, new_trait)
                    setattr(self, trait_key, new_trait)


if __name__ == '__main__':

    brain_module = importlib.import_module(f"brain_models.{sim_config.config.BRAIN_TYPE}")
    brain_obj = getattr(brain_module, 'Brain')
    brain = brain_obj([sim_config.config.INPUT_SIZE, sim_config.config.OUTPUT_SIZE])

    creature = Creature(creature_id=0, gen=0, parent_id="0", birth_step=0,
                        max_age=100, max_mass=20, max_height=2, max_strength=sim_config.config.INIT_MAX_STRENGTH,
                        max_speed=sim_config.config.INIT_MAX_SPEED, max_energy=sim_config.config.INIT_MAX_ENERGY,
                        color=np.random.rand(3),
                        digest_dict=sim_config.config.INIT_HERBIVORE_DIGEST_DICT,
                        reproduction_energy=sim_config.config.REPRODUCTION_ENERGY,
                        reproduction_cooldown=sim_config.config.REPRODUCTION_COOLDOWN,
                        eyes=sim_config.config.EYES,
                        vision_limit=sim_config.config.VISION_LIMIT,
                        brain=brain,
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
