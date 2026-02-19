import copy

import numpy as np

from static_traits import StaticTraits
from creatures_log import CreaturesLogs
from input.codes.config import config

import importlib


class Creature(StaticTraits):
    """
    A dynamic creature in the simulation.
    Inherits static traits and adds dynamic properties such as position, speed, hunger, etc.
    """

    def __init__(self, creature_id: int, gen: int, parent_id: str | None, birth_step: int, color: np.ndarray,
                 max_age: int, max_mass: float, max_height: float, max_strength: float,
                 max_speed: float, max_energy: float,
                 digest_dict: dict, reproduction_energy: float,
                 eyes_channels: list[str], eyes_params: list[tuple], vision_limit: float,
                 brain,
                 position: np.ndarray):
        super().__init__(creature_id=creature_id,
                         gen=gen, parent_id=parent_id, birth_step=birth_step, color=color,
                         max_age=max_age, max_mass=max_mass, max_height=max_height, max_strength=max_strength,
                         max_speed=max_speed, max_energy=max_energy,
                         digest_dict=digest_dict, reproduction_energy=reproduction_energy,
                         eyes_channels=eyes_channels, eyes_params=eyes_params, vision_limit=vision_limit,
                         brain=brain)

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

    def init_state(self, balance=False):
        """
        :return:
        """
        self.is_agent = False
        # static trait - update for current max_age
        self.adolescence = self.max_age * config.ADOLESCENCE_AGE_FRACTION
        self.reproduction_cooldown = config.REPRODUCTION_COOLDOWN

        # dynamic traits
        self.age = 0
        if not balance:
            self.mass = 0.1 * self.max_mass
            self.height = 0.1 * self.max_height
            self.strength = 0.1 * self.max_strength

        self.energy = 0.8 * (
                config.REPRODUCTION_ENERGY + config.MIN_LIFE_ENERGY)  # TODO: patch for runability | was self.max_energy
        self.velocity = (np.random.rand(2) - 0.5) * self.max_speed
        self.max_speed_exp = np.linalg.norm(self.velocity)
        self.calc_speed()

        self.hunger = 100
        self.thirst = 100

        # TODO: logs are in step units for now
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
        self.ancestors.append(self.creature_id)
        self.creature_id = None  # updated in simulation

        self.mutate()
        self.brain.mutate(brain_mutation_rate=config.MUTATION_BRAIN)
        self.init_state()
        self.age = config.DT  # fix a delay in the logs
        self.velocity = -self.velocity  # go opposite to father
        self.calc_speed()
        # TODO: make the solution better than this patch...
        self.log.add_record('speed', self.speed)

    def mutate(self):
        """
        mutate the desired traits.
        """
        std_mutation_factors = config.STD_MUTATION_FACTORS
        for trait_key in std_mutation_factors:
            if np.random.rand(1) < config.MUTATION_CHANCE:
                if trait_key == 'eyes_params':
                    for eye_idx in range(len(self.eyes_params)):
                        for j in range(2):  # 2 for: angle offset, aperture
                            std_mutation_factor = std_mutation_factors[trait_key]
                            mutation_factor = np.random.normal(scale=std_mutation_factor)
                            self.eyes_params[eye_idx][j] += mutation_factor
                elif trait_key == 'digest_dict':
                    std_mutation_factor = np.array(list(std_mutation_factors[trait_key].values()))
                    mutation_factor = np.random.normal(scale=std_mutation_factor)
                    for i, food_type in enumerate(self.digest_dict.keys()):
                        if self.digest_dict[food_type] > 0:  # verify creature can eat this food type
                            self.digest_dict[food_type] += mutation_factor[i]
                            self.digest_dict[food_type] = np.clip(self.digest_dict[food_type], 0, 1)
                else:
                    std_mutation_factor = std_mutation_factors[trait_key]
                    mutation_factor = np.random.normal(scale=std_mutation_factor)
                    new_trait = getattr(self, trait_key) + mutation_factor
                    if trait_key == 'color':
                        new_trait = np.clip(new_trait, 0, 1)
                    else:
                        new_trait = max(0, new_trait)
                    setattr(self, trait_key, new_trait)


if __name__ == '__main__':

    brain_module = importlib.import_module(f"brain_models.{config.BRAIN_TYPE}")
    brain_obj = getattr(brain_module, 'Brain')
    brain = brain_obj([config.INPUT_SIZE, config.OUTPUT_SIZE])

    creature = Creature(creature_id=0, gen=0, parent_id="0", birth_step=0,
                        max_age=100, max_mass=20, max_height=2, max_strength=config.INIT_MAX_STRENGTH,
                        max_speed=config.MAX_SPEED, max_energy=config.INIT_MAX_ENERGY, color=np.random.rand(3),
                        digest_dict=config.INIT_HERBIVORE_DIGEST_DICT, reproduction_energy=config.REPRODUCTION_ENERGY,
                        eyes_channels=config.EYE_CHANNEL, eyes_params=config.EYES_PARAMS,
                        vision_limit=config.VISION_LIMIT,
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
