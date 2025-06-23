import dataclasses

import numpy as np

from input.codes import repos_utils
from input.codes.yaml_reading import read_yaml

physical_model = None


def load_physical_model(yaml_relative_path: str = ""):
    global physical_model
    if physical_model is None:
        physical_model = PhysicalModel(yaml_relative_path=yaml_relative_path)
    return physical_model


@dataclasses.dataclass
class PhysicalModel:
    def __init__(self, yaml_relative_path):
        # init config based on data from yaml
        self.project_folder = repos_utils.fetch_directory()
        self.yaml_path = self.project_folder.joinpath(yaml_relative_path)
        yaml_data = read_yaml(filepath=self.yaml_path)
        for key, value in yaml_data.items():
            setattr(self, key, value)

        # make needed adjustments
        self.trait_energy_func = lambda factor, rate, age: factor * np.exp(-rate * age)

    def calc_gravity_and_normal_forces(self, mass):
        """
        # Calculate gravity and normal force.
        Right now (2D movement) normal force is only used for friction
        and not directly in equation of motion.
        :param mass: creature mass
        :return:
        """
        gravity_force = mass * self.g
        normal_force = - gravity_force
        return gravity_force, normal_force

    def calc_reaction_friction_force(self, normal_force, propulsion_force):
        normal_force_mag = np.linalg.norm(normal_force)
        propulsion_force_mag = np.linalg.norm(propulsion_force)
        if propulsion_force_mag > self.mu_static * normal_force_mag:
            propulsion_force_direction = propulsion_force / propulsion_force_mag
            reaction_friction_force = - self.mu_kinetic * normal_force_mag * \
                                      propulsion_force_direction
        else:
            reaction_friction_force = - propulsion_force
        return reaction_friction_force

    def calc_drag_force(self, height, velocity, speed):
        drag_force = [0, 0]

        if speed > 1e-3:
            current_direction = velocity / speed
            linear_drag_force = - self.gamma * height ** 2 * velocity
            quadratic_drag_force = - self.c_drag * height ** 2 * speed ** 2 * current_direction
            drag_force = linear_drag_force + quadratic_drag_force

        return drag_force
