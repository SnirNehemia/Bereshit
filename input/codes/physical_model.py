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
