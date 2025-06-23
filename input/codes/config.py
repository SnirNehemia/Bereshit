import dataclasses

import platform
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from input.codes import repos_utils
from input.codes.yaml_reading import read_yaml

config = None


def load_config(yaml_relative_path: str | Path = ""):
    global config
    if config is None:
        config = Config(yaml_relative_path=yaml_relative_path)
    return config


@dataclasses.dataclass
class Config:
    def __init__(self, yaml_relative_path):
        # init config based on data from yaml
        self.project_folder = repos_utils.fetch_directory()
        self.yaml_path = self.project_folder.joinpath(yaml_relative_path)
        yaml_data = read_yaml(filepath=self.yaml_path)
        for key, value in yaml_data.items():
            setattr(self, key, value)

        # Define time formats
        now = datetime.now()
        self.datestamp = now.strftime('%Y-%m-%d')  # a directory of this name will be created
        self.timestamp = now.strftime('%Y-%m-%d_T_%H-%M-%S')   # id for all run's outputs

        # make needed adjustments
        self.update_config()

    def update_config(self):
        # set random seed (for mutation repeatability)
        np.random.seed = 0

        # init plt things
        if platform.system() == 'Darwin':
            matplotlib.use('MacOSX')
        else:
            matplotlib.use('TkAgg')

        plt.rcParams['animation.ffmpeg_path'] = \
            self.project_folder.joinpath(self.FFMPEG_PATH)

        # Environment parameters
        setattr(self, 'ENV_PATH', self.project_folder.joinpath(self.ENV_PATH))

        # Food parameters
        setattr(self, 'FOOD_SIZE', self.FOOD_DISTANCE_THRESHOLD / 2)  # for display

        # Eyes parameters
        # Convert degrees to radians
        for i in range(len(self.EYES_PARAMS)):
            self.EYES_PARAMS[i][0] = np.radians(self.EYES_PARAMS[i][0])
            self.EYES_PARAMS[i][1] = np.radians(self.EYES_PARAMS[i][1])

        # Brain parameters  # TODO - need to change norm_input to depend on creature if have different eyes
        norm_input = self.NORM_INPUT
        for _ in range(len(self.EYES_PARAMS) * len(self.EYE_CHANNEL)):
            norm_input = np.append(norm_input, [1, self.VISION_LIMIT, 1])
        setattr(self, 'NORM_INPUT', norm_input)

        # Mutations parameters
        if self.BRAIN_TYPE == 'fully_connected_brain':  # 'fully_connected_brain' or 'graphic_brain'
            setattr(self, 'MUTATION_BRAIN', self.MUTATION_FC_BRAIN)
        elif self.BRAIN_TYPE == 'graphic_brain':
            setattr(self, 'MUTATION_BRAIN', self.MUTATION_GRAPH_BRAIN)

        self.STD_MUTATION_FACTORS['color'] = np.ones(3) * self.STD_MUTATION_FACTORS['color']
        self.STD_MUTATION_FACTORS['eyes_params'] = np.radians(self.STD_MUTATION_FACTORS['eyes_params'])

        # Filepaths
        setattr(self, 'OUTPUT_FOLDER',
                self.project_folder.joinpath(f'outputs').joinpath(self.datestamp).joinpath(self.timestamp))

        setattr(self, 'ANIMATION_FILEPATH',
                self.OUTPUT_FOLDER.joinpath(f"{self.timestamp}_simulation.mp4"))
        setattr(self, 'SPECIFIC_FIG_FILEPATH',
                self.OUTPUT_FOLDER.joinpath(f"{self.timestamp}_specific_fig.png"))
        setattr(self, 'STATISTICS_FIG_FILEPATH',
                self.OUTPUT_FOLDER.joinpath(f"{self.timestamp}_statistics_fig.png"))
        setattr(self, 'ENV_FIG_FILE_PATH',
                self.OUTPUT_FOLDER.joinpath(f"{self.timestamp}_env_fig.png"))
        setattr(self, 'STATISTICS_LOGS_JSON_FILEPATH',
                self.OUTPUT_FOLDER.joinpath(f"{self.timestamp}_statistics_logs.json"))
