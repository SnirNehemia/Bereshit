import matplotlib.pyplot as plt
import matplotlib
import platform
from pathlib import Path

# make sure we can plot for debugging (did not test on debugger mode)
import matplotlib
import numpy as np

if platform.system() == 'Darwin':
    matplotlib.use('MacOSX')
else:
    matplotlib.use('TkAgg')

# for the video saving
project_folder = Path(__file__).parent
plt.rcParams['animation.ffmpeg_path'] = project_folder.joinpath(
    r'ffmpeg-2025-02-20-git-bc1a3bfd2c-essentials_build\bin\ffmpeg.exe')

# --------------------------------------- CONFIG PARAMETERS -------------------------------------------------------- #
# general config params
NOISE_STD = 0.5
DT = 1.0
NUM_FRAMES = 150
NUM_CREATURES = 100
SIMULATION_SPACE = 1000

# environment
ENV_PATH = r"Penvs\Env1.png"
GRASS_GENERATION_RATE = 5  # 5
LEAVES_GENERATION_RATE = 3  # 3

# Define eye parameters: (angle_offset in radians, aperture in radians)
# eyes_params = [(np.radians(30), np.radians(45)),(np.radians(-30), np.radians(45))]
EYES_PARAMS = ((np.radians(0), np.radians(60)))

# parameters of network
INPUT_SIZE = 2 + 2 + 3 * len(EYES_PARAMS) * 4
# 2 for hunger and thirst, 2 for speed, 3 (flag, distance, angle) for each eye * 4 channels
OUTPUT_SIZE = 2

SAVE_FILENAME = "simulation.mp4"

# For food
FOOD_DISTANCE_THRESHOLD = 5
LEAF_HEIGHT = 10
GRASS_ENERGY = 50
LEAF_ENREGY = 100

# For reproduction
MIN_LIFE_ENREGY = 500  # energy to be left after reproduction
MAX_MUTATION_FACTORS = {'max_age': 2,
                        'max_weight': 1,
                        'max_height': 1,
                        'max_speed': np.array([1, 1]),
                        'color': np.array([0, 0, 0]),  # +-in each RGB color

                        'energy_efficiency': 3,
                        'speed_efficiency': 2,
                        'food_efficiency': 2,
                        'reproduction_energy': 5,

                        'eyes_params': np.radians(5),  # +-degrees for each eye
                        'vision_limit': 4,
                        'brain': {'layer_addition': 0.5,
                                  'modify_weights': 0.2,
                                  'modify_layer': 0.3},

                        'weight': 3,
                        'height': 3,
                        'position': np.array([5, 5]),
                        'speed': np.array([2, 2]),
                        'energy': 3,
                        'hunger': 3,
                        'thirst': 3
                        }
