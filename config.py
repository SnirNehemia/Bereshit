import matplotlib.pyplot as plt
import matplotlib
import platform
from pathlib import Path
import os, sys

# make sure we can plot for debugging (did not test on debugger mode)
import matplotlib
import numpy as np
from datetime import datetime

if platform.system() == 'Darwin':
    matplotlib.use('MacOSX')
else:
    matplotlib.use('TkAgg')

# for the video saving
project_folder = Path(__file__).parent
plt.rcParams['animation.ffmpeg_path'] = project_folder.joinpath(
    r'ffmpeg-2025-02-20-git-bc1a3bfd2c-essentials_build\bin\ffmpeg.exe')

OUTPUT_FOLDER = project_folder.joinpath('outputs')
OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)
now = datetime.now()
date_str = now.strftime('%d_%m_%Y')
# --------------------------------------- DEBUGGING -------------------------------------------------------- #

DEBUG_MODE = False
if DEBUG_MODE:
    np.seterr(all='raise')  # Convert NumPy warnings into exceptions

# --------------------------------------- CONFIG PARAMETERS -------------------------------------------------------- #
run_str = 'V_' + now.strftime('%H_%M')  # change this to a different string to create a new output file
# general config params
np.random.seed = 0
NOISE_STD = 0.5
DT = 2.0  # time passing from frame to frame (relevant when calculating velocities)
NUM_FRAMES = 400  # the actual number of steps will be NUM_FRAMES * UPDATE_ANIMATION_INTERVAL
UPDATE_ANIMATION_INTERVAL = 30  # update the animation every n steps
FRAME_INTERVAL = 75  # interval between frames in ms
UPDATE_KDTREE_INTERVAL = 50  # update the kdtree every n steps
NUM_CREATURES = 500
MAX_NUM_CREATURES = 1100
SIMULATION_SPACE = 0  # will be updated in Environment class per the map size

# creature parameters
INIT_MAX_AGE = 4000
INIT_MAX_ENERGY = 2e3  # maybe useful for maturity test before reproduction
MAX_SPEED = 5.0  # maximum speed of the creature
PURGE_SPEED_THRESHOLD = 1  # if the creature's speed is below this threshold at first reproduction, it will be removed

# environment
ENV_PATH = r"Penvs\Env8.png"
GRASS_GENERATION_RATE = 2  # 5
GRASS_GROWTH_CHANCE = 0.5  # maybe will be useful to create droughts
LEAVES_GENERATION_RATE = 0  # 3
MAX_GRASS_NUM = 70
MAX_LEAVES_NUM = 50

# Define eye parameters: (angle_offset in radians, aperture in radians)
# eyes_params = ((np.radians(30), np.radians(45)),(np.radians(-30), np.radians(45)))
EYE_CHANNEL = ['grass'] #['grass', 'leaves', 'water', 'creatures']
EYES_PARAMS = [np.radians(0), np.radians(90)]     # angle_offset, aperture
VISION_LIMIT = 1000  # maximum distance that the creature can see

# parameters of network
INPUT_SIZE = 2 + 2 + 3 * len(EYES_PARAMS) * len(EYE_CHANNEL)
# 2 for hunger and thirst, 2 for speed, 3 (flag, distance, angle) for each eye * 4 channels
OUTPUT_SIZE = 2
MAX_D_SPEED = 0.5  # maximum change in speed per frame
MAX_D_ANGLE = np.radians(2)  # maximum change in angle per frame
NORM_INPUT = np.array([1,1,1,1])
for _ in range(len(EYES_PARAMS) * len(EYE_CHANNEL)):
    NORM_INPUT = np.append(NORM_INPUT, [1, VISION_LIMIT, 1])

# For food
FOOD_DISTANCE_THRESHOLD = 75
FOOD_SIZE = FOOD_DISTANCE_THRESHOLD/2 #3.14*(FOOD_DISTANCE_THRESHOLD/2)**2
LEAF_HEIGHT = 10
GRASS_ENERGY = 100
LEAF_ENERGY = 100

# For reproduction
REPRODUCTION_ENERGY = 700
MIN_LIFE_ENERGY = 150  # energy to be left after reproduction
MUTATION_CHANCE = 0.3  # number between 0-1 indicating chance of trait to be mutated
MAX_MUTATION_FACTORS = {'max_age': 2,
                        'max_weight': 1,
                        'max_height': 1,
                        'max_speed': 0.1,
                        'color': np.array([0.01, 0.01, 0.01]),  # +-in each RGB color

                        'energy_efficiency': 0,
                        'motion_efficiency': 0,
                        'food_efficiency': 0,
                        'reproduction_energy': 0,
                        'max_energy': 0,

                        'eyes_params': np.radians(5),  # +-degrees for each eye for angle and width
                        'vision_limit': 4,
                        'weight': 3,
                        'height': 3
                        # 'position': np.array([5, 5]),
                        # 'velocity': np.array([2, 2]),
                        # 'energy': 3,
                        # 'hunger': 3,
                        # 'thirst': 3
                        }
MUTATION_BRAIN = {'layer_addition': 0.1,
                   'modify_weights': 0.2,
                   'modify_layer': 0.2,
                  'modify_activation': 0.1}
# --------------------------------------- FILEPATHS -------------------------------------------------------- #
OUTPUT_FOLDER = OUTPUT_FOLDER.joinpath(date_str)
if not os.path.exists(OUTPUT_FOLDER):
    try:
        os.makedirs(OUTPUT_FOLDER)
        print(f"Directory '{OUTPUT_FOLDER}' created successfully.")
    except Exception as e:
        print(f"Error creating directory '{OUTPUT_FOLDER}': {e}")
        sys.exit(1)
else:
    print(f"Directory '{OUTPUT_FOLDER}' already exists.")

ANIMATION_FILEPATH = OUTPUT_FOLDER.joinpath(f"simulation_{date_str}_{run_str}.mp4")
SPECIFIC_FIG_FILEPATH = OUTPUT_FOLDER.joinpath(f"specific_fig_{date_str}_{run_str}.png")
STATISTICS_FIG_FILEPATH = OUTPUT_FOLDER.joinpath(f"statistics_fig_{date_str}_{run_str}.png")
