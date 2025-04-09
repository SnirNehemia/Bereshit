import dataclasses

import platform
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

if platform.system() == 'Darwin':
    matplotlib.use('MacOSX')
else:
    matplotlib.use('TkAgg')

# for the video saving
project_folder = Path(__file__).parent
plt.rcParams['animation.ffmpeg_path'] = project_folder.joinpath(
    r'ffmpeg-2025-02-20-git-bc1a3bfd2c-essentials_build\bin\ffmpeg.exe')

np.random.seed = 0


@dataclasses.dataclass
class Config:
    # Simulation parameters
    DT = 2.5  # time passing from frame to frame (relevant when calculating velocities) and what is the resolution of every calculation?
    NUM_FRAMES = 120  # the actual number of steps = NUM_FRAMES * UPDATE_ANIMATION_INTERVAL
    UPDATE_ANIMATION_INTERVAL = 40  # 30  # update the animation every n steps = every n*dt seconds
    FRAME_INTERVAL = 75  # interval between frames in animation [in ms]
    STATUS_EVERY_STEP = True  # choose if to update every step or every frame
    UPDATE_KDTREE_INTERVAL = 120  # update the kdtree every n steps
    DEBUG_MODE = False  # breakpoint in update_func after each frame (not step!)
    REBALANCE = True
    PURGE_POP_PERCENTAGE = 0.75  # percentage of MAX_NUM_CREATURES to stuck simulation

    # Purge parameters
    DO_PURGE = True
    PURGE_PERCENTAGE = 0.1  # percentage out of MAX_NUM_CREATURES to do purge
    PURGE_SPEED_THRESHOLD = 1  # if the creature's speed is below this threshold at first reproduction, it will be removed
    PURGE_FRAME_FREQUENCY = 50  # do purge every PURGE_FRAME_FREQUENCY frames

    # Environment parameters
    ENV_PATH = r"Penvs\Env8.png"
    BOUNDARY_CONDITION = 'zero'  # what to do with the velocity on the boundaries - 'mirror' or 'zero'
    MAX_GRASS_NUM = 100
    GRASS_GENERATION_RATE = 5  # if exceeding MAX_GRASS_NUM replace GRASS_GENERATION_RATE grass points
    GRASS_GROWTH_CHANCE = 0.5  # maybe will be useful to create droughts
    MAX_LEAVES_NUM = 50
    LEAVES_GENERATION_RATE = 0  # if exceeding MAX_LEAVES_NUM replace LEAVES_GENERATION_RATE grass points
    LEAVES_GROWTH_CHANCE = 0.5  # maybe will be useful to create droughts

    # Food parameters
    LEAF_HEIGHT = 10
    FOOD_DISTANCE_THRESHOLD = 75
    FOOD_SIZE = FOOD_DISTANCE_THRESHOLD / 2  # for display

    # Energy parameters
    INIT_MAX_ENERGY = 20000  # maybe useful for maturity test before reproduction
    REPRODUCTION_ENERGY = 10000  # energy cost of reproduction
    MIN_LIFE_ENERGY = 5000  # energy to be left after reproduction
    GRASS_ENERGY = 500
    LEAF_ENERGY = 2000
    INIT_DIGEST_DICT = {'grass': 1, 'leaf': 0.5, 'creature': 0}

    for c_digest in INIT_DIGEST_DICT.values():
        assert 0 <= c_digest <= 1, 'Config Error: INIT_DIGEST_DICT is not set correctly (values between 0-1).'

    # Creatures parameters
    NUM_CREATURES = 800  # init size of population
    MAX_NUM_CREATURES = 1250
    INIT_MAX_AGE = 15000  # [sec]
    ADOLESCENCE_AGE_FRACTION = 0.25  # [0-1] fraction of INIT_MAX_AGE for adulthood
    REPRODUCTION_COOLDOWN = 300  # [sec] minimal time between reproduction events
    INIT_MAX_MASS = 10  # [kg]
    INIT_MAX_HEIGHT = 0.5  # [m]
    INIT_MAX_STRENGTH = 25  # [N]
    MAX_SPEED = 2.5  # [m/sec] maximum speed of the creature

    # Eyes parameters
    # Define eye parameters: (angle_offset in radians, aperture in radians)
    # eyes_params = ((np.radians(30), np.radians(45)),(np.radians(-30), np.radians(45)))
    EYE_CHANNEL = ['grass']  # ['grass', 'leaves', 'water', 'creatures']
    EYES_PARAMS = [
        [np.radians(0), np.radians(90)]
    ]  # list of lists - angle_offset, aperture
    VISION_LIMIT = 1000  # maximum distance that the creature can see
    NOISE_STD = 0.5  # gaussian noise for vision - currently not in use

    # Brain parameters
    BRAIN_TYPE = 'graphic_brain'  # 'fully_connected_brain' or 'graphic_brain'
    # 2 for hunger and thirst, 1 for speed, 3 (flag, distance, angle) for each eye * X channels
    INPUT_SIZE = 2 + 1 + 3 * len(EYES_PARAMS) * len(EYE_CHANNEL)
    # d_velocity and d_angle
    OUTPUT_SIZE = 2
    NORM_INPUT = np.array([1, 1, 1])
    for _ in range(len(EYES_PARAMS) * len(EYE_CHANNEL)):
        NORM_INPUT = np.append(NORM_INPUT, [1, VISION_LIMIT, 1])

    # Mutations
    MUTATION_CHANCE = 0.3  # number between 0-1 indicating chance of trait to be mutated
    STD_MUTATION_FACTORS = {
        'color': np.ones(3) * 0.05,  # +-in each RGB color

        'max_age': 2,
        'max_mass': 1,
        'max_height': 1,
        'max_strength': 2,
        'max_speed': 0.1,
        'max_energy': 0,

        'digest_dict': {'grass': 0.1, 'leaf': 0.1, 'creature': 0.05},
        'reproduction_energy': 0,

        'eyes_params': np.radians(5),  # +-degrees for each eye for angle and width
        'vision_limit': 4,
    }
    MUTATION_FC_BRAIN = {'layer_addition': 0.1,
                      'modify_weights': 0.2,
                      'modify_layer': 0.2,
                      'modify_activation': 0.1}
    MUTATION_GRAPH_BRAIN = {'add_node': 0.1,
                            'remove_node': 0.1,
                            'modify_edges': 0.7,  # chance to perform a change to the weights
                            'modify_edges_percentage': 0.5,  # percentage of edges to change
                            'add_edge': 0.3,
                            'remove_edge': 0.1,
                            'change_activation': 0.1,
                            'forget_magnitude': 10,
                            'add_loop': 0.2,
                            'break_edge': 0.5
                            }
    if BRAIN_TYPE == 'fully_connected_brain':  # 'fully_connected_brain' or 'graphic_brain'
        MUTATION_BRAIN = MUTATION_FC_BRAIN
    elif BRAIN_TYPE == 'graphic_brain':
        MUTATION_BRAIN = MUTATION_GRAPH_BRAIN

    # Filepaths
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    OUTPUT_FOLDER = project_folder.joinpath('outputs').joinpath(date_str)
    OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)

    hour_str = now.strftime('%H-%M-%S')  # change this to a different string to create a new output file
    ANIMATION_FILEPATH = OUTPUT_FOLDER.joinpath(f"{date_str}_T_{hour_str}_simulation.mp4")
    SPECIFIC_FIG_FILEPATH = OUTPUT_FOLDER.joinpath(f"{date_str}_T_{hour_str}_specific_fig.png")
    STATISTICS_FIG_FILEPATH = OUTPUT_FOLDER.joinpath(f"{date_str}_T_{hour_str}_statistics_fig.png")
    ENV_FIG_FILE_PATH = OUTPUT_FOLDER.joinpath(f"{date_str}_T_{hour_str}_env_fig.png")
